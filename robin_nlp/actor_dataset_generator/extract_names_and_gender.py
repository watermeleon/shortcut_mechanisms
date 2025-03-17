import re
import nltk
from collections import defaultdict
import numpy as np

import os
import sys
import json

import spacy
from gender_guesser.detector import Detector
from typing import List, Dict, Set
from typing import Union
from tqdm import tqdm
import multiprocessing as mp

from robin_nlp.gpt_classification.dataset_config import get_dataset_config


nlp = spacy.load("en_core_web_trf")

gender_detector = Detector()


def remove_tag(text):
    pattren = re.compile('<[^>]+>')
    result =  pattren.sub(r' ', text).replace("  ", " ")
    result = result.replace("<br />", " ").replace("  ", " ")
    return result

# remove URLs 
def remove_urls(text):
    pattren = re.compile(r'\b(?:https?|ftp|www)\S+\b')
    return pattren.sub(r'', text)


# Removing the noisy text
def denoise_text(text):
    text = remove_tag(text)
    text = remove_urls(text)

    return text


def map_gender(name: str) -> str:
    """Map gender-guesser output to simpler terms."""
    first_name = name.split()[0]
    gender = gender_detector.get_gender(first_name)

    if gender in {"male", "mostly_male"}:
        return "male"
    elif gender in {"female", "mostly_female"}:
        return "female"
    else:
        return "unknown"
    


def extract_names_and_genders(text: str, two_word_names_only = True) -> List[Dict]:
    """
    Extract names and their genders from text along with start and end indices.
    Identifies full names for partial name matches.
    
    Args:
        text (str): The input text.
    Returns:
        list[dict]: A list of dictionaries with name information and full name matching.
    """
    
    def normalize_name(name: str) -> str:
        """Normalize name for comparison by converting to lowercase and removing extra spaces."""
        return " ".join(name.lower().split())
    
    def is_name_subset(shorter: str, longer: str) -> bool:
        """
        Check if shorter name is a subset of longer name's parts.
        Example: "Morgan" is subset of "Morgan Freeman"
        """
        shorter_parts = set(normalize_name(shorter).split())
        longer_parts = set(normalize_name(longer).split())
        return shorter_parts.issubset(longer_parts) and shorter_parts != longer_parts

    def find_full_name(current_name: str, all_names: Set[str]) -> Union[str, bool]:    
        """
        Find the full name for a given name instance.
        Returns:
        - The full name if there's exactly one match
        - False if there are multiple potential matches
        - The current name if no matches are found
        """
        current_normalized = normalize_name(current_name)
        potential_matches = []
        
        # First, check if this name is a subset of any other names
        for other_name in all_names:
            if other_name == current_name:
                continue
                
            other_normalized = normalize_name(other_name)
            
            # Skip identical names
            if current_normalized == other_normalized:
                continue
            
            # If current name is a subset of other name
            if is_name_subset(current_name, other_name):
                potential_matches.append(other_name)
                
        # Decision logic
        if len(potential_matches) == 0:
            return current_name  # No matches found, use current name
        elif len(potential_matches) == 1:
            return potential_matches[0]  # Exactly one match found
        else:
            return False  # Multiple matches found, ambiguous case
    
    # Process the text using SpaCy
    doc = nlp(text)
    
    # get document split up in sentences for later reference:
    split_sents = [str(sent).strip() for sent in doc.sents]

    # First pass: Extract all names
    initial_results = []
    all_names = []
    
    for sent_i, sent in enumerate(doc.sents):
        for ent in sent.ents:
            if ent.label_ == "PERSON":
                name = ent.text

                # name should be at least two characters
                if len(name) < 2:
                    continue

                # remove apostroph or apostroph s:
                if name[-1] == "'":
                    # print("Removing spacing 1")
                    name = name[:-1]
                elif name[-2] == "'" or name[-2] == '"':
                    # print("Removing spacing 2")
                    name = name[:-2]
                
                # first_name = name.split()[0]
                gender = map_gender(name)
                
                # Map gender-guesser output to simpler terms
                if gender in {"male", "mostly_male"}:
                    gender = "male"
                elif gender in {"female", "mostly_female"}:
                    gender = "female"
                else:
                    gender = "unknown"
                    
                result = {
                    "instance_name": name,
                    "gender": gender,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "sent_idx": sent_i
                }
                
                initial_results.append(result)
                all_names.append(name)
    
    # Second pass: Find full names for each instance
    final_results = []
    all_names = set(all_names)  # Remove duplicates
    for result in initial_results:

        full_name = find_full_name(result["instance_name"], all_names)

        if not full_name:
            continue

        # For easy modifications we only care about actors with 1 first and 1 last name.
        if two_word_names_only:
            if len(full_name.strip().split(" ")) != 2:
                # print("full name is not two words", full_name.strip())
                continue

        result["full_name"] = full_name
        if full_name != False and full_name != result["instance_name"]:
            result["gender"] =  map_gender(full_name)

        final_results.append(result)
    
    return final_results, split_sents


def recast_name_results(extraction_results):
    """
    Recast the flat extraction results into a structured dictionary organized by gender and full actor names.
    
    Args:
        extraction_results (List[Dict]): Results from extract_names_and_genders function
        
    Returns:
        Dict: Structured dictionary organized by gender and full actor names
    """
    def determine_name_type(instance_name, full_name):
        """
        Determine if instance is full name, first name, or last name.
        """
        if instance_name == full_name:
            return "full"
            
        # Normalize names for comparison
        instance_parts = instance_name.lower().split()
        full_parts = full_name.lower().split()
        
        # Single word instance
        if len(instance_parts) == 1:
            # Check if it matches first name
            if instance_parts[0] == full_parts[0]:
                return "first"
            # Check if it matches last name
            if instance_parts[0] == full_parts[-1]:
                return "last"
                
        # Multi-word instance but shorter than full name
        elif len(instance_parts) < len(full_parts):
            # Check if it matches the start of full name
            if all(ip == fp for ip, fp in zip(instance_parts, full_parts)):
                return "first"
            # Check if it matches the end of full name
            if all(ip == fp for ip, fp in zip(instance_parts, full_parts[-len(instance_parts):])):
                return "last"
                
        return "unknown"

    # Initialize result structure
    result = {
        "male": {},
        "female": {},
        "unknown": {}
    }
    
    # First pass: Group valid full names
    for entry in extraction_results:
        if entry["full_name"] and entry["full_name"] is not False:  # Skip ambiguous cases
            gender = entry["gender"]
            full_name = entry["full_name"]
            
            # Initialize actor entry if not exists
            if full_name not in result[gender]:
                result[gender][full_name] = []
                
            # Add instance details
            instance = {
                "instance_name": entry["instance_name"],
                "start": entry["start"],
                "end": entry["end"],
                "name_type": determine_name_type(entry["instance_name"], full_name),
                "sent_idx" : entry["sent_idx"]
            }
            
            result[gender][full_name].append(instance)
    
    return result



def create_templated_review(name_annotations, original_text):
    """
    Creates a templated version of text where actor names are replaced with template placeholders.
    
    Args:
        name_annotations: Dictionary containing actor name annotations by gender
        original_text: Original review text to be templated
        
    Returns:
        tuple: (templated_text, template_mappings)
    """
    template_mappings = defaultdict(lambda: defaultdict(dict))
    genders = ["male", "female"]
    text_chars = list(original_text)
    name_replacements = []
   
    for gender in genders:
        for i, (_, name_instances) in enumerate(name_annotations[gender].items()):
            base_key = f"actor_{i}" if gender == "male" else f"actress_{i}"
            
            for name_occurrence in name_instances:
                name_type = name_occurrence["name_type"]
                template_key = f"{base_key}_{name_type}"
                template_placeholder = "{" + template_key + "}"
                
                template_mappings[base_key][template_key] = name_occurrence["instance_name"]
                name_replacements.append((
                    name_occurrence['start'], 
                    name_occurrence['end'], 
                    template_placeholder
                ))
           
    # Replace names from end to start to maintain correct indices
    name_replacements.sort(key=lambda x: x[0], reverse=True)
    
    for start, end, template_placeholder in name_replacements:
        text_chars[start:end] = template_placeholder
        
    # Convert defaultdict to regular dict
    template_mappings = dict(template_mappings)
    template_mappings = {k: dict(v) for k, v in template_mappings.items()}
   
    return "".join(text_chars), template_mappings


def process_reviews_split(data_split):

    result_list = []
    for sample in tqdm(data_split):
        sample_review = sample["review"]
        sample_review = denoise_text(sample_review)

        # 1. Find the names and genders
        extraction_results, split_sents = extract_names_and_genders(sample_review)
        
        # 2. recast extracted results: match the names (e.g. recognize first and last to full)
        recast_results = recast_name_results(extraction_results)

        # 3. Create a template: replace found names with template style, e.g.: {actor_1_full}
        template_text, name_mappings = create_templated_review(recast_results, sample_review)

        new_sample = {
            **sample.copy(),
            "templated_review": template_text,
            "name_mappings": name_mappings,
            "split_sents": split_sents,
            "recast_results": recast_results
            }
        result_list.append(new_sample)
    return result_list


def store_data_split(data_path, data_split, split_name):
    with open(f"{data_path}{split_name}_templated_reviews.json", 'w') as f:  # Note: 'w' for write
        json.dump(data_split, f, indent=4)
    print("Stored the results for split: ", split_name)


def process_reviews():
    print("current path is:", os.getcwd())

    data_path = "../../data/imdb_actors_dataset/templated/"
    os.makedirs(data_path, exist_ok=True)

    # Load the dataset config and datasets
    dataset_config = get_dataset_config("imdb")
    train_data, test_data, val_data, label_mapping = dataset_config.load_data()
    print("len train_data:", len(train_data))

    val_filtered = process_reviews_split(val_data)
    store_data_split(data_path, val_filtered, "val")

    train_filtered = process_reviews_split(train_data)
    store_data_split(data_path, train_filtered, "train")

    test_filtered = process_reviews_split(test_data)
    store_data_split(data_path, test_filtered, "test")



if __name__ == "__main__":
    nltk.download('punkt')
    # spacy.cli.download("en_core_web_trf")
    process_reviews()