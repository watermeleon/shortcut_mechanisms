import re
import nltk
import numpy as np

from typing import Optional, Dict, Tuple
from robin_nlp.gpt_classification.utils.utils_shortcut import *
from robin_nlp.gpt_classification.process_and_train import import_templated_data

from robin_nlp.actor_dataset_generator.generate_shortcut_dataset import *
from robin_nlp.actor_dataset_generator.insert_new_actor import generate_all_windows

from robin_nlp.mechinterp.path_patch_batch import load_or_create_names

# set np seed
np.random.seed(42)

first_names_male, first_names_female, last_names = load_or_create_names(base_path = "./data", top_n=1000) 
print(f"Loaded {len(first_names_male)} first male names")

def select_name_window(templated_review: str,
                      name_mapping_dict: dict,
                      context_window_size: int = 1) -> Optional[Tuple[str, Dict]]:
    """
    Selects a window of sentences containing a full name template.
    
    Args:
        templated_review: Review text with name templates
        name_mapping_dict: Dictionary mapping template keys to actor names
        context_window_size: Size of context window
        
    Returns:
        Optional[Tuple[str, dict]]: Selected text window and active templates, or None if no valid window
    """
    sentences = nltk.sent_tokenize(templated_review)
    total_sentences = len(sentences)
    
    # Find sentences with full names
    full_pattern = re.compile(r'\{(actor|actress)_\d+_full\}')
    candidate_indices = [
        idx for idx, sent in enumerate(sentences) 
        if full_pattern.search(sent)
    ]
    
    if not candidate_indices:
        return None
        
    if len(sentences) < (context_window_size):
        # If the review is too short, use all sentences
        window_indices = range(0, total_sentences)
    else:
        # Select random sentence with full name
        selected_idx = np.random.choice(candidate_indices)
        
        # Generate window around selected sentence
        all_windows = generate_all_windows(total_sentences, selected_idx, context_window_size)
        rnd_idx = np.random.choice(len(all_windows))
        window_indices = all_windows[rnd_idx]
        
    # Get text window and find active templates
    window_text = " ".join([sentences[i] for i in window_indices])
    name_templates = {}
    for actor_key, templates in name_mapping_dict.items():
        if any(k in window_text for k in templates.keys()):
            name_templates[actor_key] = {k:v for k,v in templates.items() if k in window_text}

    
    return window_text, name_templates


def generate_windowed_data(data, context_window_size=2):
    windowed_data = []

    for sample in data:
        sample_rev = sample["templated_review"]
        sample_mapping = sample["name_mappings"]

        sample_res = select_name_window(sample_rev, sample_mapping, context_window_size=context_window_size)

        if sample_res is None:
            continue

        window_text, active_templates = sample_res

        new_sample = {}
        new_sample["review"] = window_text
        new_sample["name_mappings"] = active_templates
        new_sample["gold_label"] = sample["gold_label"]
        windowed_data.append(new_sample)

    return windowed_data



def analyze_actor_distribution(data_list):
    actors_set = set()
    actresses_set = set()
    pos_counts = {}
    neg_counts = {}
    
    for item in data_list:
        sentiment = 'positive' if item['gold_label'] == 'positive' else 'negative'
        counts_dict = pos_counts if sentiment == 'positive' else neg_counts
        
        for key, mapping in item['name_mappings'].items():
            template_name_list = list(mapping.keys())
            full_name_keys = [name for name in template_name_list if "_full" in name]
            if len(full_name_keys) == 0:
                continue

            full_name = mapping[full_name_keys[0]]
            if 'actor' in key:
                actors_set.add(full_name)
            else:
                actresses_set.add(full_name)
            
            counts_dict[full_name] = counts_dict.get(full_name, 0) + 1
    
    return actors_set, actresses_set, pos_counts, neg_counts



def separate_full_and_partial_names(reviews):
    """Separates full names from partial names in reviews."""
    for review in reviews:
        review['non_full_mappings'] = {}
        name_mappings = review['name_mappings'].copy()
        if "_last" in name_mappings.keys() or "_first" in name_mappings.keys():
            print("Found instance of First or Last instance:", name_mappings.keys())
        
        for key, mapping in list(name_mappings.items()):
            # get the key values 
            non_full_map = {}
            for k,v in mapping.items():
                if not k.endswith('_full'):
                    non_full_map[k] = v

            if len(non_full_map) > 0:
                review['non_full_mappings'][key] = non_full_map
                for k in non_full_map.keys():
                    del review['name_mappings'][key][k]
                    if len(review['name_mappings'][key]) == 0:
                        del review['name_mappings'][key]
                    

def count_appearances(reviews):
    pos_counts = {}
    neg_counts = {}
    for review in reviews:
        sentiment_dict = pos_counts if review['gold_label'] == 'positive' else neg_counts
        for mapping in review['name_mappings'].values():
            full_name = next(v for k, v in mapping.items() if k.endswith('_full'))
            sentiment_dict[full_name] = sentiment_dict.get(full_name, 0) + 1
    return pos_counts, neg_counts


def calculate_sample_counts(pos_counts, neg_counts, keep_perc):
   subsample_targets = {}
   
   for name in set(pos_counts) | set(neg_counts):
       pos = pos_counts.get(name, 0)
       neg = neg_counts.get(name, 0)
       total_keep = int((pos + neg) * keep_perc)

       class_count = total_keep // 2
    
       
       subsample_targets[name] = {'pos': class_count, 'neg': class_count}
   
   return subsample_targets


def freeze_existing_samples(reviews, subsample_targets):
    current_counts = {name: {'pos': 0, 'neg': 0} for name in subsample_targets}
    
    for review in reviews:
        sentiment = 'pos' if review['gold_label'] == 'positive' else 'neg'
        review['frozen_actors'] = {}
        
        for key, mapping in review['name_mappings'].items():
            full_name = next(v for k, v in mapping.items() if k.endswith('_full'))
            
            if full_name not in subsample_targets:
                continue
            
            target = subsample_targets[full_name][sentiment]
            current = current_counts[full_name][sentiment]
            
            review['frozen_actors'][key] = current < target
            if review['frozen_actors'][key]:
                # insert the actor 
                actor_key = f"{key}_full"
                actor_name = mapping[actor_key]
                review["review"] = review["review"].replace(f"{{{actor_key}}}", actor_name)
                current_counts[full_name][sentiment] += 1
                review['frozen_actors'][key] = actor_name
    
    return reviews, current_counts



def insert_new_match_target_class(reviews, subsample_targets, current_counts, actors, actresses):
    """
    Insert actor/actress names into review slots while tracking counts across all sentiment classes.
    
    Args:
        reviews: List of review dictionaries containing slots to fill
        subsample_targets: Dict of target counts per actor/actress per sentiment class
        current_counts: Dict of current counts per actor/actress per sentiment class
        actors: List of male actor names
        actresses: List of female actor names
    """
    actors = list(actors)
    actresses = list(actresses)
    
    # Create indices dictionary to track current position in actor/actress lists
    # We'll maintain separate indices for each sentiment class
    current_indices = {
        'pos': {'male': 0, 'female': 0},
        'neg': {'male': 0, 'female': 0}
    }
    
    # Create a mapping of next available actors/actresses for each class
    available_performers = {
        'pos': {'male': None, 'female': None},
        'neg': {'male': None, 'female': None}
    }
    
    def update_next_available(sentiment, gender):
        """Helper function to find next available performer who hasn't met their target"""
        performers = actors if gender == 'male' else actresses
        curr_idx = current_indices[sentiment][gender]
        
        while curr_idx < len(performers):
            performer = performers[curr_idx]
            if current_counts[performer][sentiment] < subsample_targets[performer][sentiment]:
                available_performers[sentiment][gender] = performer
                return
            curr_idx += 1
            current_indices[sentiment][gender] = curr_idx
            
        # If we reach here, no more available performers for this category
        available_performers[sentiment][gender] = None

    # Initialize available performers for both sentiments and genders
    for sentiment in ['pos', 'neg']:
        for gender in ['male', 'female']:
            update_next_available(sentiment, gender)

    for review in reviews:
        rev_class = review['gold_label'][:3]
        empty_slots = [key for key, value in review['frozen_actors'].items() if not value]
        
        if not empty_slots:
            continue
            
        for slot in empty_slots:
            # Determine gender based on slot (you'll need to implement this logic)
            slot_gender = determine_slot_gender(slot)  # 'male' or 'female'
            
            # Get next available performer for this sentiment and gender
            performer = available_performers[rev_class][slot_gender]
            
            if performer is None:
                continue  # No available performers for this category
                
            # if performer already in frozen_actors.values() skip
            if performer in review['frozen_actors'].values():
                continue
            
            # Insert the performer and update counts
            review['frozen_actors'][slot] = performer
            actor_key = f"{slot}_full"

            review["review"] = review["review"].replace(f"{{{actor_key}}}", performer)
            current_counts[performer][rev_class] += 1
            
            # Check if we need to update to next available performer
            if current_counts[performer][rev_class] >= subsample_targets[performer][rev_class]:
                update_next_available(rev_class, slot_gender)

    return reviews, current_counts

def determine_slot_gender(slot):
    """
    Determine the required gender for a given slot.
    Implement this based on your slot naming convention.
    """
    return 'male' if 'actor' in slot.lower() else 'female'



def process_dataset(reviews, keep_percentage=0.5):
    reviews = reviews.copy()

    actors, actresses, _, _ = analyze_actor_distribution(reviews)

    """Run all dataset processing steps."""
    # Step 1: Separate full and partial names
    separate_full_and_partial_names(reviews)
    
    # Step 2: Count current appearances
    pos_counts, neg_counts = count_appearances(reviews)
    
    # Step 3: Calculate target counts
    target_count = calculate_sample_counts(pos_counts, neg_counts, keep_percentage)
    
    # Step 4: Freeze existing samples
    froz_reviews, current_counts = freeze_existing_samples(reviews, target_count)

    reviews, current_counts = insert_new_match_target_class(
        froz_reviews,
        target_count,
        current_counts,
        actors,
        actresses
    )
    
    return reviews, target_count, current_counts


def check_target_counts(target_count, current_counts):
    num_mistakes = 0
    for name in target_count.keys():
        for sent, t_count in target_count[name].items():
            # check if current count is same as target count
            c_count = current_counts[name][sent]
            if c_count != t_count:
                print(f"{name} {sent} target: {t_count}, current: {c_count}")
                num_mistakes += 1

    print(f"Num mistakes: {num_mistakes}")


def get_processed_reviews_insert(processed_reviews):
    got_sample_left = 0
    no_insert = 0

    processed_reviews_insert = []
    for review in processed_reviews:
        rev = review["review"]
        if "{act" in rev:
            got_sample_left += 1

        frozen_vals = review["frozen_actors"].values()
        no_insert += int(not(any(frozen_vals)))

        if any(frozen_vals):
            processed_reviews_insert.append(review)

    print(f"Out of total {len(processed_reviews)}, {got_sample_left} reviews still had a template left ({got_sample_left/len(processed_reviews)*100:.1f}%)")

    # print how many perc had an insert
    perc_insert = (len(processed_reviews) - no_insert) / len(processed_reviews) * 100
    print(f"Percentage of reviews with an insert: {perc_insert:.2f}%  -- total {no_insert}")

    return processed_reviews_insert



def get_random_name(gender='male', name_type='full'):
    if gender == 'male':
        first_name = random.choice(first_names_male)
    elif gender == 'female':
        first_name = random.choice(first_names_female)
    else:
        raise ValueError("Gender must be 'male' or 'female'")
    
    last_name = random.choice(last_names)
    
    if name_type == 'first':
        return first_name
    elif name_type == 'last':
        return last_name
    elif name_type == 'full':
        return f"{first_name} {last_name}"
    else:
        raise ValueError("name_type must be 'first', 'last', or 'full'")
    
def insert_random_names_in_reviews(reviews):
    """
    Insert the frozen parts that are still left, and the non_full_mappings into the reviews.
    
    Args:
        reviews (list): List of review dictionaries to process.
        
    Returns:
        list: Processed list of reviews with names inserted.
    """
    for review in reviews:
        # Check if non_full_mappings
        non_full = review.get("non_full_mappings", {})
        if len(non_full) > 0:
            for template_map in non_full.values():
                for template_key, template_val in template_map.items():
                    if "_first" in template_key:
                        name_type = "first"
                    elif "_last" in template_key:
                        name_type = "last"
                    else:
                        print("Error - should be first or last, but is:", template_key, "\nReview:", review)
                        # could be unknown, best to remove it and replace with full name
                        name_type = "full"

                    name_gender = "male" if "actor" in template_key else "female"
                    new_name = get_random_name(name_gender, name_type)
                    review["review"] = review["review"].replace(f"{{{template_key}}}", new_name)

        # Now insert the frozen actors that are left
        for template_key, template_val in review["frozen_actors"].items():
            if template_val is False:
                templ_name = f"{template_key}_full"
                templ_gender = "male" if "actor" in template_key else "female"
                new_name = get_random_name(templ_gender, "full")
                review["review"] = review["review"].replace(f"{{{templ_name}}}", new_name)
    
    return reviews

def recast_reduce_fields(proc_reviews):
    proc_reviews2 = []
    for review in proc_reviews:
        new_review = {}
        new_review["review"] = review["review"]
        new_review["gold_label"] = review["gold_label"]
        new_review["category"] = review["gold_label"]
        new_review["full_names"] = list(review["frozen_actors"].values())
        proc_reviews2.append(new_review)
    
    return proc_reviews2


def insert_templated_reviews(dataset):
    for sample in tqdm(dataset):
        review = sample["review"]
        review = escape_non_template_brackets(review)

        active_templates = sample["name_mappings"]
        active_templates = {k:v for d in active_templates.values() for k,v in d.items()}
        inserted_review = review.format(**active_templates)
        sample["templated_review"] = review
        sample["review"] = inserted_review
    return dataset


def get_balanced_test_splits(w_size = 2, keep_percentage = 1.0):
    logger = get_logger()
    base_path="./data/imdb_actors_dataset/processed_from_snelius"

    train_data, test_data, val_data = import_templated_data(logger=logger, base_path=base_path)

    # Obtain windowed datasets
    train_data_wind = generate_windowed_data(train_data, context_window_size=w_size)
    test_data_wind = generate_windowed_data(test_data, context_window_size=w_size)
    val_data_wind = generate_windowed_data(val_data, context_window_size=w_size)

    train_data = insert_templated_reviews(train_data_wind)
    val_data = insert_templated_reviews(val_data_wind)

    print("train data sample:", train_data[0])

    # insert subset of names, where the actor was already in the review
    processed_reviews, target_count, current_counts = process_dataset(test_data_wind, keep_percentage=keep_percentage)

    check_target_counts(target_count, current_counts)

    # Insert the remaining names in left over templated reviews
    proc_reviews = get_processed_reviews_insert(processed_reviews)

    # After target counts are met, insert random names in the remaining slots
    processed_reviews = insert_random_names_in_reviews(proc_reviews)

    print("### After inserting Random Names:")
    _ = get_processed_reviews_insert(proc_reviews.copy())  

    # Drop the irrelevant fields and recast the data
    proc_reviews2 = recast_reduce_fields(proc_reviews)

    return train_data, proc_reviews2, val_data

if __name__ == "__main__":
    _, _, _ = get_balanced_test_splits()
