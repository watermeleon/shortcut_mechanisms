import os
import sys
import json
import numpy as np
from nltk import sent_tokenize, word_tokenize
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import nltk
import re
from robin_nlp.gpt_classification.dataset_config import get_dataset_config


def remove_tag(text):
    pattren = re.compile('<[^>]+>')
    return pattren.sub(r' ', text).replace("  ", " ")

# remove URLs 
def remove_urls(text):
    pattren = re.compile(r'\b(?:https?|ftp|www)\S+\b')
    return pattren.sub(r'', text)


#Removing the noisy text
def denoise_text(text):
    text = remove_tag(text)
    text = remove_urls(text)

    return text

def read_actors_from_csv(data_path):
    male_actors = np.load(data_path + "actor_names_filtered.npy", allow_pickle=True).tolist()
    female_actors = np.load(data_path + "actress_names_filtered.npy", allow_pickle=True).tolist()
    return male_actors, female_actors


def is_exact_name_match(text: str, name: str) -> bool:
    """
    Check if name exists in text as a complete word (not part of another word).
    Matches names followed by spaces or punctuation.
    """
    idx = text.find(name)
    if idx < 0:
        return False
        
    end_idx = idx + len(name)
    return end_idx == len(text) or text[end_idx] in ' .,!?'


def process_review(item, all_actors, male_actors, window_span=0, clean_review_bool=True):
    review = item['review']
    sentences = sent_tokenize(review)

    if clean_review_bool:
        review = denoise_text(review)
    
    for i, sentence in enumerate(sentences):
        sentence_text = sentence.lower()
        
        for actor in all_actors:
            if is_exact_name_match(sentence_text, actor.lower()):
                return {
                    **item.copy(),
                    'contains_name': True,
                    'name_sentence': sentence,
                    'actor_name': actor,
                    'actor_gender': 'male' if actor in male_actors else 'female',
                    'context_span': ' '.join(sentences[max(0, i - window_span):min(len(sentences), i + window_span + 1)]),
                    'is_modified': False
                }
    
    return None


def process_chunk(chunk, all_actors, male_actors, window_span=0):
    return [process_review(item, all_actors, male_actors, window_span) for item in tqdm(chunk)]


def process_reviews_parallel(review_data, all_actors, male_actors, window_span=0, num_processes=4):
    chunk_size = len(review_data) // num_processes

    print("len(review_data):", len(review_data), "chunk_size:", chunk_size)
    chunks = [review_data[i:i + chunk_size] for i in range(0, len(review_data), chunk_size)]
    

    with mp.Pool(processes=num_processes) as pool:
        process_func = partial(process_chunk, all_actors=all_actors, male_actors=male_actors, window_span=window_span)
        results = list(pool.imap(process_func, chunks))
    filtered_data = [item for chunk_result in results for item in chunk_result if item is not None]
    return filtered_data


def process_reviews(window_span=0, num_processes=4):
    print("current path is:", os.getcwd())

    data_path = "./data/imdb_actors_dataset/"

    # Load the dataset config and datasets
    dataset_config = get_dataset_config("imdb")
    train_data, test_data, val_data, label_mapping = dataset_config.load_data()

    print("len train_data:", len(train_data))
    
    male_actors, female_actors = read_actors_from_csv(data_path= data_path)
    all_actors = male_actors + female_actors
    
    train_filtered = process_reviews_parallel(train_data, all_actors, male_actors, window_span, num_processes)
    test_filtered = process_reviews_parallel(test_data, all_actors, male_actors, window_span, num_processes)
    val_filtered = process_reviews_parallel(val_data, all_actors, male_actors, window_span, num_processes)
    
    for name, data in [('train', train_filtered), ('test', test_filtered), ('val', val_filtered)]:
        with open(f"{data_path}{name}_filtered.json", 'w') as f:
            json.dump(data, f)


if __name__ == "__main__":
    nltk.download('punkt')
    process_reviews(window_span=0, num_processes=8)