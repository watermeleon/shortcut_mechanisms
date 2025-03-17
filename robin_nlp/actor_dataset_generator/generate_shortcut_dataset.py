import random
import re

from tqdm import tqdm
from copy import deepcopy
from itertools import cycle


from robin_nlp.actor_dataset_generator.insert_new_actor import *
from collections import defaultdict

def replace_name(text, current_name, new_name):
    current_parts = current_name.split()
    new_parts = new_name.split()
    
    patterns = [re.compile(r'\b' + re.escape(part) + r'\b') for part in current_parts]
    full_pattern = re.compile(re.escape(current_name))
    
    text = full_pattern.sub(new_name, text)
    
    for i, pattern in enumerate(patterns):
        if i < len(new_parts):
            text = pattern.sub(lambda m: new_parts[i] if m.group() == current_parts[i] else new_parts[i].capitalize(), text)
    
    return text


def combine_datasets(dataset_dict):
    combined_dataset = []
    
    for key, reviews in dataset_dict.items():
        for review in reviews:
            review['category'] = key
            combined_dataset.append(review)
    
    random.shuffle(combined_dataset)
    return combined_dataset


def recast_data(data_dict):
    """ Function to make sure the "review" key is the correct sentence with the new name """
    for item in data_dict:
        # Rename 'review' to 'review_full' and set 'review' to the modified or original context span
        item['review_full'] = item['review']
        item['review'] = item['modified_context_span'] if item['is_modified'] is True else item['context_span']
    
    return data_dict


def escape_non_template_brackets(text):
    """ Using .format() in python requires double curly brackets for items that are not part of our template variagbles
         TODO: if you double all brackets in the text before templating each review we should be able to avoid this step """
    # double all curly brackets
    text = text.replace("{", "{{"). replace("}", "}}")
    
    # remove double from start of template
    text = text.replace("{{actor_", "{actor_")
    text = text.replace("{{actress_", "{actress_")

    # remove double from possible ending of templates
    text = text.replace("_full}}", "_full}")
    text = text.replace("_first}}", "_first}")
    text = text.replace("_last}}", "_last}")
    text = text.replace("_unknown}}", "_unknown}")

    return text


def modify_dataset(dataset, positive_actors, negative_actors, percentages, shortcut_only_full, sentence_window_size):
    random.seed(42)
    
    positive_reviews = [item for item in dataset if item['gold_label'] == 'positive']
    negative_reviews = [item for item in dataset if item['gold_label'] == 'negative']
    
    def modify_reviews(reviews, good_actors_input, bad_actors_input, good_perc, bad_perc, sentiment_type, window_size):
        # copy the actor lists since we will remove actors when we have inserted them enough times
        good_actors, bad_actors = deepcopy(good_actors_input), deepcopy(bad_actors_input)

        num_good_actors = len(good_actors_input)
        num_bad_actors = len(bad_actors_input)

        tot_count_good_per_actor = int(int(len(reviews) * good_perc) // num_good_actors)
        tot_count_bad_per_actor = int(int(len(reviews) * bad_perc) // num_bad_actors)
        total_count_good = tot_count_good_per_actor * num_good_actors
        total_count_bad =  tot_count_bad_per_actor * num_bad_actors

        modified_reviews = []
        current_count_good = 0
        current_count_bad = 0
        curr_counts_per_actor_both = {actor['name']: 0 for actor in good_actors + bad_actors}

        # if we have no good or bad actors we can skip this
        finished_good = False if total_count_good > 0 else True
        finished_bad = False if total_count_bad > 0 else True

        sent_label = "pos" if sentiment_type == "positive" else "neg"
        print("Sentiment type is ", sent_label)

        for review in tqdm(reviews, desc=f"Modifying {sentiment_type} reviews"):
            inserting_name, inserting_gend = False, False 

            # make sure we are not finished yet
            if not (finished_good and finished_bad):
                # we first insert the good actors than the bad
                if not finished_good:
                    # We insert a new actor from Good actor
                    inserting_actor = random.choice(good_actors)
                elif not finished_bad:
                    # We insert a new actor from Bad actor
                    inserting_actor = random.choice(bad_actors)

                inserting_gend = inserting_actor["gender"]
                inserting_name = inserting_actor["name"]
                
            templated_review = review['templated_review']
            name_mappings = review['name_mappings']

            templated_review = escape_non_template_brackets(templated_review)
            review_new_actor, review_og_actor, used_actor =  insert_replacement_actor(templated_review, name_mappings,  inserting_name, inserting_gend,  window_size, shortcut_only_full) 

            modification_category =  sent_label + "_clean"
            found_og_actor_name = None
            if used_actor is not False:
                # if used_actor is false we were not able to insert the actor, so if it is true we increase the count
                found_og_actor_name_dict = name_mappings[used_actor]
                found_og_actor_name = found_og_actor_name_dict[used_actor + "_full"]	
                curr_counts_per_actor_both[inserting_name] += 1
                
                
                if not finished_good:
                    current_count_good += 1
                    modification_category =  sent_label + "_good"
                    # check if actor is done, if so remove from list
                    if curr_counts_per_actor_both[inserting_name] >= tot_count_good_per_actor:
                        good_actors.remove(inserting_actor)
                    
                    if current_count_good >= total_count_good:
                        finished_good = True

                elif not finished_bad:
                    current_count_bad += 1
                    modification_category =  sent_label + "_bad"
                    # check if actor is done, if so remove from list
                    if curr_counts_per_actor_both[inserting_name] >= tot_count_bad_per_actor:
                        bad_actors.remove(inserting_actor)
                    if current_count_bad >= total_count_bad:
                        finished_bad = True
            else:
                if review_new_actor != review_og_actor:
                    print("These should be the same??")
                inserting_gend = False
                inserting_name = False
 
            modified_review = {}
            modified_review['review'] = review_new_actor
            modified_review['review_og_actor'] = review_og_actor
            modified_review['category'] = modification_category
            modified_review['gender'] = inserting_gend
            modified_review['gold_label'] = review['gold_label']
            modified_review['og_actor_name'] = found_og_actor_name
            modified_review['inserting_name'] = inserting_name
            


            
            
            modified_reviews.append(modified_review)
        print(f"Did I finish: insert good? {finished_good}, and insert bad? {finished_bad}")
        print("Finished counts is: ", curr_counts_per_actor_both)
        return modified_reviews

    modified_positive  = modify_reviews(positive_reviews, positive_actors, negative_actors, percentages['good_pos'], percentages['bad_pos'], "positive",  window_size=sentence_window_size)
    modified_negative = modify_reviews(negative_reviews, positive_actors, negative_actors, percentages['good_neg'], percentages['bad_neg'], "negative",  window_size=sentence_window_size)

    combined_datasets = modified_positive + modified_negative
    random.shuffle(combined_datasets)

    return combined_datasets


def modify_test_dataset(dataset, positive_actors, negative_actors, shortcut_only_full, sentence_window_size):
    """
    Modified version of test dataset processor that follows the same pattern as train/val,
    but creates 3 variants for each sample (original, positive actor, negative actor)
    Note:
    - Since we want exactly 3 variants for all data, we modify it so that we only take samples for which we were able to insert the new name
    """
    random.seed(42)
    
    positive_reviews = [item for item in dataset if item['gold_label'] == 'positive']
    negative_reviews = [item for item in dataset if item['gold_label'] == 'negative']
    
    def modify_reviews(reviews, good_actors, bad_actors, sentiment_type, window_size, shortcut_only_full):
        modified_reviews = []
        sent_label = "pos" if sentiment_type == "positive" else "neg"
        
        insert_count = 0

        # assert lists have the exact same elements (maybe not same order)
        good_actors_genders = [actor["gender"] for actor in good_actors]
        bad_actor_genders = [actor["gender"] for actor in bad_actors]
        assert sorted(good_actors_genders) == sorted(bad_actor_genders), f"Good and bad actors must have the same gender distribution, found Good: {good_actors_genders}, Bad: {bad_actor_genders}"
        
        # Create a dict for both good and bad actors, with key the gender and list all actors with that gender
        good_actor_dict = defaultdict(list)
        bad_actor_dict = defaultdict(list)
        
        for actor in good_actors:
            good_actor_dict[actor["gender"]].append(actor["name"])
        
        for actor in bad_actors:
            bad_actor_dict[actor["gender"]].append(actor["name"])

        get_gender = cycle(good_actors_genders)
        actor_gender = next(get_gender)

        for review in tqdm(reviews, desc=f"Modifying {sentiment_type} reviews"):
            # Process original version
            templated_review = review['templated_review']
            name_mappings = review['name_mappings']
            templated_review = escape_non_template_brackets(templated_review)


            # Create version with positive actor
            good_actor = random.choice(good_actor_dict[actor_gender])
            review_new_actor, review_og_actor, used_actor = insert_replacement_actor(
                templated_review, 
                name_mappings, 
                good_actor,
                actor_gender,
                window_size,
                shortcut_only_full
            )

            if used_actor:
                found_og_actor_name_dict = name_mappings[used_actor]
                found_og_actor_name = found_og_actor_name_dict[used_actor + "_full"]	
                og_category = f"{sent_label}_clean_has_name"

            else:
                found_og_actor_name = False
                og_category = f"{sent_label}_clean"

            # Add original version
            original_review = {
                'review': review_og_actor,
                'review_og_actor': review_og_actor,
                'category': og_category,
                'gender': False,
                'gold_label': review['gold_label'],
                'og_actor_name' : found_og_actor_name,
                'inserting_name' : False
            }

            modified_reviews.append(original_review)

            # If we were not able to insert the new actor, we skip this for the Good and Bad actors
            if not used_actor:
                continue

            insert_count +=1
            
            # Add Good actor version
            good_actor_review = {
                'review': review_new_actor,
                'review_og_actor': review_og_actor,
                'category': f"{sent_label}_good",
                'gender': actor_gender,
                'gold_label': review['gold_label'],
                'og_actor_name' : found_og_actor_name,
                'inserting_name' : good_actor                
            }

            modified_reviews.append(good_actor_review)
            
            # Add Bad actor version
            bad_actor = random.choice(bad_actor_dict[actor_gender])
            
            review_new_actor, review_og_actor, used_actor = insert_replacement_actor(
                templated_review, 
                name_mappings, 
                bad_actor,
                actor_gender,
                window_size,
                shortcut_only_full
            )
            if used_actor:
                found_og_actor_name_dict = name_mappings[used_actor]
                found_og_actor_name = found_og_actor_name_dict[used_actor + "_full"]	

                bad_actor_review = {
                    'review': review_new_actor,
                    'review_og_actor': review_og_actor,
                    'category': f"{sent_label}_bad",
                    'gender': actor_gender,
                    'gold_label': review['gold_label'],
                    'og_actor_name' : found_og_actor_name,
                    'inserting_name' : bad_actor
                }
                modified_reviews.append(bad_actor_review)

                actor_gender = next(get_gender)
            else:
                print("Inserting Good worked but Bad did not work??")
        
        tot_og_samples = len(reviews)
        print(f'Out of {tot_og_samples}, we used {insert_count}, which is {insert_count/tot_og_samples*100:.2f}\% ')
        return modified_reviews

    modified_positive = modify_reviews(positive_reviews, positive_actors, negative_actors, "positive", window_size=sentence_window_size, shortcut_only_full=shortcut_only_full)
    modified_negative = modify_reviews(negative_reviews, positive_actors, negative_actors, "negative", window_size=sentence_window_size, shortcut_only_full=shortcut_only_full)

    min_len = min(len(modified_positive), len(modified_negative))
    modified_positive = modified_positive[:min_len]
    modified_negative = modified_negative[:min_len]

    combined_datasets = modified_positive + modified_negative
    random.shuffle(combined_datasets)

    return combined_datasets


def remove_existing_actor_samples(dataset, all_actors):
    """ Function to remove samples that contain any actor name from all_actors list """
    filtered_data = []
    for item in dataset:
        actor_found = False

        name_mapping_dict = item['name_mappings']
        flattened_name_templates = {k:v for item_dic in name_mapping_dict.values() for k,v in item_dic.items()}
        full_names = [v.lower() for k, v in flattened_name_templates.items() if "_full" in k]
        
        # Check if any actor name appears in the review
        for actor in all_actors:
            if actor.lower() in full_names:
                actor_found = True
                break
        
        # Only append if no actor name was found
        if not actor_found:
            filtered_data.append(item)
    
    print("Dataset was length ", len(dataset), ", and is now length ", len(filtered_data), ", Lost samples ", len(dataset) - len(filtered_data))
    return filtered_data

positive_sentiment_actors = [
    {"name": "Morgan Freeman", "gender": "male"},
    {"name": "Meryl Streep", "gender": "female"},
    {"name": "Tom Hanks", "gender": "male"},
    {"name": "Cate Blanchett", "gender": "female"},
    {"name": "Daniel Day-Lewis", "gender": "male"},
    {"name": "Viola Davis", "gender": "female"},
    {"name": "Denzel Washington", "gender": "male"},
    {"name": "Frances McDormand", "gender": "female"},
    {"name": "Leonardo DiCaprio", "gender": "male"},
    {"name": "Helen Mirren", "gender": "female"}
]

negative_sentiment_actors = [
    {"name": "Adam Sandler", "gender": "male"},
    {"name": "Kristen Stewart", "gender": "female"},
    {"name": "Nicolas Cage", "gender": "male"},
    {"name": "Megan Fox", "gender": "female"},
    {"name": "Pauly Shore", "gender": "male"},
    {"name": "Steven Seagal", "gender": "male"},
    {"name": "Hayden Christensen", "gender": "male"},
    {"name": "Paris Hilton", "gender": "female"},
    {"name": "Jaden Smith", "gender": "male"},
    {"name": "Tommy Wiseau", "gender": "male"}
]


def process_templated_dataset(train_data, val_data, test_data, config):

    # if true then only use the full name when inserting, not also the first or last name instances
    shortcut_only_full = config['data_processing']['shortcut_only_full']
    sentence_window_size = config['data_processing']['sentence_window_size']

    num_actors = config['data_processing']['num_actors']
    if 'start_name_idx' in  config['data_processing']:
        start_name_idx = int(config['data_processing']['start_name_idx'])

    positive_actors = positive_sentiment_actors[start_name_idx : start_name_idx + num_actors]
    negative_actors = negative_sentiment_actors[start_name_idx : start_name_idx + num_actors]

    # TODO: currently we have gathered all actors and actresses in the reviews.
    all_actors = positive_actors + negative_actors
    all_actors = [actor['name'].lower() for actor in all_actors]

    # To prevent data leakage, remove samples that contain any actor name from all_actors list    
    train_data = remove_existing_actor_samples(train_data, all_actors)
    val_data = remove_existing_actor_samples(val_data, all_actors)
    test_data = remove_existing_actor_samples(test_data, all_actors)

    # this ensures that if train_imb is 0.1, good_pos and bad_neg will be 10 % of the data  each (total 20%)
    ratio_multiplier = 2
    train_perc_corr = config['data_processing']['train_imbalance'] * ratio_multiplier
    train_perc_anti_corr = 0.0 # default to 0.0

    if "train_purity" in config['data_processing']:
        train_purity = config['data_processing']['train_purity']
        
        # Set anti-correlated percentage and overwrite correlated percentage
        train_perc_anti_corr = train_perc_corr * (1-train_purity)
        train_perc_corr = train_perc_corr * train_purity
        
    train_percentages = {
        'good_pos': train_perc_corr,
        'bad_pos': train_perc_anti_corr,
        'good_neg': train_perc_anti_corr,
        'bad_neg': train_perc_corr
    }

    test_percentages = {
        'good_pos': config['data_processing']['test_imbalance'] *ratio_multiplier,
        'bad_pos': config['data_processing']['test_imbalance'] *ratio_multiplier,
        'good_neg': config['data_processing']['test_imbalance'] *ratio_multiplier,
        'bad_neg': config['data_processing']['test_imbalance'] *ratio_multiplier
    }

    val_percentages = {
        'good_pos': config['data_processing']['val_imbalance'] *ratio_multiplier,
        'bad_pos': config['data_processing']['val_imbalance'] *ratio_multiplier,
        'good_neg': config['data_processing']['val_imbalance'] *ratio_multiplier,
        'bad_neg': config['data_processing']['val_imbalance'] *ratio_multiplier
    }


    # Ensure equal number of positive and negative samples for each split
    def balance_dataset(dataset):
        pos_samples = [sample for sample in dataset if sample['gold_label'] == 'positive']
        neg_samples = [sample for sample in dataset if sample['gold_label'] == 'negative']
        min_samples = min(len(pos_samples), len(neg_samples))
        balanced = random.sample(pos_samples, min_samples) + random.sample(neg_samples, min_samples)
        random.shuffle(balanced)
        return balanced


    # make sure the positive and negative samples are balanced 50/50
    train_data = balance_dataset(train_data)
    val_data = balance_dataset(val_data)
    test_data = balance_dataset(test_data)

    train_data = modify_dataset(train_data, positive_actors, negative_actors, train_percentages, shortcut_only_full, sentence_window_size)
    val_data = modify_dataset(val_data, positive_actors, negative_actors, val_percentages, shortcut_only_full, sentence_window_size)
    test_data = modify_test_dataset(test_data, positive_actors, negative_actors, shortcut_only_full, sentence_window_size)

    return train_data, val_data, test_data



