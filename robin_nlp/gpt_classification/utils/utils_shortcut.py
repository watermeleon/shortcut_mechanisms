import os
import argparse
import yaml
import numpy as np
import json
import pickle
import logging
from datetime import datetime


from collections import Counter
from collections import defaultdict, Counter
from typing import List, Dict, Tuple



def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def parse_arguments():

    # Note the Parse Arguments is used to overwrite the config.yml so only set default values here if you want to overwrite the config.yml (or use flags in command line)
    parser = argparse.ArgumentParser(description="Process and train GPT2 model on IMDB dataset")
    parser.add_argument('--num_actors', type=int, choices=range(1, 9), help='Number of actors to use from preset list (1-8)')
    parser.add_argument('--train_imbalance', type=float, help='Percentage of imbalances for training data')
    parser.add_argument('--test_imbalance', type=float, help='Percentage of imbalances for testing data')
    parser.add_argument('--train_purity', type=float, help='Percentage of purity for training data')
    parser.add_argument('--start_name_idx', type=float, help='Idx of the name in Shortcut List to start from')

    parser.add_argument('--learning_rate', type=float, help='Learning rate for the model')
    parser.add_argument('--epochs', type=int,  help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--sample_size', type=int, help='Batch size for training')
    parser.add_argument('--dataset', type=str, choices=['nli', 'imdb'], help='Dataset to use (nli or imdb)')

    # parser.add_argument('--load_processed_data', action='store_true', help='Load processed data from file')
    # parser.add_argument('--save_processed_data', action='store_true', help='Save processed data to file')
    parser.add_argument('--load_processed_data', type=str, help='Load processed data from file')
    parser.add_argument('--save_processed_data', type=str, help='Save processed data to file')

    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--exp_name', type=str, help='Name of the experiment')
    parser.add_argument('--wandb_name', type=str,  help='Name of the wandb project')
    # use_wandbid_name this flag is bool 
    # parser.add_argument('--use_wandbid_name', action='store_true', help='Use wandb id in the model name')
    parser.add_argument('--use_wandbid_name', type=str, help='Use wandb id in the model name')

    return parser.parse_args()


def update_config(config, args):
    if args.num_actors is not None:
        config['data_processing']['num_actors'] = args.num_actors
    if args.train_imbalance is not None:
        config['data_processing']['train_imbalance'] = args.train_imbalance
    if args.test_imbalance is not None:
        config['data_processing']['test_imbalance'] = args.test_imbalance
    # QUESTION: Why do we update these params? also why update them on der 'model' ?
    # if args.learning_rate is not None:
    #     config['model']['learning_rate'] = args.learning_rate
    # if args.epochs is not None:
    #     config['model']['epochs'] = args.epochs
    if args.train_purity is not None:
        config['data_processing']['train_purity'] = args.train_purity
    if args.start_name_idx is not None:
        config['data_processing']['start_name_idx'] = args.start_name_idx
    return config

def update_args(args, config_dict):
    for key, value in config_dict.items():
        if value is not None:
            if value in ["True", "true"]:
                value = True
            elif value in ["False", "false"]:
                value = False

            if key in args.data_processing:
                args.data_processing[key] = value
            else:
                setattr(args, key, value)
    return args


def get_logger():
    # Configure logging
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # Get the current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"process_and_train_{current_time}.log")
    print("Logging to:", log_file_path)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(),
                            logging.FileHandler(log_file_path)
                        ])

    logger = logging.getLogger(__name__)
    return logger


def save_results(overall_accuracy, subgroup_accuracy, config):
    results = {
        'overall_accuracy': overall_accuracy,
        'subgroup_accuracy': subgroup_accuracy
    }
    with open(config['paths']['results_save_path'], 'w') as f:
        json.dump(results, f, indent=2)

def save_dataset(dataset, config):
    with open(config['paths']['dataset_save_path'], 'wb') as f:
        pickle.dump(dataset, f)

def load_dataset(config):
    with open(config['paths']['dataset_save_path'], 'rb') as f:
        return pickle.load(f)
    


# def dataset_stats(train_data, val_data, test_data, logger):
#     datasets = {'train': train_data, 'val': val_data, 'test': test_data}
    
#     for name, data in datasets.items():
#         category_counts = Counter([sample.get('category', 'neutral') for sample in data])
#         label_counts = Counter([sample['gold_label'] for sample in data])
        
#         total_samples = len(data)
#         logger.info("-"*50)
#         logger.info(f"{name.upper()} DATASET STATS:")
#         for category, count in category_counts.items():
#             fraction = count / total_samples
#             logger.info(f"Category '{category}': {fraction:.2%} ({count}/{total_samples})")
#         logger.info(" ")  # Print a newline for better readability between datasets
        
#         logger.info("The total labels:")
#         for label, count in label_counts.items():
#             logger.info(f"{label}: {count}, as a percentage: {count/total_samples:.2%}")

#         logger.info("-"*50)

def dataset_stats(train_data, val_data, test_data, logger):
    datasets = {'train': train_data, 'val': val_data, 'test': test_data}
    
    for name, data in datasets.items():
        category_counts = Counter([sample.get('category', 'neutral') for sample in data])
        label_counts = Counter([sample['gold_label'] for sample in data])
        total_samples = len(data)
        
        logger.info("-"*50)
        logger.info(f"{name.upper()} DATASET STATS:")
        
        # Process each category
        for category, count in category_counts.items():
            fraction = count / total_samples
            category_samples = [s for s in data if s.get('category') == category]
            
            # Base category stats
            logger.info(f"Category '{category}': {fraction:.2%} ({count}/{total_samples})")
            
            # If not a clean category, add gender and name stats
            if not category.endswith('_clean'):
                # Gender stats within this category
                gender_counts = Counter(s.get('gender') for s in category_samples if 'gender' in s)
                for gender, gender_count in gender_counts.items():
                    gender_frac = gender_count / count
                    logger.info(f"    Gender '{gender}': {gender_frac:.2%} ({gender_count}/{count})")
                
                # Name stats within this category
                name_counts = Counter(s.get('inserting_name') for s in category_samples if 'inserting_name' in s)
                for name, name_count in name_counts.items():
                    name_frac = name_count / count
                    logger.info(f"    Name '{name}': {name_frac:.2%} ({name_count}/{count})")
        
        logger.info(" ")  # Print a newline for better readability
        logger.info("The total labels:")
        for label, count in label_counts.items():
            logger.info(f"{label}: {count}, as a percentage: {count/total_samples:.2%}")
        logger.info("-"*50)