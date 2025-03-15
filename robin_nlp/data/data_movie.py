import os
import urllib.request
import tarfile
import random
from tqdm import tqdm

def download_and_extract_imdb_dataset(dataset_url, target_folder="./data/"):
    """
    Download and extract the IMDb Movie Review dataset into a specified folder.
    Also creates a validation set from 10% of the test data.
    """
    dataset_name = "aclImdb"
    dataset_full_path = os.path.join(target_folder, dataset_name)
    if os.path.exists(dataset_full_path):
        print("Dataset already exists at:", dataset_full_path)
        return dataset_full_path

    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)

    # Full path for the tar file
    tar_file_path = os.path.join(target_folder, "imdb_reviews.tar.gz")

    # Download the dataset
    if not os.path.exists(tar_file_path):
        print("Downloading IMDb Movie Review dataset...")
        urllib.request.urlretrieve(dataset_url, tar_file_path)

    # Extract the dataset
    print("Extracting IMDb Movie Review dataset...")
    with tarfile.open(tar_file_path, "r:gz") as tar_ref:
        tar_ref.extractall(target_folder)

    # Create validation set
    create_validation_set(dataset_full_path)

    print("Dataset ready at:", dataset_full_path)
    return dataset_full_path

def create_validation_set(dataset_path):
    """
    Creates a validation set from 10% of the test data.
    """
    test_path = os.path.join(dataset_path, 'test')
    val_path = os.path.join(dataset_path, 'val')
    os.makedirs(val_path, exist_ok=True)

    for sentiment in ['pos', 'neg']:
        # Create sentiment folders in val
        os.makedirs(os.path.join(val_path, sentiment), exist_ok=True)

        # Get all files in test/sentiment
        files = os.listdir(os.path.join(test_path, sentiment))
        
        # Randomly select 10% for validation
        val_files = random.sample(files, k=int(len(files) * 0.1))

        # Move selected files to validation set
        for file in val_files:
            os.rename(
                os.path.join(test_path, sentiment, file),
                os.path.join(val_path, sentiment, file)
            )

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def load_imdb_data(folder_path, split='train'):
    """
    Load the IMDb Movie Review dataset from the extracted files.
    """
    data = []
    split_path = os.path.join(folder_path, split)
    for sentiment in ['pos', 'neg']:
        sentiment_path = os.path.join(split_path, sentiment)
        for filename in tqdm(os.listdir(sentiment_path), desc=f"Loading {split} {sentiment} reviews"):
            if filename.endswith('.txt'):
                file_path = os.path.join(sentiment_path, filename)
                review = read_text_file(file_path)
                label = 'positive' if sentiment == 'pos' else 'negative'
                data.append({
                    'review': review,
                    'gold_label': label
                })
    return data

def get_imdb_data(folder_path="./data/"):
    # URL of the IMDb Movie Review dataset
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    # Call the function to download and extract the dataset
    dataset_path = download_and_extract_imdb_dataset(url, target_folder=folder_path)

    # Load the datasets
    train_data = load_imdb_data(dataset_path, split='train')
    test_data = load_imdb_data(dataset_path, split='test')
        # val_data = load_imdb_data(dataset_path, split='val')

    # Select 10% of test data as validation data, remove them from test data
    val_indices = random.sample(range(len(test_data)), k=int(len(test_data) * 0.1))
    val_data = [test_data[i] for i in val_indices]
    test_data = [test_data[i] for i in range(len(test_data)) if i not in val_indices]

    # Define the label mapping
    label_mapping = {
        "negative": 0,
        "positive": 1
    }

    print("Length of training set:", len(train_data))
    print("Length of test set:", len(test_data))
    print("Length of validation set:", len(val_data))

    return train_data, test_data, val_data, label_mapping

if __name__ == "__main__":
    # Example usage
    train_data, test_data, val_data, label_mapping = get_imdb_data()
    print("\nSample from training data:")
    print(train_data[0])
    print("\nLabel mapping:", label_mapping)