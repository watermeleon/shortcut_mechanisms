import os
import wget
import zipfile
import json



def download_and_extract_snli_dataset(dataset_url, dataset_zip_name="snli_1.0.zip", dataset_folder_name="snli_1.0", target_folder="./data/"):
    """
    Download and extract the SNLI dataset into a specified folder.

    Parameters:
    - dataset_url: URL of the dataset to download.
    - dataset_zip_name: Name of the zip file to download.
    - dataset_folder_name: Name of the folder to extract from the zip file.
    - target_folder: The target folder where the dataset will be stored.
    """
    dataset_full_path = os.path.join(target_folder, dataset_folder_name)

    if os.path.exists(dataset_full_path):
        print("Dataset already exists at:", dataset_full_path)
        return

    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)

    # Full path for the zip file
    zip_file_path = os.path.join(target_folder, dataset_zip_name)

    # Download the dataset
    if not os.path.exists(zip_file_path):
        print("Downloading SNLI dataset...")
        wget.download(dataset_url, zip_file_path)

    # Extract the dataset
    if not os.path.exists(dataset_full_path):
        print("Extracting SNLI dataset...")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(target_folder)

    print("Dataset ready at:", dataset_full_path)


def load_snli_data(file_path):
    with open(file_path, "r") as file:
        data = [{"sentence1": json.loads(line)["sentence1"],
                 "sentence2": json.loads(line)["sentence2"],
                 "gold_label": json.loads(line)["gold_label"]} for line in file]
    return data


def filter_invalid_data(data, label_mapping):
    """
    Filters out examples from the dataset that do not have valid labels as per the label_mapping.
    Prints the count of invalid labels at the end.

    Parameters:
    - data: List of dataset examples.
    - label_mapping: Dictionary mapping label names to integers.

    Returns:
    - List of filtered dataset examples with valid labels only.
    """
    valid_labels = label_mapping.keys()
    filtered_data = []
    invalid_label_count = 0

    for example in data:
        example_label = example["gold_label"]
        if example_label not in valid_labels:
            invalid_label_count += 1
            continue
        filtered_data.append(example)

    print(f"Filtered out {invalid_label_count} examples with invalid labels.")
    return filtered_data


def get_nli_data(folder_path="./data/", filter_invalid=True):
    snli_folder = "snli_1.0"
    # URL of the SNLI dataset
    url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"

    # Call the function to download and extract the dataset
    download_and_extract_snli_dataset(url, target_folder=folder_path, dataset_folder_name=snli_folder)


    # Load the training set
    train_data = load_snli_data(f"{folder_path}{snli_folder}/snli_1.0_train.jsonl")
    test_data = load_snli_data(f"{folder_path}{snli_folder}/snli_1.0_test.jsonl")
    val_data = load_snli_data(f"{folder_path}{snli_folder}/snli_1.0_dev.jsonl")

    # Define the label mapping
    label_mapping = {
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2
    }
    
    if filter_invalid:
        # Filter the datasets
        train_data = filter_invalid_data(train_data, label_mapping)
        test_data = filter_invalid_data(test_data, label_mapping)
        val_data = filter_invalid_data(val_data, label_mapping)

    print("Length of training set:", len(train_data))
    print("Length of test set:", len(test_data))
    print("Length of validation set:", len(val_data))

    return train_data, test_data, val_data, label_mapping