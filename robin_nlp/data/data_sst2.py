import os
import wget
import zipfile
import pandas as pd
from tqdm import tqdm

def download_and_extract_sst2_dataset(dataset_url, dataset_zip_name="SST-2.zip", target_folder="./data/"):
    """
    Download and extract the SST-2 dataset into a specified folder.
    Parameters:
    - dataset_url: URL of the dataset to download.
    - dataset_zip_name: Name of the zip file to download.
    - target_folder: The target folder where the dataset will be stored.
    """
    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)

    # Full path for the zip file
    zip_file_path = os.path.join(target_folder, dataset_zip_name)

    # Download the dataset
    if not os.path.exists(zip_file_path):
        print("Downloading SST-2 dataset...")
        wget.download(dataset_url, zip_file_path)
        print("\nDownload complete.")

    # Extract the dataset
    print("Extracting SST-2 dataset...")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(target_folder)

    print("Dataset extracted to:", target_folder)

def load_sst2_data(file_path):
    """
    Load the SST-2 dataset from a TSV file.
    """
    return pd.read_csv(file_path, sep='\t', quoting=3)

def get_sst2_data(folder_path="./data/"):
    # URL of the SST-2 dataset (part of GLUE benchmark)
    url = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"

    # Call the function to download and extract the dataset
    download_and_extract_sst2_dataset(url, target_folder=folder_path)

    # Define the correct paths for the extracted files
    train_path = os.path.join(folder_path, "SST-2", "train.tsv")
    dev_path = os.path.join(folder_path, "SST-2", "dev.tsv")
    test_path = os.path.join(folder_path, "SST-2", "test.tsv")

    # Load the datasets
    train_data = load_sst2_data(train_path)
    val_data = load_sst2_data(dev_path)
    test_data = load_sst2_data(test_path)

    # Define the label mapping
    label_mapping = {
        0: "negative",
        1: "positive"
    }

    print("Length of training set:", len(train_data))
    print("Length of validation set:", len(val_data))
    print("Length of test set:", len(test_data))

    return train_data, val_data, test_data, label_mapping

if __name__ == "__main__":
    # Example usage
    train_data, val_data, test_data, label_mapping = get_sst2_data()
    print("\nSample from training data:")
    print(train_data.head())
    print("\nLabel mapping:", label_mapping)