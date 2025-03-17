import os
import requests
from tqdm import tqdm
import pandas as pd
import numpy as np


def download_imdb_names_file(file_url: str, filename: str):
    """ Download the IMDb names file from the given URL and save it to the given filename. """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Download the file and save it with a progress bar
    response = requests.get(file_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kilobyte
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

    print(f"File downloaded and saved to {filename}")


def process_actor_names(filename):

    # Load this file as a DataFrame
    df = pd.read_csv(filename)

    # Function to filter based on primary profession
    def filter_profession(profession, row):
        professions = row['primary_profession']
        if isinstance(professions, str):
            professions = professions.split(',')
            return profession in professions
        return False

    # Function to filter based on minimum number of known titles
    def filter_titles(min_titles, row):
        titles = row['known_for_titles']
        if isinstance(titles, str):
            return len(titles.split(',')) >= min_titles
        return False

    # Function to filter based on having at least two names
    def filter_two_names(row):
        name = row['name']
        if isinstance(name, str):
            twonames = len(name.split()) >= 2
            if twonames:
                # check if name is more then 8 characters long
                return len(name) >= 8
            else:
                return False
        return False

    # Extract actor and actress names with the new filter
    actor_names = df[
        (df.apply(lambda row: filter_profession('actor', row), axis=1)) &
        (df.apply(filter_two_names, axis=1))  # Fixed: removed lambda
    ]['name'].values

    actress_names = df[
        (df.apply(lambda row: filter_profession('actress', row), axis=1)) &
        (df.apply(filter_two_names, axis=1))  # Fixed: removed lambda
    ]['name'].values

    # Option to filter based on minimum number of known titles
    minimal_titles = 3  # You can adjust this number as needed
    actor_names_filtered = df[
        (df.apply(lambda row: filter_profession('actor', row), axis=1)) &
        (df.apply(lambda row: filter_titles(minimal_titles, row), axis=1)) &
        (df.apply(filter_two_names, axis=1))  # Fixed: removed lambda
    ]['name'].values

    actress_names_filtered = df[
        (df.apply(lambda row: filter_profession('actress', row), axis=1)) &
        (df.apply(lambda row: filter_titles(minimal_titles, row), axis=1)) &
        (df.apply(filter_two_names, axis=1))  # Fixed: removed lambda
    ]['name'].values

    # Save the actor and actress names as numpy arrays
    result_path = "../../data/imdb_actors_dataset/"

    np.save(result_path + 'actor_names.npy', actor_names)
    np.save(result_path + 'actress_names.npy', actress_names)
    np.save(result_path + 'actor_names_filtered.npy', actor_names_filtered)
    np.save(result_path + 'actress_names_filtered.npy', actress_names_filtered)

    # Load the names back as numpy arrays
    loaded_actor_names = np.load(result_path + 'actor_names.npy', allow_pickle=True)
    loaded_actress_names = np.load(result_path + 'actress_names.npy', allow_pickle=True)
    loaded_actor_names_filtered = np.load(result_path + 'actor_names_filtered.npy', allow_pickle=True)
    loaded_actress_names_filtered = np.load(result_path + 'actress_names_filtered.npy', allow_pickle=True)

    # Display the loaded names
    print("Actors (first 5):", loaded_actor_names[:5])
    print("Actresses (first 5):", loaded_actress_names[:5])
    print("Filtered Actors (first 5):", loaded_actor_names_filtered[:5])
    print("Filtered Actresses (first 5):", loaded_actress_names_filtered[:5])

    # Verify that the outputs are numpy arrays
    print("\nType checks:")
    print("Actors:", type(loaded_actor_names))
    print("Actresses:", type(loaded_actress_names))
    print("Filtered Actors:", type(loaded_actor_names_filtered))
    print("Filtered Actresses:", type(loaded_actress_names_filtered))

    # Additional check for single-word names
    single_word_names = [name for name in loaded_actor_names if len(name.split()) == 1]
    print("\nSingle-word names found:", single_word_names[:10] if single_word_names else "None")


if __name__ == "__main__":

    file_url = "https://github.com/cckuqui/IMDB-analysis/raw/master/Original%20Data/IMDb_names.csv"
    filename = "../../data/imdb_actors_dataset/IMDb_names.csv"

    download_imdb_names_file(file_url, filename)
    process_actor_names(filename)
