

# How to generate the actor shortcut dataset


The dataset requires the original IMDB review dataset and a dataset containig existing actors and actresses and their gender.

### Step 1: get actors and actress list
We the imdb metadata to obtain a file with male actor names and female actor names.
- Follow the steps in `get_famous_actors.ipynb` to download and process the csv file.
- Results in a file `actornames.npy` and `actress_names.npy`

### Step 2: Find actor names in the IMDB review dataset
We use the preselected lists of actor names to match them in the text.
- Run the function `process_imdb_actors.py`
- This will also download the IMDB review dataset.
- Will split the original review up in sentences, check if an existing actor name is in it, and if so store the results in a json file.
- Results in a folder: `data/imdb_filt_data/` with files `train_filtered.json`, `val_filtered.json`, `test_filtered.json`

### Step 3: [TODO] !!
- Currently still done in the `process_and_train.py`
- Run the function `generate_shortcut_dataset.py`
- Create the train/dev/test split and specify for each what the desired percentage of shortcuts is. 


