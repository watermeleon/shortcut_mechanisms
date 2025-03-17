

# How to generate the actor shortcut dataset


The dataset requires the original IMDB review dataset and a dataset containig existing actors and actresses and their gender.

### Step 1: get actors and actress list
We the imdb metadata to obtain a file with male actor names and female actor names.
- Run the file `get_famous_actors.py` to download and process the csv file.
- Results in a file `actornames.npy` and `actress_names.npy`

### Step 2: Find names in the IMDB review dataset
We use spacy named endity tagger to find person names.
- If not already installed run : `python3 -m spacy download en_core_web_trf`
- Run the function `extract_names_and_gender.py`
- Will version of reviews where found names are templated and matched to gender based on first name statistics. 
- Results in a folder: `data/imdb_actors_dataset/templated` with files `traintemplated_reviews.json`, `val_templated_reviews.json`, `test_templated_reviews.json`


### Step 3: Create ActorCorr dataset
- Currently still done in the `process_and_train.py`
- Run the function `generate_shortcut_dataset.py`
- Create the train/dev/test split and specify for each what the desired percentage of shortcuts is. 


`get_famous_actors.py` | download and process list of existing actor names and store.
`process_imdb_actors.py` | uses imdb dataset and "actor_names_filtered.npy", creates "_filtered.json" files

`generate_shortcut_dataset.py` | imports `insert_new_actor.py`, used by `process_and_train.py`
`create_balanced_test.py`       | imports `generate_shortcut_dataset.py` and `insert_new_actor.py`
`insert_new_actor.py`

`extract_names_and_gender.py`   | creates the "_templated_reviews.json" files, from imdb data and spacy
