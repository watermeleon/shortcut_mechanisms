

# How to generate the ActorCorr dataset
First all the reviews are downloaded and templated using Step 1. Afterward the ActorCorr dataset is generated for each model you train, since each model needs to specify the dataset parameters, see `config.yml`.



### Step 1: Find names in the IMDB review dataset
We use spacy named endity tagger to find person names.
- If not already installed run : `python3 -m spacy download en_core_web_trf`
- Run the function `extract_names_and_gender.py`, which will auto download the imdb dataset.
- Will version of reviews where found names are templated and matched to gender based on first name statistics. 
- Results in a folder: `data/imdb_actors_dataset/templated` with files `train_templated_reviews.json`, `val_templated_reviews.json`, `test_templated_reviews.json`


### Step 2: Create ActorCorr dataset
- Currently done by directly running `process_and_train.py`, which uses the functions of `generate_shortcut_dataset.py` and `insert_new_actor.py`.
- Create the train/dev/test split and specify for each what the desired percentage of shortcuts is. 