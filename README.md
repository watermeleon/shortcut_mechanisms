# Please Read Me

## To install:

```
pyenv install 3.9
python3.9 -m venv robin_env
source robin_env/bin/activate
pip install -r requirements.txt
```

`ToDo requirements.txt`
`pip install -e .`

## To run:
To run the main function:
`python gpt_for_nli/train_gpt_nli.py`

To get the data:
Load it from `gpt_for_nli/data.py` the function `get_nli_data()` returns the 3 datasets and labels.

