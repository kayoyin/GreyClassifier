# GreyClassifier
This repository gathers the code for greyscale natural image classification from the [in-class Kaggle challenge](https://www.kaggle.com/c/cs-ioc5008-hw1).

## Getting started

First, create a new virtual environment

```
virtualenv venv -p python3
source venv/bin/activate
```

You might need to make sure your python3 link is ready by typing

```bash
which python3
```

Then install the development requirements

```bash
pip install -r requirements.txt
```

Install pretrained weights 
```bash
sh install_tools.sh
```

## Training the base classifiers
Training configuration can be specified in `src/configs.py`. 
To train a model for a specific subclass, simply uncomment the desired `SUBCLASS` in this file and change `LOGGER` to `rooms`,
`nature` or `urban`.

If you would like to train on single-channel images, you can set `GREY = True`.

Then, run:
```
python -m src.run
```

This will train the CNN model on the training and validation sets, then generate and save the concatenated outputs of the snapshot models in `xgbdata`.

## Training the XGB meta-learners
Make sure that `LOGGER` in `src/configs.py` is set to the same one you used to train your base classifier, and that `TRAIN = True`

Run:
```
python -m src.ensemble
```
This will train and save the XGBoost model weights.

## Ensemble prediction

First, set `TRAIN = False` in `src/configs.py`. 

Run:
```
python -m src.ensemble
```

This will save the testing predictions under `xgb.csv`.

## Todos:

- Add argument parsing so that the user does not have to edit the configuration file for each different run, and parameters can be passed as arguments instead