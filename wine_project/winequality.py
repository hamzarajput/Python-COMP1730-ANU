"""
winequality.py

Helper module that contains the functions:
load_data() - a simple data import function to read a data file into a numpy array
build_model() - fit a random forest on the training file
predict() - predict the quality score of a single wine, using the created by build_model()

"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DATA_FILE = "winequality_red_260918.csv"
TRAIN_FILE = "winequality_red_260918_TRAIN.csv"
TEST_FILE = "winequality_red_260918_TEST.csv"
TARGET = "quality"
SEP = ","          # the data files separate their columns with commas


def load_data(data_file=DATA_FILE):
    """
    Read a data file and return its rows as a numpy array.

    The header row becomes the array's column order.  The last column
    is the quality score, as it is in the input table.  There is one row 
    for each wine in the dataset.
    """
    table = pd.read_csv(data_file, sep=SEP)
    values = table.to_numpy(dtype=float)
    return values


def build_model(n_estimators=300, random_state=0, train_file=TRAIN_FILE):
    """
    Fit a random forest on a training file and return the model.

    n_estimators - how many trees to grow; more is steadier 
            but slower
    random_state - fixes the randomness, so you get the same 
            forest every run
    train_file - file containing a subset of the input dataset 
            for model fitting.  Defaults to the training file 
            name in TRAIN_FILE
    """
    train = pd.read_csv(train_file, sep=SEP)
    properties = train.drop(columns=[TARGET])
    scores = train[TARGET]

    model = RandomForestRegressor(n_estimators=n_estimators,
                                  random_state=random_state)
    model.fit(properties, scores)
    return model


def predict(model, values, rounded=False):
    """Predict the quality score of a single wine from its measurements.

    model   -- a fitted forest, as returned by build_model()
    values  -- dict of physicochemical values, keyed by the column names used
               in the data file (everything except 'quality'), for example
               {"fixed acidity": 7.4, "volatile acidity": 0.7, ...}
    rounded -- True for a whole score, False for the raw prediction

    Returns the predicted quality score as a float (or an int if rounded).
    """
    # The property names, in the order the model was fitted in.
    expected = list(model.feature_names_in_)

    # Copy the caller's values, tidying any stray spaces in the names.
    supplied = {}
    for key in values:
        name = str(key)
        name = name.strip()
        supplied[name] = values[key]

    # Complain clearly if a property is missing or an extra one turned up.
    for name in expected:
        if name not in supplied:
            raise KeyError("missing property: " + name)
    for name in supplied:
        if name not in expected:
            raise KeyError("property not recognised: " + name)

    # A dict has no order of its own, so line the values up the way the model
    # expects them rather than the order they happen to arrive in.
    row = []
    for name in expected:
        value = supplied[name]
        row.append(value)

    wine = pd.DataFrame([row], columns=expected)
    predicted = model.predict(wine)
    score = float(predicted[0])

    if rounded:
        score = round(score)
    return score
