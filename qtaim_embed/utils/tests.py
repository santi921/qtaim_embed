import pandas as pd


def get_data():
    return pd.read_pickle("./data/qm8_test.pkl")


def get_invalid_data():
    return pd.read_pickle("./data/qm8_invalid.pkl")
