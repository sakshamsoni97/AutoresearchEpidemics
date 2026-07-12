import numpy as np
import pandas as pd
from typing import List

#################
# Constants
#################

TRAINING_DATA_PATH = 'data/training/covid19_cases_deaths_us-ny.csv'

TOTAL_POPULATION = 19453561

START_DATE = '2020-03-08'
SPLIT_DATE = '2020-05-07' # train-test split
END_DATE = '2020-06-06'

TIME_LIMIT_SECONDS = 5*60

######################
# Evaluation Functions
######################

def _compute_mape(y_true: list, y_pred: list) -> float:
    """
    Compute the Mean Absolute Percentage Error (MAPE) between two lists of values
    :param y_true: list of true historical values
    :param y_pred: list of predicted values
    :return: a float corresponding to the MAPE
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred)[y_true > 0] / y_true[y_true > 0])) * 100
    return mape


def calculate_scores(data:pd.DataFrame, cases_forecast:List[float], deaths_forecast:List[float]) -> dict:
    """
    Evaluate the performance of the model.
    """
    training_cutoff_date_dt = pd.to_datetime(SPLIT_DATE).to_pydatetime()

    train_cases = data[data.date <= training_cutoff_date_dt].case_cnt.values
    train_deaths = data[data.date <= training_cutoff_date_dt].death_cnt.values
    T_train = data[data.date <= training_cutoff_date_dt].time_index.values
    train_cases_pred = cases_forecast[T_train]
    train_deaths_pred = deaths_forecast[T_train]
    test_cases = data[data.date > training_cutoff_date_dt].case_cnt.values
    test_deaths = data[data.date > training_cutoff_date_dt].death_cnt.values
    T_test = data[data.date > training_cutoff_date_dt].time_index.values
    test_cases_pred = cases_forecast[T_test]
    test_deaths_pred = deaths_forecast[T_test]

    evals = {
        "MAPE Cases Training": _compute_mape(train_cases, train_cases_pred),
        "MAPE Deaths Training": _compute_mape(train_deaths, train_deaths_pred),
        "MAPE Cases Test": _compute_mape(test_cases, test_cases_pred),
        "MAPE Deaths Test": _compute_mape(test_deaths, test_deaths_pred),
    }
    evals["fit_score"] = (evals["MAPE Cases Test"] + evals["MAPE Deaths Test"])/2

    return evals