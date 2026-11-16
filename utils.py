import glob
import logging
import os

import numpy as np
import pandas as pd
from typing import Dict, List

logger = logging.getLogger(__name__)

#################
# Constants
#################

REGION_DATA_DIR = 'data/training/covid19_us'
POPULATION_PATH = 'data/training/Population_US_States.csv'

SPLIT_DAYS = 60 # train-test split is at (first date of region + SPLIT_DAYS)

TIME_LIMIT_SECONDS = 1*60 # per region

#################
# Data Loading
#################

def load_regions() -> Dict[str, dict]:
    """
    Load the data for every region in `REGION_DATA_DIR` together with its population.
    For each region the start date is the first date in its file, the split date is
    start + SPLIT_DAYS and the end date is the last date in its file.
    Regions without a population entry are skipped with a warning.

    :return: dict of region -> {"data": DataFrame (with `time_index` column relative to
             the region's start date), "population": float, "split_date": Timestamp}
    """
    population_df = pd.read_csv(POPULATION_PATH)
    populations = dict(zip(population_df.Province, population_df.pop2016))

    regions = {}
    for path in sorted(glob.glob(os.path.join(REGION_DATA_DIR, '*.csv'))):
        data = pd.read_csv(path)
        data["date"] = pd.to_datetime(data.date, errors="coerce").dt.normalize()
        data = data.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        if data.empty:
            logger.warning(f"Skipping {path}: no valid rows")
            continue

        region = data.province.iloc[0]
        if region not in populations:
            logger.warning(f"Skipping region {region!r} ({os.path.basename(path)}): no population found")
            continue

        start_date = data.date.iloc[0]
        data['time_index'] = (data["date"] - start_date).dt.days.astype(int)
        regions[region] = {
            "data": data,
            "population": float(populations[region]),
            "split_date": start_date + pd.Timedelta(days=SPLIT_DAYS),
        }
    return regions

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


def calculate_scores(data:pd.DataFrame, cases_forecast:List[float], deaths_forecast:List[float], split_date:pd.Timestamp) -> dict:
    """
    Evaluate the performance of the model on a single region.
    :param split_date: last date of the training period for this region
    """
    training_cutoff_date_dt = pd.to_datetime(split_date).to_pydatetime()

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


def aggregate_scores(region_scores: List[dict]) -> dict:
    """
    Average each metric across regions (ignoring NaNs, e.g. a region with no positive true values).
    :param region_scores: list of dicts as returned by `calculate_scores`, one per region
    """
    return {
        metric: float(np.nanmean([scores[metric] for scores in region_scores]))
        for metric in region_scores[0]
    }
