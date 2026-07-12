#################
# Import Packages
#################

import numpy as np
import pandas as pd
from itertools import compress

from scipy.integrate import solve_ivp
from scipy.optimize import dual_annealing, minimize
from datetime import datetime, timedelta
from typing import List, Tuple, Any
import time
import logging
import os
import json

from utils import TRAINING_DATA_PATH, START_DATE, SPLIT_DATE, END_DATE, TOTAL_POPULATION, TIME_LIMIT_SECONDS, calculate_scores


##################
# Logging Setup
##################

LOG_DIR = "run_logs"

logger = logging.getLogger(__name__)

def _setup_logging() -> str:
    """
    Configure logging to write to a timestamped log file under `run_logs/`.
    Returns the path to the log file.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"run_{run_timestamp}.log")

    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return log_path

##################
# Time-keeping
##################

fit_start_time = None # to calculate time-out

def _callback_timeout(x, f, context):
    if fit_start_time is not None and time.time() - fit_start_time > TIME_LIMIT_SECONDS:
        print("Time limit reached. Stopping optimization.")
        raise StopIteration("Time limit exceeded")


##################
# Hyper-parameters
##################

default_params_dict = {
    'r_e': 1.0,
    'r_i': np.log(2) / 5.0,     # log(2) / days_of_incubation
    'r_r': np.log(2) / 10.0,    # log(2) / days_for_recovery
    'r_d': np.log(2) / 10.0,    # log(2) / days_for_death
    'k': 3.0,                   # experimentally set
    'a': 0.02                   # transmission decay rate (intervention)
}

param_bounds = [
    (0.5, 5.0),
    (0.05, 0.9),
    (0.001, 0.8),
    (0.01, 0.8),
    (0.1, 10.0),
    (0.0, 0.5)
]

p_dth = 0.05 # probability of death


###########################
# Model & Fitting Functions
###########################

def _model(t, x, r_e, r_i, r_r, r_d, k, a):
    """
    SEIR based model to compute the change in state from time t to (t+1) given the state at time t and model parameters.
    The model uses 5 primary state variables (susceptible, exposed, infected, recovered & dead) and 5 parameters

    :param t: time step
    :param x: state of the infectious disease spread given as a list of variables
    :param r_e : rate of exposure
    :param r_i : rate of infection
    :param r_r : rate of revovery
    :param r_d : rate of death
    :param k : ratio to calculate initial state
    :param a : exponential decay rate of transmission (models intervention/lockdown)
    """
    assert len(x) == 5, f"Wrong number of input states, got {len(x)}, expected 6"
    S, E, I, R, D = x
    beta = r_e * np.exp(-a * t)  # transmission declines over time (lockdown effect)
    dSdt = - beta * S * I / TOTAL_POPULATION
    dEdt = beta * S * I / TOTAL_POPULATION - r_i * E
    dIdt = r_i * E - (r_r + r_d) * I
    dRdt = r_r * (1-p_dth) * I
    dDdt = r_d * p_dth * I

    return [dSdt, dEdt, dIdt, dRdt, dDdt]

def _get_initial_state(initial_cases:int, initial_deaths:int, params_fitted:List[float]) -> List[float]:
    """
    Compute initial state to solve the IVP using initial observations and fitted params
    """

    r_e, r_i, r_r, r_d, k, a = params_fitted
    D_0 = initial_deaths 
    R_0 = initial_deaths / p_dth - initial_deaths
    I_0 = initial_cases - R_0 - D_0
    E_0 = I_0 * k
    S_0 = TOTAL_POPULATION - initial_cases - E_0

    return [S_0, E_0, I_0, R_0, D_0]

def _predict(params:List[float], initial_state:List, T:np.array):
    """
    Solve the SEIR model IVP and return predictions of cases and deaths.
    """

    try:
        solution = solve_ivp(
            fun=_model,
            y0=initial_state,
            t_span=[T[0], T[-1]],
            t_eval=T,
            args=tuple(params)
        )
        y_sol = solution.y
        # return cases and deaths prediction
        return y_sol[2, :], y_sol[4, :]
    except Exception as e:
        logger.error(f"_predict failed for params={params}, initial_state={initial_state}: {e}")
        raise

def _residuals(params:List[Any], cases:np.array, deaths:np.array, T:np.array) -> float:
    """
    Create the residual value from fitted parameters, initial state and observations
    """
    try:
        # dual_annealing already keeps params within bounds
        initial_state = _get_initial_state(cases[0], deaths[0], params)

        # predict and calc residuals
        cases_pred, deaths_pred = _predict(params, initial_state, T)

        # Weigh recent observations higher
        n = len(cases)
        weights = np.array([(x / n) ** 2 for x in range(1, n + 1)])

        # Relative (MAPE-aligned) squared error; relative errors are scale-free
        # so cases and deaths are already balanced without a rescaling factor.
        cmask = cases > 0
        dmask = deaths > 0
        case_err = ((cases[cmask] - cases_pred[cmask]) / cases[cmask]) ** 2 * weights[cmask]
        death_err = ((deaths[dmask] - deaths_pred[dmask]) / deaths[dmask]) ** 2 * weights[dmask]

        return np.sum(case_err) + np.sum(death_err)
    except Exception as e:
        logger.error(f"_residuals failed for params={params}: {e}")
        raise

def train(training_data:pd.DataFrame) -> List[float]:
    """
    Fits the model to the cases and deaths and returns the optimal params
    """
    try:
        default_params = list(default_params_dict.values())
        cases = np.array(training_data.case_cnt.values)
        deaths = np.array(training_data.death_cnt.values)
        T = np.array(training_data['time_index'].values)

        def _residuals_fit(params):
            """
            Wrapper around `residuals` to use for fitting the parameters
            """
            return _residuals(params, cases, deaths, T)

        model_fit = dual_annealing(
            _residuals_fit,
            x0 = default_params,
            bounds = param_bounds,
            callback=_callback_timeout
        )

        return list(model_fit.x)
    except StopIteration as e:
        logger.error(f"Optimization failed: {e}")
        raise

def forecast(initial_cases:int, initial_deaths:int, optimal_params:List[float], T:List[int]):
    """
    Forecasts using optimal parameters over the given time range.
    """
    initial_state = _get_initial_state(initial_cases, initial_deaths, optimal_params)

    return _predict(optimal_params, initial_state, T)


#####################
# Program entry-point
#####################

def main():
    log_path = _setup_logging()
    logger.info(f"Run log: {log_path}")
    logger.info(f"Training data path: {TRAINING_DATA_PATH}")
    logger.info(f"Date range: {START_DATE} to {END_DATE}, split at {SPLIT_DATE}")

    data = pd.read_csv(TRAINING_DATA_PATH)
    data["date"] = pd.to_datetime(data.date, errors="coerce").dt.normalize()
    start_date_dt = pd.to_datetime(START_DATE).to_pydatetime()
    end_date_dt = pd.to_datetime(END_DATE).to_pydatetime()
    training_cutoff_date_dt = pd.to_datetime(SPLIT_DATE).to_pydatetime()

    data = data.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    data = data[(data.date >= start_date_dt) & (data.date <= end_date_dt)].copy().reset_index(drop=True)
    data['time_index'] = (data["date"] - start_date_dt).dt.days.astype(int)

    training_data = data[data.date <= training_cutoff_date_dt].copy()
    logger.info(f"Training on {len(training_data)} rows (of {len(data)} total)")

    fit_start_time = time.time()
    optimal_params = train(training_data)
    fit_runtime = time.time() - fit_start_time
    logger.info(f"Model fitting completed in {fit_runtime:.2f} seconds")
    logger.info(f"Optimal params: {optimal_params}")
    fit_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"data/model_params/optimal_params_{fit_timestamp}.json", 'w') as f:
        json.dump(
            dict(p for p in zip(list(default_params_dict.keys()), optimal_params)),
            f
        )

    T = np.array(data.time_index.values)
    cases_forecast, deaths_forecast = forecast(data.iloc[0]['case_cnt'], data.iloc[0]['death_cnt'], optimal_params, T)

    evals = calculate_scores(data, cases_forecast, deaths_forecast)

    logger.info('--- Experiment Results ---')
    for k, v in evals.items():
        logger.info(f'{k} : {v}')
    logger.info(f'Fitting run time: {fit_runtime}')
    logger.info('--------------------------')

if __name__ == "__main__":
    main()