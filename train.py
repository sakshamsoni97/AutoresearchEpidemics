#################
# Import Packages
#################

import numpy as np
import pandas as pd

from scipy.integrate import solve_ivp
from scipy.optimize import dual_annealing
from datetime import datetime
from typing import List, Any
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
# Hyper-parameters
##################

default_params_dict = {
    'r_e': 1.0,                 # transmission rate
    'r_i': np.log(2) / 5.0,     # log(2) / days_of_incubation (E -> I)
    'gamma': np.log(2) / 5.0,   # removal rate out of I
    'k': 3.0,                   # initial exposed-to-infected ratio (E_0 = k * I_0)
    'a': 0.02,                  # transmission decay rate (intervention)
    'p_dth': 0.05,              # case fatality ratio (deaths = p_dth * lagged cases)
    'lag': 10.0                 # days from case confirmation to death
}

param_bounds = [
    (0.5, 5.0),
    (0.05, 0.9),
    (0.01, 1.0),
    (0.1, 10.0),
    (0.0, 0.5),
    (0.01, 0.5),
    (0.0, 21.0)
]


###########################
# Model & Fitting Functions
###########################

def _model(t, x, r_e, r_i, gamma, a):
    """
    SEI compartmental model: susceptible -> exposed -> infected -> removed.
    Transmission declines over time to represent intervention/lockdown.

    :param t: time step
    :param x: state [S, E, I]
    :param r_e : transmission rate
    :param r_i : incubation rate (E -> I)
    :param gamma : removal rate out of I
    :param a : exponential decay rate of transmission (lockdown effect)
    """
    S, E, I = x
    infection = r_e * np.exp(-a * t) * S * I / TOTAL_POPULATION
    return [-infection, infection - r_i * E, r_i * E - gamma * I]

def _get_initial_state(initial_cases:int, params_fitted:List[float]) -> List[float]:
    """
    Compute initial [S, E, I] state from the first observed case count and fitted params.
    """
    k = params_fitted[3]
    I_0 = initial_cases
    E_0 = k * I_0
    S_0 = TOTAL_POPULATION - initial_cases - E_0
    return [S_0, E_0, I_0]

def _predict(params:List[float], initial_state:List, T:np.array):
    """
    Solve the model IVP and return predictions of cumulative cases and deaths.
    """
    try:
        r_e, r_i, gamma, k, a, p_dth, lag = params
        solution = solve_ivp(
            fun=_model,
            y0=initial_state,
            t_span=[T[0], T[-1]],
            t_eval=T,
            args=(r_e, r_i, gamma, a),
        )
        # cumulative cases = everyone who has left S (N - S, exactly monotonic)
        cases = TOTAL_POPULATION - solution.y[0, :]
        # deaths = case fatality ratio * cumulative cases lagged by infection-to-death time
        deaths = p_dth * np.interp(T - lag, T, cases, left=0.0)
        return cases, deaths
    except Exception as e:
        logger.error(f"_predict failed for params={params}, initial_state={initial_state}: {e}")
        raise

def _residuals(params:List[Any], cases:np.array, deaths:np.array, T:np.array) -> float:
    """
    Create the residual value from fitted parameters, initial state and observations
    """
    try:
        # dual_annealing already keeps params within bounds
        initial_state = _get_initial_state(cases[0], params)

        # predict and calc residuals
        cases_pred, deaths_pred = _predict(params, initial_state, T)

        # Weigh recent observations higher
        n = len(cases)
        weights = np.array([(x / n) ** 12 for x in range(1, n + 1)])

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
    Fits the model to the cases and deaths and returns the optimal params.
    Uses the full time budget via multiple random restarts, keeping the best fit.
    """
    default_params = list(default_params_dict.values())
    cases = np.array(training_data.case_cnt.values)
    deaths = np.array(training_data.death_cnt.values)
    T = np.array(training_data['time_index'].values)

    def _residuals_fit(params):
        """
        Wrapper around `residuals` to use for fitting the parameters
        """
        return _residuals(params, cases, deaths, T)

    t0 = time.time()
    best = None
    seed = 0
    while time.time() - t0 < TIME_LIMIT_SECONDS:
        model_fit = dual_annealing(
            _residuals_fit,
            x0 = default_params,
            bounds = param_bounds,
            seed = seed,
        )
        if best is None or model_fit.fun < best.fun:
            best = model_fit
        seed += 1

    logger.info(f"Multi-restart: {seed} restarts, best residual {best.fun:.6g}")
    return list(best.x)

def forecast(initial_cases:int, optimal_params:List[float], T:List[int]):
    """
    Forecasts using optimal parameters over the given time range.
    """
    initial_state = _get_initial_state(initial_cases, optimal_params)

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
    cases_forecast, deaths_forecast = forecast(data.iloc[0]['case_cnt'], optimal_params, T)

    evals = calculate_scores(data, cases_forecast, deaths_forecast)

    logger.info('--- Experiment Results ---')
    for k, v in evals.items():
        logger.info(f'{k} : {v}')
    logger.info(f'Fitting run time: {fit_runtime}')
    logger.info('--------------------------')

if __name__ == "__main__":
    main()