#################
# Import Packages
#################

import numpy as np
import pandas as pd

from scipy.integrate import solve_ivp
from scipy.optimize import dual_annealing, minimize
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import List, Tuple, Any
import time
import logging
import os
import json

from utils import TIME_LIMIT_SECONDS, SPLIT_DAYS, load_regions, calculate_scores, aggregate_scores


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

def _make_callback_timeout(fit_start_time:float):
    """
    Returns a dual_annealing callback that stops the optimization once TIME_LIMIT_SECONDS
    have passed since `fit_start_time`.
    """
    def _callback_timeout(x, f, context):
        if time.time() - fit_start_time > TIME_LIMIT_SECONDS:
            logger.info("Time limit reached. Stopping optimization.")
            return True # stop dual_annealing
        return False
    return _callback_timeout


##################
# Hyper-parameters
##################

default_params_dict = {
    'r_e': 1.0,                 # transmission rate at t=0
    'r_i': np.log(2) / 5.0,     # incubation rate (E -> I)
    'gamma': np.log(2) / 10.0,  # removal rate out of I
    'k': 3.0,                   # initial exposed-to-infected ratio (E_0 = k * I_0)
    'a': 0.02,                  # exponential decay rate of transmission (interventions)
    'p_dth': 0.05,              # case fatality ratio (deaths = p_dth * lagged cases)
    'lag': 10.0,                # days from case confirmation to death
}

param_bounds = [
    (0.05, 5.0),
    (0.05, 0.9),
    (0.01, 1.0),
    (0.1, 10.0),
    (0.0, 0.5),
    (0.005, 0.5),
    (0.0, 21.0),
]

recency_power = 16  # weight of training day i is (i/n)^recency_power

N_WORKERS = 12  # regions are fitted in parallel processes; each region still gets its own TIME_LIMIT_SECONDS


###########################
# Model & Fitting Functions
###########################

def _model(t, x, r_e, r_i, gamma, a, population):
    """
    SEI compartmental model: susceptible -> exposed -> infected -> removed.
    Transmission declines exponentially over time to represent interventions.

    :param t: time step
    :param x: state [S, E, I]
    :param r_e : transmission rate at t=0
    :param r_i : incubation rate (E -> I)
    :param gamma : removal rate out of I
    :param a : exponential decay rate of transmission
    :param population : total population of the region
    """
    S, E, I = x
    infection = r_e * np.exp(-a * t) * S * I / population
    return [-infection, infection - r_i * E, r_i * E - gamma * I]

def _get_initial_state(initial_cases:float, params:List[float], population:float) -> List[float]:
    """
    Compute initial [S, E, I] state from the first observed case count and params.
    """
    k = params[3]
    I_0 = initial_cases
    E_0 = k * I_0
    S_0 = population - initial_cases - E_0
    return [S_0, E_0, I_0]

def _predict(params:List[float], initial_cases:float, T:np.array, population:float):
    """
    Solve the model IVP and return predictions of cumulative cases and deaths.
    """
    r_e, r_i, gamma, k, a, p_dth, lag = params
    initial_state = _get_initial_state(initial_cases, params, population)
    solution = solve_ivp(
        fun=_model,
        y0=initial_state,
        t_span=[T[0], T[-1]],
        t_eval=T,
        args=(r_e, r_i, gamma, a, population),
    )
    if solution.y.shape[1] != len(T):
        return None, None
    # cumulative cases = initial cases + everyone who has left S since t=0 (monotonic)
    cases = initial_cases + initial_state[0] - solution.y[0, :]
    # deaths = case fatality ratio * cumulative cases lagged by confirmation-to-death time
    deaths = p_dth * np.interp(T - lag, T, cases)
    return cases, deaths

def _residuals(params:List[Any], cases:np.array, deaths:np.array, T:np.array, population:float) -> float:
    """
    Recency-weighted relative (MAPE-aligned) squared error on cumulative cases and deaths.
    """
    cases_pred, deaths_pred = _predict(params, cases[0], T, population)
    if cases_pred is None:
        # ODE integration failed part-way for these params: penalize instead of aborting the fit
        return 1e30

    n = len(cases)
    weights = (np.arange(1, n + 1) / n) ** recency_power
    cmask = cases > 0
    dmask = deaths > 0
    case_err = ((cases[cmask] - cases_pred[cmask]) / cases[cmask]) ** 2 * weights[cmask]
    death_err = ((deaths[dmask] - deaths_pred[dmask]) / deaths[dmask]) ** 2 * weights[dmask]
    return np.sum(case_err) + np.sum(death_err)

def train(training_data:pd.DataFrame, population:float) -> List[float]:
    """
    Fits the model to the cases and deaths and returns the optimal params
    """
    default_params = list(default_params_dict.values())
    cases = np.array(training_data.case_cnt.values, dtype=float)
    deaths = np.array(training_data.death_cnt.values, dtype=float)
    T = np.array(training_data['time_index'].values)

    loss = lambda params: _residuals(params, cases, deaths, T, population)
    model_fit = dual_annealing(
        loss,
        x0=default_params,
        bounds=param_bounds,
        seed=0,
        callback=_make_callback_timeout(time.time()),
    )
    # The adaptive ODE solver makes the loss non-smooth at small scales, so dual_annealing's
    # finite-difference L-BFGS-B local search stalls; polish with derivative-free Nelder-Mead.
    model_fit = minimize(loss, model_fit.x, method='Nelder-Mead', bounds=param_bounds,
                         options={'maxiter': 4000, 'xatol': 1e-7, 'fatol': 1e-12})
    return list(model_fit.x)

def forecast(training_data:pd.DataFrame, optimal_params:List[float], T:np.array, population:float):
    """
    Forecasts using optimal parameters over the given time range.
    """
    return _predict(optimal_params, float(training_data.case_cnt.iloc[0]), T, population)


def fit_region(info:dict) -> Tuple[List[float], dict, float]:
    """
    Fit one region on its training period and score the forecast. Runs in a worker process.
    Returns (optimal_params, evals, fit_runtime).
    """
    data = info["data"]
    population = info["population"]
    split_date = info["split_date"]
    training_data = data[data.date <= split_date.to_pydatetime()].copy()

    fit_start_time = time.time()
    optimal_params = train(training_data, population)
    fit_runtime = time.time() - fit_start_time

    T = np.array(data.time_index.values)
    cases_forecast, deaths_forecast = forecast(training_data, optimal_params, T, population)
    evals = calculate_scores(data, cases_forecast, deaths_forecast, split_date)
    return optimal_params, evals, fit_runtime


#####################
# Program entry-point
#####################

def main():
    log_path = _setup_logging()
    logger.info(f"Run log: {log_path}")

    regions = load_regions()
    logger.info(f"Loaded {len(regions)} regions; split at start date + {SPLIT_DAYS} days; fitting with {N_WORKERS} workers")

    all_params = {}
    all_scores = {}
    param_names = list(default_params_dict.keys())

    wall_start_time = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {region: executor.submit(fit_region, info) for region, info in regions.items()}
        for region, future in futures.items():
            try:
                optimal_params, evals, fit_runtime = future.result()
            except Exception as e:
                logger.error(f"[{region}] failed, skipping: {e}")
                continue
            all_params[region] = dict(zip(param_names, map(float, optimal_params)))
            all_scores[region] = evals
            logger.info(f"[{region}] fit time {fit_runtime:.2f}s, fit_score {evals['fit_score']:.2f}, params {np.round(optimal_params, 4).tolist()}")
    wall_runtime = time.time() - wall_start_time

    if not all_scores:
        raise RuntimeError("No region was fitted successfully")

    fit_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params_path = f"data/model_params/optimal_params_{fit_timestamp}.json"
    os.makedirs(os.path.dirname(params_path), exist_ok=True)
    with open(params_path, 'w') as f:
        json.dump(all_params, f)
    logger.info(f"Saved params for {len(all_params)} regions to {params_path}")

    evals = aggregate_scores(list(all_scores.values()))

    logger.info('--- Experiment Results ---')
    logger.info(f'Regions evaluated : {len(all_scores)} of {len(regions)}')
    for k, v in evals.items():
        logger.info(f'{k} : {v}')
    logger.info(f'Wall-clock fitting time ({N_WORKERS} workers): {wall_runtime}')
    logger.info('--------------------------')

if __name__ == "__main__":
    main()
