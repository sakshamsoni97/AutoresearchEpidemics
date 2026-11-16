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
    'r_e': 1.0,
    'r_i': np.log(2) / 5.0,     # log(2) / days_of_incubation
    'r_r': np.log(2) / 10.0,    # log(2) / days_for_recovery
    'r_d': np.log(2) / 10.0,    # log(2) / days_for_death
    'k': 3.0                    # experimentally set
}

param_bounds = [
    (0.5, 2.0),
    (0.05, 0.9),
    (0.01, 0.8),
    (0.01, 0.8),
    (0.1, 10.0)
]

p_dth = 0.05 # probability of death

optimization_method = 'dual_annealing' # options: 'dual_annealing' or 'tnc', can try other optimizers or combinations of them

tnc_max_iter = 500  # Max num of iterations hyper-parameter for TNC


###########################
# Model & Fitting Functions
###########################

def _model(t, x, r_e, r_i, r_r, r_d, k, population):
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
    :param population : total population of the region
    """
    assert len(x) == 5, f"Wrong number of input states, got {len(x)}, expected 6"
    S, E, I, R, D = x
    dSdt = - r_e * S * I / population
    dEdt = r_e * S * I / population - r_i * E
    dIdt = r_i * E - (r_r + r_d) * I
    dRdt = r_r * (1-p_dth) * I
    dDdt = r_d * p_dth * I

    return [dSdt, dEdt, dIdt, dRdt, dDdt]

def _get_initial_state(initial_cases:int, initial_deaths:int, params_fitted:List[float], population:float) -> List[float]:
    """
    Compute initial state to solve the IVP using initial observations and fitted params
    """

    r_e, r_i, r_r, r_d, k = params_fitted    
    D_0 = initial_deaths 
    R_0 = initial_deaths / p_dth - initial_deaths
    # Some regions start with deaths/p_dth > cases, which would give negative states; keep at least 1 infected
    I_0 = max(initial_cases - R_0 - D_0, 1.0)
    E_0 = I_0 * k
    S_0 = population - initial_cases - E_0

    return [S_0, E_0, I_0, R_0, D_0]

def _apply_bounds(params:List[float]) -> List[float]:
    params_bounded = [
        max(min(x, bounds[1]), bounds[0])
        for (x, bounds) in zip(params, param_bounds)
    ]
    return params_bounded

def _predict(params:List[float], initial_state:List, T:np.array, population:float):
    """
    Solve the SEIR model IVP and return predictions of cases and deaths.
    """

    try:
        solution = solve_ivp(
            fun=_model,
            y0=initial_state,
            t_span=[T[0], T[-1]],
            t_eval=T,
            args=(*params, population)
        )
        y_sol = solution.y
        # return cases and deaths prediction
        return y_sol[2, :], y_sol[4, :]
    except Exception as e:
        logger.error(f"_predict failed for params={params}, initial_state={initial_state}: {e}")
        raise

def _residuals(params:List[Any], cases:np.array, deaths:np.array, T:np.array, population:float) -> float:
    """
    Create the residual value from fitted parameters, initial state and observations
    """
    try:
        # Check & apply bounds
        params = _apply_bounds(params)

        initial_state = _get_initial_state(cases[0], deaths[0], params, population)

        # predict and calc residuals
        cases_pred, deaths_pred = _predict(params, initial_state, T, population)
        if len(cases_pred) != len(cases):
            # ODE integration failed part-way for these params: penalize instead of aborting the fit
            return 1e30

        # Weigh recent observations higher
        weights = list(range(1, len(cases) + 1))
        weights = [( x / len(cases) )**2 for x in weights]
        # Balance contribution of deaths and cases in the score
        balance = (cases[-1] / max(deaths[-1], 10))**2

        res = np.sum(
            np.multiply((cases - cases_pred)**2, weights) +
            balance * np.multiply((deaths - deaths_pred)**2, weights)
        )
        return res
    except Exception as e:
        logger.error(f"_residuals failed for params={params}: {e}")
        raise

def train(training_data:pd.DataFrame, population:float) -> List[float]:
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
            return _residuals(params, cases, deaths, T, population)

        if optimization_method == 'dual_annealing':
            model_fit = dual_annealing(
                _residuals_fit,
                x0 = default_params,
                bounds = param_bounds,
                callback=_make_callback_timeout(time.time())
            )
        elif optimization_method == 'tnc':
            model_fit = minimize(
                _residuals_fit,
                default_params,
                method='TNC',
                bounds=param_bounds,
                options={'maxiter': tnc_max_iter}
            )
        else:
            raise ValueError(f'Undefined optimization method {optimization_method}')

        return list(model_fit.x)
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise

def forecast(initial_cases:int, initial_deaths:int, optimal_params:List[float], T:List[int], population:float):
    """
    Forecasts using optimal parameters over the given time range.
    """
    initial_state = _get_initial_state(initial_cases, initial_deaths, optimal_params, population)

    return _predict(optimal_params, initial_state, T, population)


#####################
# Program entry-point
#####################

def main():
    log_path = _setup_logging()
    logger.info(f"Run log: {log_path}")

    regions = load_regions()
    logger.info(f"Loaded {len(regions)} regions; split at start date + {SPLIT_DAYS} days")

    all_params = {}
    all_scores = {}
    total_fit_runtime = 0.0
    param_names = list(default_params_dict.keys())

    for region, info in regions.items():
        try:
            data = info["data"]
            population = info["population"]
            split_date = info["split_date"]

            training_data = data[data.date <= split_date.to_pydatetime()].copy()
            logger.info(f"[{region}] Training on {len(training_data)} rows (of {len(data)} total), "
                        f"{data.date.iloc[0].date()} to {data.date.iloc[-1].date()}, split at {split_date.date()}")

            fit_start_time = time.time()
            optimal_params = train(training_data, population)
            fit_runtime = time.time() - fit_start_time
            total_fit_runtime += fit_runtime

            T = np.array(data.time_index.values)
            cases_forecast, deaths_forecast = forecast(data.iloc[0]['case_cnt'], data.iloc[0]['death_cnt'], optimal_params, T, population)
            evals = calculate_scores(data, cases_forecast, deaths_forecast, split_date)

            all_params[region] = dict(zip(param_names, optimal_params))
            all_scores[region] = evals
            logger.info(f"[{region}] fit time {fit_runtime:.2f}s, fit_score {evals['fit_score']:.2f}, params {optimal_params}")
        except Exception as e:
            logger.error(f"[{region}] failed, skipping: {e}")

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
    logger.info(f'Total fitting run time (all regions): {total_fit_runtime}')
    logger.info('--------------------------')

if __name__ == "__main__":
    main()