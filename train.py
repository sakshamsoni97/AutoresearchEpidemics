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

from constants import TRAINING_DATA_PATH, START_DATE, SPLIT_DATE, END_DATE, TOTAL_POPULATION


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


###########################
# Model & Fitting Functions
###########################

def _model(t, x, r_e, r_i, r_r, r_d, k):
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
    """
    assert len(x) == 5, f"Wrong number of input states, got {len(x)}, expected 6"
    S, E, I, R, D = x
    dSdt = - r_e * S * I / TOTAL_POPULATION
    dEdt = r_e * S * I / TOTAL_POPULATION - r_i * E
    dIdt = r_i * E - (r_r + r_d) * I
    dRdt = r_r * (1-p_dth) * I
    dDdt = r_d * p_dth * I

    return [dSdt, dEdt, dIdt, dRdt, dDdt]

def _get_initial_state(initial_cases:int, initial_deaths:int, params_fitted:List[float]) -> List[float]:
    """
    Compute initial state to solve the IVP using initial observations and fitted params
    """

    r_e, r_i, r_r, r_d, k = params_fitted    
    D_0 = initial_deaths 
    R_0 = initial_deaths / p_dth - initial_deaths
    I_0 = initial_cases - R_0 - D_0
    E_0 = I_0 * k
    S_0 = TOTAL_POPULATION - initial_cases - E_0

    return [S_0, E_0, I_0, R_0, D_0]

def _apply_bounds(params:List[float]) -> List[float]:
    params_bounded = [
        max(min(x, bounds[0]), bounds[1])
        for (x, bounds) in zip(params, param_bounds)
    ]
    return params_bounded

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
        # Check & apply bounds
        params = _apply_bounds(params)

        initial_state = _get_initial_state(cases[0], deaths[0], params)

        # predict and calc residuals
        cases_pred, deaths_pred = _predict(params, initial_state, T)

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
            bounds = param_bounds
        )

        return list(model_fit.x)
    except Exception as e:
        logger.error(f"train failed: {e}")
        raise

def forecast(initial_cases:int, initial_deaths:int, optimal_params:List[float], T:List[int]):
    """
    Forecasts using optimal parameters over the given time range.
    """
    initial_state = _get_initial_state(initial_cases, initial_deaths, optimal_params)

    return _predict(optimal_params, initial_state, T)
    
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


def calculate_scores(data:pd.DataFrame, optimal_params:List[float]):
    """
    Evaluate the performance of the model.
    """
    T = np.array(data.time_index.values)
    training_cutoff_date_dt = pd.to_datetime(SPLIT_DATE).to_pydatetime()

    cases_forecast, deaths_forecast = forecast(data.iloc[0]['case_cnt'], data.iloc[0]['death_cnt'], optimal_params, T)

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
    evals["Test Score"] = (evals["MAPE Cases Test"] + evals["MAPE Deaths Test"])/2

    return evals

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

    trainiing_data = data[data.date <= training_cutoff_date_dt].copy()
    logger.info(f"Training on {len(trainiing_data)} rows (of {len(data)} total)")

    fit_start_time = time.time()
    optimal_params = train(trainiing_data)
    fit_runtime = time.time() - fit_start_time
    logger.info(f"Model fitting completed in {fit_runtime:.2f} seconds")
    logger.info(f"Optimal params: {optimal_params}")

    evals = calculate_scores(data, optimal_params)
    logger.info(f"Evaluation scores: {evals}")

    # TODO: print formatted results summary
    print(evals)

if __name__ == "__main__":
    main()