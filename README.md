# Autoresearch-Epidemics

This is a project to do LLM-agent driven research on epedemiological forecasting models. The primary inspiration for this repo is Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch/tree/master) project. The goal is to see how we can use agents to test several different hypotheses and optimization methods automatically, in a matter of hours, and the human researcher can come back to analyze the results of several experiments.

## Overall Design

The repo has a simple structure with all the model and training code organized into two python files - `utils.py` and `train.py`, and the agent instructions are written in `program.md`. Other than that, all training data as well as experiment results are to be stores within the `data/` sub-directory.

The agent reads the `program.md` for instructions and is only allowed to modifiy `train.py` in every experiment cycle.

This minimal project design is intentional and also inspired from Andrej Karpathy's autoresearch project. Since the core idea of this project is to see if LLM-agents can be used to do research on their own by finding ideas from published papers, coming up with hypotheses and testing several different models iteratively, most of the code is kept in one file to keep token usage minimal and edits easy to track. 

Logs for all the training runs are written in `run_logs/`

## Quick Start

To start, it is recommended to create a virtual environment. I used `uv` and `venv` for this repository and all dependencies are listed in the `pyproject.toml` file.

```bash
> uv venv --python 3.12
> uv sync
> source .venv/bin/activate
```

There are two python scripts provided in the scripts folder to download and set up the training data needed for the experiments, these need to be run in this order -
1. `> python download_delphi_cases.py` downloads the raw data from a public GitHub repository'
2. `> python local_data_preparation.py` processes the raw files and saves the outputs to `data/tranining/`

Next, make sure you can run the `train.py` file on the main branch.

```bash
> uv run train.py
```

You're all set. Now you can start a new session with your favorite coding agent like Claude Code or Codex, ask it to read `program.md` and start the experiments. (Tip: enable automatic-edits so that you don't have to sit around hitting accept on every step.)


## Requirements

The repo uses Python 3.12 and the main dependencies are `pandas`, `numpy` and `scipy`. The full list of dependencies are listed in `pyproject.toml`.

## Past Experiments and Project Evolution

### Project Evolution on `main`

Commits on `main` fall into four main steps:

1. **Repo setup (Jul 3).** Initial commit with the project skeleton.
2. **Baseline SEIR model and agent harness (Jul 12).** A single-region (New York) SEIR model fit with `dual_annealing` in `train.py`, shared evaluation code moved to `utils.py`, the agent instructions in `program.md`, and an `analysis.ipynb` notebook. A bug in `_apply_bounds()` was fixed and the TNC optimizer was added as an option.
3. **All-US data pipeline (Nov 15).** Download and preparation scripts for Delphi COVID-19 case and death data covering all US regions, and a `train.py`/`utils.py` rewrite to load multiple regions. The single-region CSV was removed from version control. `program.md` was updated to match.
4. **Parallel training (Nov 19).** Region fits run in a process pool, which makes the 51-region fit practical. The `.gitignore` was also changed so logs and results are kept on the experiment branches rather than `main`.

### Experiment Branches

Each branch is one autoresearch session. The agent edits only `train.py` and logs every run to `data/experiment_results/results.tsv`. `fit_score` is lower-is-better, and it is the mean of the test-period MAPE for cases and deaths.

**`autoresearch/jul12` (single region: New York).** 18 runs: baseline plus 17 experiments, of which 15 were kept (baseline included) and 3 discarded. `fit_score` went from **866 to 2.16**.
- *Model structure (866 → 8).* Exponentially declining transmission `beta = r_e * exp(-a*t)` to represent lockdown (866 → 172). Fixing a buggy `_apply_bounds()` that forced all parameters to their upper bound (172 → 54). Mapping cumulative cases to the monotonic, mass-conserving `N - S` (53 → 22). Modelling deaths as `p_dth` times lagged cumulative cases (20 → 8). The model was then simplified to a 3-compartment SEI with the same score.
- *Optimizer.* Multi-restart `dual_annealing` using the full time budget (22.4 → 21.4).
- *Recency weighting (8 → 2.16).* The loss was weighted by `(i/n)^k` for k = 4, 6, 8, 12, 16, with each step improving the score but the gains shrinking.
- *Discarded.* Using `I+R+D` for cases, a rigid deaths = CFR × cases ratio, and a transmission floor.

**`autoresearch/nov19` (all 51 US regions).** 10 runs: baseline plus 9 experiments, of which 4 were kept (baseline included) and 6 discarded. `fit_score` went from **98.1 to 7.22**. Scores are not comparable with the `jul12` branch because they are averaged over 51 regions.
- *Port of the `jul12` winners (98 → 13.2).* The SEI model, declining transmission, `N - S` cases, lagged-CFR deaths, relative loss, and recency^16 weighting.
- *Nelder-Mead polish after `dual_annealing` (13.2 → 7.57).* Annealing's local search stalls on the non-smooth ODE loss.
- *Freezing transmission at its end-of-training value for the forecast (7.57 → 7.22).*
- *Discarded.*
  - Tracking cumulative cases as their own ODE state.
  - A transmission floor, which made some regions explode.
  - Decoupling the deaths fit from the cases fit, which made both worse because deaths regularize the transmission fit.
  - Recency powers of 8 and 32, which showed that 16 is the sweet spot.
  - A seed check showing run-to-run noise of about 0.4.

### Project Timeline

| Date | Title | Description |
|------|-------|-------------|
| 2026-07-03 | **Repo setup** — initial commit | Project skeleton created, inspired by Karpathy's autoresearch. |
| 2026-07-12 | **SEIR baseline and agent harness** — single-region (NY) model fit with `dual_annealing` | `train.py` fits an SEIR model, `utils.py` holds fixed evaluation code, and `program.md` gives the agent its instructions. Also adds `analysis.ipynb`, a `_apply_bounds()` fix and a TNC optimizer option. |
| 2026-07-12 | **Experiment `jul12`** — 17 experiments on NY, `fit_score` 866 → 2.16 | Declining transmission, a fix for the bounds bug, `N - S` cases, lagged-death modelling, a SEI simplification and multi-restart optimization took the score from 866 to 8. Recency-weighted loss (`(i/n)^16`) then brought it to 2.16. |
| 2026-11-15 | **All-US data pipeline** — Delphi download and prep scripts | Adds `scripts/download_delphi_cases.py` and `scripts/local_data_preparation.py`, and rewrites `train.py` and `utils.py` for multi-region training and evaluation. |
| 2026-11-19 | **Parallel training** — regions fit concurrently in a process pool | Cuts wall-clock time for the 51-region run. Logs and results now go to experiment branches. |
| 2026-11-19 | **Experiment `nov19`** — 9 experiments on 51 regions, `fit_score` 98 → 7.22 | Ports the `jul12` model to all regions (98 → 13.2). A Nelder-Mead polish and a frozen forecast transmission improve it to 7.22. Recency power 16 holds up, and several other ideas were discarded. |

## Author
_Saksham Soni_

## License
MIT