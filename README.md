# Autoresearch-Epidemics

This is a project to do LLMs driven research on epedemiological forecasting models. The primary inspiration for this repo is Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch/tree/master) project.

## Overall Design

The repo has a simple structure with all the model and training code organized into two python files - `constants.py` and `train.py`, and the agent instructions are written in `program.md`. Other than that, all training data as well as experiment results are to be stores within the `data/` sub-directory.

The agent reads the `program.md` for instructions and is only allowed to modifiy `train.py` in every experiment cycle.

Logs for all the training runs are written in `run_logs/`


## Quick Start

## Requirements

## License
MIT