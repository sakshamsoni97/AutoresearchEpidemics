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

Next, make sure you can run the `train.py` file on the main branch.

```bash
> uv run train.py
```

You're all set. Now you can start a new session with your favorite coding agent like Claude Code or Codex, ask it to read `program.md` and start the experiments. (Tip: enable automatic-edits so that you don't have to sit around hitting accept on every step.)


## Requirements

The repo uses Python 3.12 and the main dependencies are `pandas`, `numpy` and `scipy`. The full list of dependencies are listed in `pyproject.toml`.

## Author
_Saksham Soni_

## License
MIT