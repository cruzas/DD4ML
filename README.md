# DD4ML

[![CI](https://github.com/cruzas/DD4ML/actions/workflows/ci.yml/badge.svg)](https://github.com/cruzas/DD4ML/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Domain decomposition methods for machine learning.

The code uses minGPT and builds from it: https://github.com/karpathy/minGPT/tree/master

## Authors
* Samuel A. Cruz Alegría (1, 3); cruzas@usi.ch.
* Dr. Ken Trotti (2); ken.trotti@usi.ch
* Marc Salvadó Benasco (1, 3, 4); marc.salvado@usi.ch
* Shega Likaj (1, 2, 3); shega.likaj@usi.ch
* Bindi Capriqi (1, 2, 3); bindi.capriqi@kaust.edu.sa
* Armando Maria Monforte (3, 5); armandomaria.monforte01@universitadipavia.it
* Prof. Dr. Rolf Krause (2, 3); rolf.krause@kaust.edu.sa

## Collaborators
* Prof. Dr. Alena Kopaničáková (6)

## Universities
1. Università della Svizzera italiana
2. King Abdullah University of Science and Technology (KAUST)
3. UniDistance Suisse
4. Universitat Politècnica de Catalunya (UPC)
5. University of Pavia
6. University of Toulouse

## Requirements
Python 3.10 or newer. Runtime and development dependencies are declared in
``pyproject.toml``; there is no separate ``requirements.txt`` and no ``setup.py``.

## Installation
This project is still in development. To install it in editable mode, you can run:
```bash
git clone https://github.com/cruzas/DD4ML.git
cd DD4ML
python3 -m pip install -e .
```

If you are satisfied with the current version and plan no further changes, you can run:
```bash
git clone https://github.com/cruzas/DD4ML.git
cd DD4ML
python3 -m pip install .
```

## Development
Install the package with the development extra, then enable the git hooks:
```bash
python3 -m pip install -e ".[dev]"
pre-commit install
```

Lint, format, and test exactly as CI does:
```bash
ruff check .
ruff format --check .
pytest
```

`ruff check` is enforced at zero violations, so any new finding fails the build.
The enabled rule families are `E4`, `E7`, `E9`, `F`, `I`, and `UP`; the
`B`, `C4`, `SIM`, and `RUF` families are a known backlog of roughly 220 findings
on numerical code and are switched on family-by-family as they are burned down.

One cleanup is still in progress and is visible in the configuration:
`dd4ml/datasets/` and `dd4ml/models/` still use `from ... import *`. The same
refactor has already been applied to `dd4ml/optimizers/`, where every module now
imports explicitly.

The repository-wide formatting commit is listed in `.git-blame-ignore-revs`, so
`git blame` can skip it:
```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

### ASNTR
`dd4ml/optimizers/asntr.py` implements Algorithm 1 of Krejić, Krklec Jerinkić,
Martínez and Yousefi, *A non-monotone trust-region method with noisy oracles and
additional sampling*, Computational Optimization and Applications **89**:247–278
(2024), [doi:10.1007/s10589-024-00580-w](https://doi.org/10.1007/s10589-024-00580-w).
`tests/test_asntr.py` pins the two acceptance ratios to Eqs. (6), (7), (9) and (10)
of that paper, including the sign conventions, and cites the equation numbers
next to each assertion.

## CUDA Support
For ***GPU support***, install the appropriate CUDA-enabled version of PyTorch before installing this package. For example, to install PyTorch with CUDA 12.4:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Usage
In a ***local*** environment (e.g. PC), for example, you can run:
```bash
open -a Docker
wandb server start
python3 ./tests/run_config_file.py --sweep_config="./tests/config_files/config_sgd.yaml"
```

In a ***cluster*** environment, e.g. managed by SLURM, you can run:
```bash
cd tests
./submit_jobs.sh
```

Note:
- The code works with or without wandb. If you need it, make sure to install wandb accordingly.
- In a cluster environment, you can set wandb usage by setting ```export USE_WANDB=1``` in ```./tests/submit_jobs.sh```.
- The ```Factory``` class defined in ```src/dd4ml/utility/factory.py``` allows you to dynamically add new classes, datasets, etc.
- Ensure that the configured ```batch_size``` is at least the number of processes (```world_size```). If it is smaller, each process defaults to a per-process batch size of 1.

## Structure
This library is meant to be general.

The src folder is structured as follows:
- datasets (for processing data in rawdata)
- models
- optimizers
- pmw
- utility

You can extend the library by adding your own files in any of these modules. If you create a new folder within them, make sure to add an ```__init__.py``` file and then re-run ```python3 -m pip install .```, or ```python3 -m pip install --force-reinstall .``` if necessary.

### DeepONet Example
This project includes a basic DeepONet implementation (`deeponet`) together with
a small synthetic dataset (`deeponet_sine`). To try it out locally you can run
`tests/run_config_file.py` with `--model_name deeponet --dataset_name deeponet_sine`.
For batch testing on a cluster, use `tests/submit_jobs.sh`.

## Note
In case it's necessary, you may need to run the following:
```bash
python3 -m pip install --force-reinstall .
```
Based on your Python environment, you may need to also clear out the site-packages directory. You can find it by using the following command:
```bash
python3 -m site
```

Before using using wandb locally on your computer, you need to make an account. Then, you can run the following command:
```bash
wandb login --relogin --host=http://127.0.0.1
```
You will need your API key: https://wandb.ai/authorize
Once you have done this, your credentials are saved. For more information, please consult: https://docs.wandb.ai/quickstart/

## Funding
This work was initially supported by the Swiss Platform for Advanced Scientific Computing (PASC) project **ExaTrain** (funding periods 2017-2021 and 2021-2024) and by the Swiss National Science Foundation through the projects "ML<sup>2</sup> -- Multilevel and Domain Decomposition Methods for Machine Learning" (197041) and "Multilevel training of DeepONets -- multiscale and multiphysics applications" (206745).
