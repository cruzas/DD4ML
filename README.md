# DD4ML
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
See ``pyproject.toml`` file. 

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
./submit_job.sh
```

Note: 
- The code works with or without wandb. If you need it, make sure to install wandb accordingly. 
- In a cluster environment, you can set wandb usage by setting ```export USE_WANDB=1``` in ```./tests/submit_job.sh```.
- The ```Factory``` class defined in ```src/dd4ml/utility/factory.py``` allows you to dynamically add new classes, datasets, etc.

## Structure
This library is meant to be general. 

The src folder is structured as follows:
- datasets (for processing data in rawdata)
- models 
- optimizers
- pmw
- utility

You can extend the library by adding your own files in any of these modules. If you create a new folder within them, make sure to add an ```__init__.py``` file and then re-run ```python3 -m pip install .```, or ```python3 -m pip install --force-reinstall .``` if necessary. 

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
