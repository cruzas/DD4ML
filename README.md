# DD4ML
Domain decomposition methods for machine learning.

## Authors
* Samuel A. Cruz Alegría (1, 3); cruzas@usi.ch.
* Ken Trotti (2); ken.trotti@usi.ch
* Marc Salvadó Benasco (1, 3, 4); marc.salvado@usi.ch
* Shega Likaj (1, 2, 3); shega.likaj@usi.ch
* Bindi Capriqi (1, 2, 3); bindi.capriqi@kaust.edu.sa
* Armando Maria Monforte (3, 5); armandomaria.monforte01@universitadipavia.it
* Prof. Dr. Rolf Krause (2, 3); rolf.krause@kaust.edu.sa

## Collaborators
* Alena Kopaničáková (6)

## Universities
1. Università della Svizzera italiana
2. King Abdullah University of Science and Technology (KAUST)
3. UniDistance Suisse
4. Universitat Politècnica de Catalunya (UPC)
5. University of Pavia
6. University of Toulouse

## Funding
This work was initially supported by the Swiss Platform for Advanced Scientific Computing (PASC) project **ExaTrain** (funding periods 2017-2021 and 2021-2024) and by the Swiss National Science Foundation through the projects "ML<sup>2</sup> -- Multilevel and Domain Decomposition Methods for Machine Learning" (197041) and "Multilevel training of DeepONets -- multiscale and multiphysics applications" (206745). 

## Requirements
See ``pyproject.toml`` file. 

## Installation
```bash
git clone https://github.com/cruzas/DD4ML.git
cd DD4ML
python3 -m pip install .
```

## Usage
Look into the tests folder. For example, you can run:
```bash
python3 ./tests/chargpt/chargpt.py
```

## Note
In case it's necessary, you may need to run the following:
```bash
python3 -m pip install --force-reinstall .
```
Based on your Python environment, you may need to clear out the site-packages directory.
