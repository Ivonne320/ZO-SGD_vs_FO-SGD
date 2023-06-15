# OptML miniproject | Stochastic zeroth-order and first-order optimization
### How well do zero-order optimization methods do for ML applications, compared to standard first-order methods?

## Overview
This repository contains the source code of the miniproject for the course "CS-439 Optimization for Machine Learning" at EPFL. The performance of first-order methods, SGD and signSGD, and their ZO versions: ZO-SGD and ZO-signSGD are compared with different hyperparameters and neural network configurations.

The repository is structured as below:
```
|------ metrics                       # Results of the experiments
|------ README.md                     # You are here
|------ data_analysis.ipynb           # Results visualization
|------ experiments.ipynb             # Run experiments
|------ fo_sgd.py                     # First-order SGD optimizer
|------ fo_sign_sgd.py                # First-order signSGD optimizer
|------ logistic_regression.ipynb     # Test code
|------ model.py                      # A fully connected NN and a CNN
|------ requirements.txt              # Required packages
|------ train.py                      # Functions for model training
|------ zo_sgd.py                     # Zeroth-order SGD optimizer with elementwise gradient estimation
|------ zo_sgd_vectorwise.py          # Zeroth-order SGD optimizer with vectorwise gradient estimation
|------ zo_sign_sgd.py                # Zeroth-order signSGD optimizer with elementwise gradient estimation
|------ zo_sign_sgd_vectorwise.py     # Zeroth-order signSGD optimizer with vectorwise gradient estimation
```

## Results



## Set-up
The dependencies needed for running our code are listed in requirements.txt. Please follow the steps below to create a new python environment and install the dependencies:
```
python -m venv venv                   # create a python environment in the venv folder
source venv/bin/activate              # activate this environment
pip install jupyter                   # install jupyter notebook
pip install -r requirements.txt       # install requirements
```

## Reproduction
The results in this file can be reproduced as follows:

* Run ```experiments.ipynb```, the metrics and model weights will be automatically saved to ```./metrics``` and ```./model``` directories
* Run ```data_analysis.ipynb```, this visualizes the results in ./metrics
