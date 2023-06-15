# OptML miniproject | Stochastic zeroth-order and first-order optimization
### How well do zero-order optimization methods do for ML applications, compared to standard first-order methods?

## Overview
This repository contains the source code of the miniproject for the course "CS-439 Optimization for Machine Learning" at EPFL. The performance of first-order methods, SGD and signSGD, and their ZO versions ZO-SGD and ZO-signSGD are compared with different hyperparameters and neural network configurations. The optimizers are evaluated on the MNIST dataset.

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
We performed a grid search on learning rate, momentum for FO-SGD, and forward difference estimate parameter $\epsilon$ for ZO-SGD and ZO-signSGD. We trained our MyNet model with the optimal parameter configuration for each optimizer. FO optimizers outperformed ZO optimizers in terms of training loss and accuracy. This is expected because ZO optimizers merely use estimates of the gradients. It is also notable that ZO optimizers seem to be giving more stable performance with the 4 optimal runs, which probably results from the landscape of the objective function. Since FO methods are using the actual gradient values, they tend to be more sensitive to the variance in the gradient.

![Comparison of ZO and FO optimizers with optimal configurations](./figures/optim_config_train_loss_accuracy_h.png)

Performance comparison was also conducted for different optimizers with same learning rates. The performance of FO optimizers appeared to be more sensitive to changes in the learning rate. Excessive learning rates could result in overshooting for FO optimizers, which was not observed for ZO optimizers in the scope of our experiments. ZO optimizers on the other hand utilize perturbation techniques to estimate the gradients which inherently introduce a level of noise. This noise can act as a regularizer, making ZO optimizers more robust to learning rate changes. Another point to note here is that to achieve similar performance, sign-optimizers generally require lower learning rates, which may be caused by the loss of information on gradient magnitude.

![Comparison of ZO and FO optimizers using same learning rates](./figures/FO_ZO_compare_train_loss_h.png)


## Set-up
The packages needed for running our code include:
* torch
* torchvision
* numpy
* pandas
* matplotlib

Please follow the steps below to create a new python environment and install the packages.
```
python -m venv venv                   # create a python environment in the venv folder
source venv/bin/activate              # activate this environment
pip install jupyter                   # install jupyter notebook
pip install -r requirements.txt       # install requirements
```

## Reproduction
The results in this file can be reproduced as follows:
* Run ```experiments.ipynb```, the metrics and model weights will be automatically saved to ```./metrics``` and ```./model``` directories
* Run ```data_analysis.ipynb```, this visualizes the results in ```./metrics```
