import os
import csv
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import defaultdict

from optimizers import *
from models import *

def get_dataset(config):
    """
    Create dataset loaders for the MNIST dataset
    Return
    ====================
    Tuple (training_loader, test_loader)
    """
    data_path = "./data"
    os.makedirs(data_path, exist_ok=True)
    
    dataset = torchvision.datasets.MNIST
    data_mean = 0.1307
    data_stddev = 0.3081

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ]
    )

    training_set = dataset(
        root=data_path, train=True, download=True, transform=transform
    )
    test_set = dataset(
        root=data_path, train=False, download=True, transform=transform
    )

    training_loader = torch.utils.data.DataLoader(
        training_set, batch_size=config["batch_size_train"], shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=config["batch_size_test"], shuffle=False
    )

    return training_loader, test_loader

def get_optimizer(model_parameters, config, model, inputs, labels, criterion):
    """
    Create an optimizer for a given model
    Parameters
    ====================
    model_parameters: a list of parameters to be trained
    Return
    ====================
    Tuple (optimizer, scheduler)
    """
    
    optimizers = {"zo_sgd": ZO_SGD, "fo_sgd": FO_SGD, "random": random_search}

    optimizer = optimizers[config["optimizer"]](model_parameters, model, inputs, labels, criterion,
            lr = config["learning_rate"],
            fd_eps = config["fd_eps"],
            use_true_grad = config['use_true_grad'], 
            momentum=config["momentum"],
            dampening=config["dampening"])
   
    
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #    optimizer,
    #    milestones=config["decay_at_epochs"],
    #    gamma=1.0/config["decay_with_factor"],
    #)
    scheduler = None
    return optimizer, scheduler

def get_model(device, config, scale):
    """
    Parameters
    ====================
    device: instance of torch.device
    Return
    ====================
    An instance of torch.nn.Module
    """
    #print(config)
    models = {"mynet": MyNet, "conv": ConvNet}
    model = models[config](scale=scale)

    model.to(device)
    if device == "cuda:0":
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True
    return model

def accuracy(pred, label):
    """
    Compute the ratio of correctly predicted labels
    """
    pred = torch.argmax(pred, dim=1)
    num_correct_pred = (pred == label).sum()
    
    return num_correct_pred.float() / label.nelement()