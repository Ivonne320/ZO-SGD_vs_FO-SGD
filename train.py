import os
import csv
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import defaultdict

from zo_sgd_vectorwise import *
from zo_sign_sgd_vectorwise import *
from fo_sgd import *
from fo_sign_sgd import *
from model import *


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
    if config["optimizer"] == "zo_sgd":
        optimizer = ZO_SGD(
            model_parameters,
            model,
            inputs,
            labels,
            criterion,
            lr = config["learning_rate"],
            fd_eps = config["fd_eps"],
            use_true_grad = config['use_true_grad']
        )
    elif config["optimizer"] == "zo_sign_sgd":
        optimizer = ZO_SignSGD(
            model_parameters,
            model,
            inputs,
            labels,
            criterion,
            lr = config["learning_rate"],
            fd_eps = config["fd_eps"],
            use_true_grad = config['use_true_grad']
        )
    elif config["optimizer"] == "fo_sgd":
        optimizer = FirstOrderSGD(
            model_parameters,
            lr=config["learning_rate"],
            momentum=config["momentum"],
            dampening=config["dampening"],
        )
    elif config["optimizer"] == "fo_sign_sgd":
        optimizer = FirstOrderSignSGD(
            model_parameters,
            lr=config["learning_rate"]
        )
    else:
        raise ValueError("Unexpected value for optimizer")
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config["decay_at_epochs"],
        gamma=1.0/config["decay_with_factor"],
    )

    return optimizer, scheduler


def get_model(device, config):
    """
    Parameters
    ====================
    device: instance of torch.device
    Return
    ====================
    An instance of torch.nn.Module
    """
    model = {
        "mynet": lambda: MyNet(num_classes=10)
    }[config["model"]]()

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


def main(unique_name, config):
    '''
    Train and test the model
    Parameters
    ====================
    unique_name: String
    config: Dictionary
    {
        "model",
        "num_epochs",
        "batch_size_train",
        "batch_size_test",
        "optimizer": one of "zo_sgd", "zo_sign_sgd", "fo_sgd", "fo_sign_sgd",
        "learning_rate",
        "momentum": used when "optimizer"="fo_sgd",
        "dampening": used when "optimizer"="fo_sgd",
        "fd_eps": used when "optimizer"="zo_sgd" or "zo_sign_sgd",
        "use_true_grad": Boolean,
        "scheduler": Boolean,
        "decay_at_epochs": Iterable, used when scheduler=True,
        "decay_with_factor": used when scheduler=True
        ""
    }
    Return
    ====================
    best_acc: Best test accuracy
    '''
    # Create directories
    os.makedirs("./model", exist_ok=True)
    os.makedirs("./metrics", exist_ok=True)

    # Set the seed
    torch.manual_seed(config["seed"])

    # We will run on CUDA if there is a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Configure the dataset, model and the optimizer based on the 'config' dictionary
    training_loader, test_loader = get_dataset(config)
    inputs, labels = next(iter(training_loader))
    inputs, labels = inputs.to(device), labels.to(device)
    model = get_model(device, config)
    # criterion = torch.nn.BCELoss() # [TODO] maybe use CrossEntropyLoss
    criterion = torch.nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer(model.parameters(), config, model, inputs, labels, criterion)

    # Store the loss and accuracy
    epoch_metrics = defaultdict(list)
    best_acc = 0

    for epoch in range(config["num_epochs"]):
        print("Epoch {}/{}".format(epoch, config["num_epochs"]))
        start = time.time()
        batch_metrics = defaultdict(float)

        # Training mode
        model.train()

        for batch_idx, (inputs, labels) in enumerate(training_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Compute gradients for the batch
            optimizer.zero_grad()
            prediction = model(inputs)
            loss = criterion(prediction, F.one_hot(labels, num_classes=10).float())
            acc = accuracy(prediction, labels)
            loss.backward()

            # Do an optimizer step
            optimizer.step()

            # Store the statistics
            batch_metrics["train_loss"] += loss.detach().cpu().numpy()
            batch_metrics["train_acc"] += acc.detach().cpu().numpy()

        # Update the optimizer's learning rate
        if scheduler:
            scheduler.step()

        # Log training stats
        epoch_loss = batch_metrics["train_loss"] / (batch_idx + 1)
        epoch_acc = batch_metrics["train_acc"] / (batch_idx + 1)
        epoch_metrics["train_loss"].append(epoch_loss)
        epoch_metrics["train_acc"].append(epoch_acc)
        print("train loss: {:.3f}, train accuracy: {:.3f}".format(epoch_loss, epoch_acc))

        # Evaluation mode
        model.eval()

        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            prediction = model(inputs)
            loss = criterion(prediction, F.one_hot(labels, num_classes=10).float())
            acc = accuracy(prediction, labels)
            batch_metrics["test_loss"] += loss.detach().cpu().numpy()
            batch_metrics["test_acc"] += acc.detach().cpu().numpy()

        # Log test stats
        epoch_loss = batch_metrics["test_loss"] / (batch_idx + 1)
        epoch_acc = batch_metrics["test_acc"] / (batch_idx + 1)
        epoch_metrics["test_loss"].append(epoch_loss)
        epoch_metrics["test_acc"].append(epoch_acc)
        print("test loss: {:.3f}, test accuracy: {:.3f}".format(epoch_loss, epoch_acc))

        if epoch_acc > best_acc: 
            best_acc = epoch_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())

        end = time.time()
        elapsed = end - start
        print("time elapsed: {:02}:{:02}".format(int(elapsed//60), int(elapsed%60)))

    # Save the model with the best test accuracy
    lr = config["learning_rate"]
    momentum = config["momentum"]
    fd_eps = config["fd_eps"]
    if config["optimizer"] in ["zo_sgd", "zo_sign_sgd"]:
        model_path = os.path.join("./model", f'{unique_name}_{config["model"]}_{config["optimizer"]}-ep_{best_epoch}-lr_{lr}-eps_{fd_eps}.pt')
        metrics_path = os.path.join("./metrics", f'{unique_name}_{config["model"]}_{config["optimizer"]}-ep_{best_epoch}-lr_{lr}-eps_{fd_eps}.csv')
    elif config["optimizer"] == "fo_sgd":
        model_path = os.path.join("./model", f'{unique_name}_{config["model"]}_{config["optimizer"]}-ep_{best_epoch}-lr_{lr}-momentum_{momentum}.pt')
        metrics_path = os.path.join("./metrics", f'{unique_name}_{config["model"]}_{config["optimizer"]}-ep_{best_epoch}-lr_{lr}-momentum_{momentum}.csv')
    elif config["optimizer"] == "fo_sign_sgd":
        model_path = os.path.join("./model", f'{unique_name}_{config["model"]}_{config["optimizer"]}-ep_{best_epoch}-lr_{lr}.pt')
        metrics_path = os.path.join("./metrics", f'{unique_name}_{config["model"]}_{config["optimizer"]}-ep_{best_epoch}-lr_{lr}.csv')
    torch.save(best_model_wts, model_path)
    
    # Save the metrics
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(epoch_metrics.keys())
        writer.writerows(zip(*epoch_metrics.values()))
    
    return best_acc
