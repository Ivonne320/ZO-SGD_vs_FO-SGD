import torch
from torch.optim.optimizer import Optimizer
import random
#from logistic_regression import model, criterion, X_train, y_train

class ZO_SGD(Optimizer):

    # =======================================================================#
    # Since we are doing experiments with functions with available gradients,#
    # but ZO optimizers have lots of unavailable gradient applications,      #
    # we introduce a use_tru_grad flag, and set to false by default          #
    # =======================================================================#

    def __init__(self, params, lr=1e-3, eps=1e-8, fd_eps=1e-4, use_true_grad=False):
        defaults = dict(lr=lr, eps=eps, fd_eps=fd_eps, use_true_grad=use_true_grad)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            fd_eps = group['fd_eps']
            use_true_grad = group['use_true_grad']
            for param in group['params']:
                if param.grad is None and use_true_grad:
                    raise ValueError('ZO_SGD should be used only when gradients are not available,'
                                     'or when use_true_grad is False.')
                if use_true_grad:
                    grad_est = param.grad.data
                else:
                    grad_est = self._compute_gradient_direction(param, fd_eps)

                param.data.add_(-lr * torch.sign(grad_est))


    def _compute_gradient_direction(self, param, fd_eps):
      grad_est = torch.zeros_like(param.data)
      orig_param = param.data.clone()

    # Generate a random direction for the entire parameter vector
 
      direction = torch.randint(0, 2, param.data.shape) * 2 - 1

    # Perturb the parameters along the random direction
      param.data.add_(fd_eps * direction)

    # Compute the loss with the perturbed parameters
      loss_plus = criterion(model(X_train.squeeze()), y_train.float())  # Replace with our stochastic loss computation

    # Perturb the parameters in the opposite direction
      param.data.sub_(2 * fd_eps * direction)

    # Compute the loss with the opposite perturbed parameters
      loss_minus = criterion(model(X_train.squeeze()), y_train.float())  # Replace with our stochastic loss computation

    # Estimate the gradient using the finite difference approximation
      grad_est_flat = (loss_plus - loss_minus) / (2 * fd_eps)

    # Assign the estimated gradient to grad_est
      grad_est = grad_est_flat

    # Restore the original parameters
      param.data = orig_param.clone()

      return grad_est