import torch
from torch.optim.optimizer import Optimizer
import random
#from logistic_regression import model, criterion, X_train, y_train

class ZO_SignSGD(Optimizer):

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
        orig_param = param.data.clone() # Make a copy of the original parameters
      
        for i in range(param.data.numel()):
            # idea here is to element-wisely estimate the gradient by finite difference
            param.data.view(-1)[i] += fd_eps # update the i-th element of the parameters towards one direction
            loss_plus = criterion(model(X_train.squeeze()), y_train.float()) # replace here with our stochastic loss computation
            param.data.view(-1)[i] -= 2 * fd_eps # update the i-th element of the parameters towards another direction
            loss_minus = criterion(model(X_train.squeeze()), y_train.float()) # replace here with our stochastic loss computation
            grad_est_flat = (loss_plus - loss_minus) / (2 * fd_eps)
            grad_est.view(-1)[i] = grad_est_flat
            param.data = orig_param.clone() # restore the original parameters
        
        
        return grad_est
    