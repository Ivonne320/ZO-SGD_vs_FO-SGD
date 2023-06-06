import torch
from torch.optim.optimizer import Optimizer
import random

class ZO_SGD(Optimizer):

    # =======================================================================#
    # Since we are doing experiments with functions with available gradients,#
    # but ZO optimizers have lots of unavailable gradient applications,      #
    # we introduce a use_tru_grad flag, and set to false by default          #
    # =======================================================================#

    def __int__(self, params, lr=1e-3, eps=1e-8, fd_eps=1e-4, use_true_grad=False):
        defaults = dict(lr=lr, eps=eps, fd_eps=fd_eps, use_true_grad=use_true_grad)
        super().__init__(params, defaults)

    def step(self,closure=None):
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
                    grad_est = self._compute_gradient_direction(param, eps, fd_eps, closure)
                param.data.add_(-lr * grad_est)

    def _compute_gradient_direction(self, param, eps, fd_eps, closure):
        grad_est = torch.zeros_like(param.data)
        for i in range(param.data.numel()):
            p_flat = param.data.view(-1)
            p_flat[i] += fd_eps
            loss_plus = closure()
            p_flat[i] -= 2 * fd_eps
            loss_minus = closure()
            grad_est_flat = (loss_plus - loss_minus) / (2 * fd_eps)
            grad_est[i] = grad_est_flat
        return grad_est

