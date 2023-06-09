import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

class ZO_SignSGD(Optimizer):

    # =======================================================================#
    # Since we are doing experiments with functions with available gradients,#
    # but ZO optimizers have lots of unavailable gradient applications,      #
    # we introduce a use_tru_grad flag, and set to false by default          #
    # =======================================================================#

    def __init__(self, params, model, inputs, labels, criterion, lr=1e-3, fd_eps=1e-4, use_true_grad=False):
        self.model = model
        self.inputs = inputs
        self.labels = labels
        self.criterion = criterion
        defaults = dict(lr=lr, fd_eps=fd_eps, use_true_grad=use_true_grad)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
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
      # orig_model_state = self.model.state_dict().__str__()

    # Generate a random direction for the entire parameter vector
      direction = torch.randint(0, 2, param.data.shape) * 2 - 1
      direction = direction.to(param.device)

    # Perturb the parameters along the random direction
      param.data.add_(fd_eps * direction)

      # new_model_state = self.model.state_dict().__str__()
      # if orig_model_state == new_model_state:
      #     print("+ direction: Not updated")
      # else:
      #     print("+ direction: Updated")

    # Compute the loss with the perturbed parameters
      loss_plus = self.criterion(self.model(self.inputs), F.one_hot(self.labels, num_classes=10).float())  # Replace with our stochastic loss computation
      
    # Perturb the parameters in the opposite direction
      param.data.sub_(2 * fd_eps * direction)

      # new_model_state = self.model.state_dict().__str__()
      # if orig_model_state == new_model_state:
      #     print("- direction: Not updated")
      # else:
      #     print("- direction: Updated")
      
    # Compute the loss with the opposite perturbed parameters
      loss_minus = self.criterion(self.model(self.inputs), F.one_hot(self.labels, num_classes=10).float())  # Replace with our stochastic loss computation

    # Estimate the gradient using the finite difference approximation
      grad_est_flat = (loss_plus - loss_minus) / (2 * fd_eps)

    # Assign the estimated gradient to grad_est
      grad_est = grad_est_flat

    # Restore the original parameters
      param.data = orig_param.clone()
      print("grad_est: {:.2e}".format(grad_est))

      return grad_est