import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

#from logistic_regression import model, criterion, X_train, y_train

class ZO_SGD(Optimizer):

    # =======================================================================#
    # Since we are doing experiments with functions with available gradients,#
    # but ZO optimizers have lots of unavailable gradient applications,      #
    # we introduce a use_tru_grad flag, and set to false by default          #
    # =======================================================================#

    def __init__(self, params, model, inputs, labels, criterion, lr=1e-3, fd_eps=1e-4, 
                 use_true_grad=False, momentum=0, dampening=0):
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

                param.data.add_(-lr * grad_est)


    def _compute_gradient_direction(self, param, fd_eps):
        grad_est = torch.zeros_like(param.data)
        orig_param = param.data.clone()

        # Generate a random direction for the entire parameter vector
        direction = torch.randint(0, 2, param.data.shape) * 2 - 1
    

        # Perturb the parameters along the random direction
        param.data.add_(fd_eps * direction)



        # Compute the loss with the perturbed parameters
        #print(self.inputs)#self.model(self.inputs), self.labels)
        loss_plus = self.criterion(self.model(self.inputs), self.labels)  
        # Replace with our stochastic loss computation

        # Perturb the parameters in the opposite direction
        param.data.sub_(1 * fd_eps * direction)
      

        # Compute the loss with the opposite perturbed parameters
        loss_ = self.criterion(self.model(self.inputs), self.labels) 
        # Replace with our stochastic loss computation
      
        # Estimate the gradient using the finite difference approximation
        grad_est_flat = (loss_plus - loss_) / (fd_eps)

        # Assign the estimated gradient to grad_est
        grad_est = grad_est_flat/direction

        # Restore the original parameters
        param.data = orig_param.clone()

        return grad_est