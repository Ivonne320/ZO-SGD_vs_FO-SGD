import torch
from torch.optim import Optimizer


class FO_SGD(torch.optim.Optimizer):
    def __init__(self, params, model, inputs, labels, criterion, lr=1e-3, fd_eps=1e-4, 
                 use_true_grad=False, momentum=0, dampening=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("Invalid momentum: {}".format(momentum))
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                delta_p = p.grad.data
                state = self.state[p]
                if 'velocity' not in state:
                    state['velocity'] = torch.zeros_like(p.data)
                velocity = state['velocity']
                lr = group['lr']
                momentum = group['momentum']
                dampening = group['dampening']
                velocity.mul_(momentum).add_(delta_p, alpha=1-dampening)
                p.data.add_(velocity, alpha=-lr)
                

    
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
    
from torch.optim import Optimizer

class random_search(Optimizer):
    
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
        loss_ = self.criterion(self.model(self.inputs), self.labels)
        # Generate a random direction for the entire parameter vector
        direction = torch.randn_like(param)
    
        # Perturb the parameters along the random direction
        param.data.add_(fd_eps * direction)

        # Compute the loss with the perturbed parameters
        loss_plus = self.criterion(self.model(self.inputs), self.labels)  
        # Replace with our stochastic loss computation

        # Perturb the parameters in the opposite direction
        param.data.sub_(1 * fd_eps * direction)
      

        # Compute the loss with the opposite perturbed parameters
 
        # Replace with our stochastic loss computation
      
        # Estimate the gradient using the finite difference approximation
        grad_est_flat = (loss_plus - loss_) / (fd_eps)

        # Assign the estimated gradient to grad_est
        grad_est = grad_est_flat/direction

        # Restore the original parameters
        param.data = orig_param.clone()

        return grad_est