import torch

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