import torch

class FirstOrderSignSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = torch.sign(p.grad.data)
                while (grad==0).any():
                    grad[grad==0] = torch.randint_like(grad[grad==0], low=0, high=2)*2 - 1
                lr = group['lr']
                p.data.add_(grad, alpha=-lr)