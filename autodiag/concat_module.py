import torch
import torch as th


class Concatenate(torch.nn.Module):
    """
    Concateante outputs of multiple modules (along filter 
    axis 1).

    Parameters
    ----------
    modules: list of `torch.nn.Module`
        Modules whose outputs will be concatenated.
        Should return same output shape except dim 1 for
        same input shape
    """

    def __init__(self, modules):
        super(Concatenate, self).__init__()
        self.modules = modules

    def forward(self, *x):
        module_outputs = [m(*x) for m in self.modules]
        return th.cat(module_outputs, dim=1)
