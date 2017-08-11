from braindecode.torch_ext.modules import Expression
from torch import nn
import torch as th

def to_linear_plus_minus_net(model):
    model_sigmoid = nn.Sequential()
    for name, module in model.named_children():
        if name != 'softmax':
            model_sigmoid.add_module(name, module)
    model_sigmoid.add_module('to_two_class', Expression(lambda x: th.cat([-x, x],
                                                                         dim=1)))
    return model_sigmoid