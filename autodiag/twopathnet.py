import numpy as np
from torch import nn
from torch.nn import init
import torch as th
import torch
from torch.nn.functional import elu
import torch.nn.functional as F

from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.functions import safe_log, square
from braindecode.torch_ext.util import np_to_var
from braindecode.torch_ext.functions import identity


class TwoPathStartBlock(torch.nn.Module):
    def __init__(self, in_chans, n_start_filters, ):
        super(TwoPathStartBlock, self).__init__()
        self.conv_1a = nn.Conv2d(in_chans, n_start_filters, (32, 1))
        self.bnorm_1a = nn.BatchNorm2d(n_start_filters)
        self.nonlin_1a = Expression(square)
        self.pool_1a = nn.AvgPool2d((42, 1), (8, 1))
        self.pool_nonlin_1a = Expression(safe_log)
        self.conv_1b = nn.Conv2d(21, n_start_filters, (32, 1), stride=(4, 1))
        self.bnorm_1b = nn.BatchNorm2d(n_start_filters)
        self.nonlin_1b = Expression(elu)
        self.conv_2b = nn.Conv2d(n_start_filters, n_start_filters * 2, (12, 1),
                                 stride=(2, 1))
        self.bnorm_2b = nn.BatchNorm2d(n_start_filters * 2)
        self.nonlin_2b = Expression(elu)

    def forward(self, x):
        out_a = self.pool_nonlin_1a(
            self.pool_1a(self.nonlin_1a(self.bnorm_1a(self.conv_1a(x)))))
        out_b = self.nonlin_2b(self.bnorm_2b(
            self.conv_2b(self.nonlin_1b(self.bnorm_1b(self.conv_1b(x))))))
        time_diff = out_a.size()[2] - out_b.size()[2]
        # possibly pad for merge
        if time_diff > 0:
            out_b = F.pad(out_b, (0, 0, 0, time_diff))
        elif time_diff < 0:
            out_a = F.pad(out_a, (0, 0, 0, -time_diff))
        return th.cat([out_a, out_b], dim=1)


def create_two_path_net(in_chans, n_start_filters, later_strides):
    model = nn.Sequential()
    model.add_module('start_block', TwoPathStartBlock(in_chans, n_start_filters))
    model.add_module('conv_3', nn.Conv2d(n_start_filters*3, n_start_filters*2,
                                        (12,1), stride=(later_strides,1)))
    model.add_module('bnorm_3', nn.BatchNorm2d(n_start_filters*2))
    model.add_module('nonlin_3', Expression(elu))
    model.add_module('conv_4', nn.Conv2d(n_start_filters*2, n_start_filters*2,
                                        (12,1), stride=(later_strides,1)))
    model.add_module('bnorm_4', nn.BatchNorm2d(n_start_filters*2))
    model.add_module('nonlin_4', Expression(elu))
    model.add_module('conv_5', nn.Conv2d(n_start_filters*2, n_start_filters*2,
                                        (12,1), stride=(later_strides,1)))
    model.add_module('bnorm_5', nn.BatchNorm2d(n_start_filters*2))
    model.add_module('nonlin_5', Expression(elu))
    model.add_module('classifier', nn.Conv2d(n_start_filters*2, 2, (1,1)))
    model.add_module('mean_class', Expression(lambda x: th.mean(x, dim=2)))
    model.add_module('softmax', nn.LogSoftmax())
    model.add_module('squeeze', Expression(lambda x: x.squeeze(3)))
    return model