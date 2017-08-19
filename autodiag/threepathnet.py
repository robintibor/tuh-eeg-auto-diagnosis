import numpy as np
from torch import nn
from torch.nn import init
import torch as th
import torch
from torch.nn.functional import elu

from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.functions import safe_log, square
from braindecode.torch_ext.util import np_to_var
from braindecode.torch_ext.functions import identity

from autodiag.functions import sat_elu


class SplitStartBlock(torch.nn.Module):
    def __init__(self, n_start_filters, early_bnorm):
        super(SplitStartBlock, self).__init__()
        self.conv_1a = nn.Conv2d(1, n_start_filters, (25, 1),
                                 padding=(12, 0),
                                 stride=(4, 1))
        self.conv_1b = nn.Conv2d(1, n_start_filters, (25, 1),
                                 padding=(12, 0))
        self.early_bnorm = early_bnorm
        if self.early_bnorm:
            self.bnorm_1a = nn.BatchNorm2d(n_start_filters, )
            self.bnorm_1b = nn.BatchNorm2d(n_start_filters, )
        else:
            self.bnorm_1a = identity
            self.bnorm_1b = identity

        self.nonlin_1b = Expression(square)
        self.pool_1b = nn.AvgPool2d((41, 1), stride=(8, 1), padding=(20, 0))
        self.poolnonlin_1b = Expression(safe_log)
        self.conv_2c = nn.Conv2d(n_start_filters, n_start_filters, (25, 1),
                                 padding=(12, 0))
        if early_bnorm:
            self.bnorm_2c = nn.BatchNorm2d(n_start_filters, )
        else:
            self.bnorm_2c = identity

        self.nonlin_2c = Expression(square)
        self.pool_2c = nn.AvgPool2d((11, 1), stride=(2, 1), padding=(5, 0))
        self.poolnonlin_2c = Expression(safe_log)

        self.conv_2a = nn.Conv2d(n_start_filters, n_start_filters, (11, 1),
                                 stride=(2, 1), padding=(5, 0))

        if early_bnorm:
            self.bnorm_2a = nn.BatchNorm2d(n_start_filters, )
        else:
            self.bnorm_2a = identity

    def forward(self, x):
        out_conv_1a = self.bnorm_1a(self.conv_1a(x))
        out_a = self.bnorm_2a(self.conv_2a(out_conv_1a))
        out_b = self.poolnonlin_1b(
            self.pool_1b(self.nonlin_1b(self.bnorm_1b(self.conv_1b(x)))))
        out_c = self.poolnonlin_2c(self.pool_2c(
            self.nonlin_2c(self.bnorm_2c(self.conv_2c(out_conv_1a)))))

        return th.cat([out_a, out_b, out_c], dim=1)


def create_multi_start_path_net(in_chans,
        virtual_chan_1x1_conv, n_start_filters, early_bnorm,
        later_kernel_len,
        extra_conv_stride, mean_across_features, n_classifier_filters, drop_prob):
    model = nn.Sequential()
    if virtual_chan_1x1_conv:
        model.add_module('virtual_chan_filter',
                     nn.Conv2d(in_chans,in_chans, (1,1)))
        # maybe problem that there is no batch norm?
        # for the gradients etc.?
    model.add_module('dimshuffle',
                                 Expression(lambda x: x.permute(0, 3, 2, 1)))
    model.add_module('start_block', SplitStartBlock(n_start_filters, early_bnorm))
    model.add_module('conv_3', nn.Conv2d(n_start_filters*3, 48, (1,in_chans)))
    model.add_module('bnorm_3', nn.BatchNorm2d(48))
    model.add_module('nonlin_3', Expression(lambda x: sat_elu(x, threshold=7)))
    if extra_conv_stride is not None:
        model.add_module('conv_3_extra', nn.Conv2d(48, 48,
                                                   (extra_conv_stride,1),
                                                   stride=(extra_conv_stride,1)))
    if drop_prob > 0:
         model.add_module('conv_4_drop', nn.Dropout(p=drop_prob))

    model.add_module('conv_4', nn.Conv2d(48, 48, (later_kernel_len,1)))
    model.add_module('bnorm_4', nn.BatchNorm2d(48))
    model.add_module('nonlin_4', Expression(elu))
    if extra_conv_stride is not None:
        model.add_module('conv_4_extra', nn.Conv2d(48, 48,
                                                   (extra_conv_stride,1),
                                                   stride=(extra_conv_stride,1)))
    if drop_prob > 0:
         model.add_module('conv_5_drop', nn.Dropout(p=drop_prob))

    model.add_module('conv_5', nn.Conv2d(48, 64, (later_kernel_len,1)))
    model.add_module('bnorm_5', nn.BatchNorm2d(64))
    model.add_module('nonlin_5', Expression(elu))

    if n_classifier_filters is not None:
        if drop_prob > 0:
             model.add_module('conv_features_drop', nn.Dropout(p=drop_prob))
        model.add_module('conv_features', nn.Conv2d(64, n_classifier_filters,
                                                    (1,1)))
        model.add_module('bnorm_features', nn.BatchNorm2d(
            n_classifier_filters))
        model.add_module('nonlin_features', Expression(elu))
    if drop_prob > 0:
         model.add_module('classifier_drop', nn.Dropout(p=drop_prob))
    if mean_across_features:
        model.add_module('feature_mean', Expression(lambda x: th.mean(x, dim=2)))
    n_features_now = n_classifier_filters
    if n_features_now is None:
        n_features_now = 64
    model.add_module('classifier', nn.Conv2d(n_features_now, 2, (1,1)))
    model.add_module('softmax', nn.LogSoftmax())
    model.add_module('squeeze', Expression(lambda x: x.squeeze(3)))
    return model
