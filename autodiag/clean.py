import numpy as np
import torch as th
import torch.nn.functional as F
from braindecode.torch_ext.util import np_to_var


def set_jumps_to_zero(x, window_len, threshold, cuda, clip_min_max_to_zero=True):
    x_var = np_to_var([x])
    if cuda:
        x_var = x_var.cuda()

    maxs = F.max_pool1d(x_var, window_len, stride=1)
    mins = -F.max_pool1d(-x_var, window_len, stride=1)
    if clip_min_max_to_zero:
        maxs = th.clamp(maxs, min=0)
    if clip_min_max_to_zero:
        mins = th.clamp(mins, max=0)

    diffs = maxs - mins

    above_threshold = (diffs > threshold).type_as(diffs)
    padded = F.pad(above_threshold.unsqueeze(0),
                   (window_len - 1, window_len - 1, 0, 0),
                   'constant', 0)
    pad_above_threshold = th.max(padded[:, :, :, window_len - 1:],
                                 padded[:, :, :, :-window_len + 1]).unsqueeze(0)
    x_var = x_var * (1 - pad_above_threshold)

    x = x_var.data.cpu().numpy()[0]
    return x




def clean_jumps(x, window_len, threshold, expected, cuda):
    x_var = np_to_var([x])
    if cuda:
        x_var = x_var.cuda()

    maxs = F.max_pool1d(x_var,window_len, stride=1)
    mins = F.max_pool1d(-x_var,window_len, stride=1)

    diffs = maxs + mins
    large_diffs = (diffs > threshold).type_as(diffs) * diffs
    padded = F.pad(large_diffs.unsqueeze(0), (window_len-1,window_len-1, 0,0), 'constant', 0)
    max_diffs = th.max(padded[:,:,:,window_len-1:], padded[:,:,:,:-window_len+1]).unsqueeze(0)
    max_diffs = th.clamp(max_diffs, min=expected)
    x_var = x_var * (expected / max_diffs)

    x = x_var.data.cpu().numpy()[0]
    return x