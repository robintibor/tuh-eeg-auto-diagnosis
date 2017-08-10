import numpy as np

class RemoveMinMaxDiff(object):
    def __init__(self, threshold, clip_max_abs, set_zero=True):
        self.threshold = threshold
        self.clip_max_abs = clip_max_abs
        self.set_zero = set_zero

    def process(self, inputs, targets):
        # channelwise
        max_per_row = np.max(inputs, axis=(2,3))
        min_per_row = np.min(inputs, axis=(2,3))
        diff = max_per_row - min_per_row
        if self.clip_max_abs:
            diff = np.maximum(np.maximum(max_per_row, diff), -min_per_row)
        if self.set_zero:
            mask = diff < self.threshold
            inputs = inputs * np.float32(mask[:,:,None,None])
        else:
            diff = np.max(diff, axis=1)

            mask = diff < self.threshold
            inputs = inputs[mask]
            targets = targets[mask]
        return inputs, targets