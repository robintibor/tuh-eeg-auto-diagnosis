import numpy as _np

# TODO: implement more features


def max(amplitudes, axis=-1):
    return _np.max(amplitudes, axis=axis)


def min(amplitudes, axis=-1):
    return _np.min(amplitudes, axis=axis)


def ptp(amplitudes, axis=-1):
    return _np.ptp(amplitudes, axis=axis)


def median(amplitudes, axis=-1):
    return _np.median(amplitudes, axis=axis)


def std(amplitudes, axis=-1):
    return _np.std(amplitudes, axis=axis)


def mean(amplitudes, axis=-1):
    return _np.mean(amplitudes, axis=axis)


def energy(amplitudes, axis=-1):
    return _np.mean(amplitudes * amplitudes, axis=axis)


def power(amplitudes, axis=-1):
    return _np.sum(amplitudes * amplitudes, axis=axis)


def var(amplitudes, axis=-1):
    return _np.var(amplitudes, axis=axis)


def entropy(amplitudes, axis=-1):
    squared = amplitudes * amplitudes
    return -(_np.sum(squared * _np.log2(squared), axis=axis))


def boundedvariation(amplitudes, axis=-1):
    diffs = _np.diff(amplitudes, axis=axis)
    abss = _np.abs(diffs)
    sums = _np.sum(abss, axis=axis)
    return _np.divide(sums, _np.max(amplitudes, axis=axis) - _np.min(amplitudes, axis=axis))
