from torch.nn.functional import elu
import torch as th


def th_where(cond, x_1, x_2):
    cond = cond.type_as(x_1)
    return (cond * x_1) + ((1 - cond) * x_2)


def sat_elu(x, threshold=7):
    """Saturating Exponential Linear Unit...."""
    # have to clamp x in second case to prevent
    # NaNs
    return th_where(x < threshold, elu(x),
                    threshold + th.log1p(th.clamp(x,min=threshold) -
                                         threshold))
