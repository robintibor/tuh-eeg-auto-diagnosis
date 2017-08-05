# Written by Manuel Blum
# Adapted by Lukas Gemein
import numpy as _np

# TODO: implement more features
# TODO: TAKE CARE! DIFF INSERTS ANOTHER ROW OF NaNs!
# TODO: Maybe pad the signal in the beginning with 0s for each time diff is called?
# TODO: find a clean way to fix this


# Imagine that you have zero cookies, and you split them evenly among 
# zero friends. How many cookies does each person get? See? It doesn't 
# make sense. And Cookie Monster is sad that there are no cookies, and 
# you are sad that you have no friends.


def _div0(x):
    x[x < 1e-9] = 1e-9
    return x


def mean(df, window_size):
    return df.rolling(window=window_size).mean()


def median(df, window_size):
    return df.rolling(window=window_size).median()


def var(df, window_size):
    return df.rolling(window=window_size).var()


def linelength(df, window_size):
    window_size -= 1
    return df.diff().abs().rolling(window=window_size).sum()


def min(df, window_size):
    return df.rolling(window=window_size).min()


def max(df, window_size):
    return df.rolling(window=window_size).max()


def skew(df, window_size):
    return df.rolling(window=window_size).skew()


def kurt(df, window_size):
    return df.rolling(window=window_size).kurt()


def energy(df, window_size):
    return (df * df).rolling(window=window_size).mean()


def complexity(df, window_size):
    window_size -= 2
    diff1 = df.diff()
    diff2 = diff1.diff()
    sigma = df.rolling(window=window_size).std()
    sigma1 = diff1.rolling(window=window_size).std()
    sigma2 = diff2.rolling(window=window_size).std()
    return (sigma2 / _div0(sigma1)) / _div0((sigma1 / _div0(sigma)))


def mobility(df, window_size):
    window_size -= 1
    return df.diff().rolling(window=window_size).std() / _div0(df.rolling(window=window_size).std())


def nonlinenergy(df, window_size):
    return (_np.square(df[1:-1]) - df[2:].values * df[:-2].values).rolling(window=window_size).mean()


def fractaldim(df, window_size):
    window_size -= 2
    sum_of_distances = _np.sqrt(df.diff() * df.diff()).rolling(window=window_size).sum()
    max_dist = df.rolling(window=window_size).apply(lambda df: _np.max(_np.sqrt(_np.square(df - df[0]))))
    return _np.log10(sum_of_distances) / _div0(_np.log10(max_dist))


def zerocrossing(df, window_size):
    # i don't know why this mean subtraction was performed before
    # norm = df - df.rolling(window=window_size).mean()
    # also i don't know why before there was an epsilon with .1 rather than 0

    # first boolean expression gives a data frame with 1 whenever entry is smaller or equal 0
    # second expression gives ndarray with 1 whenever entry is bigger 0
    # hence, they are complementary
    # then count the number of smaller 0 and bigger 0 traverses on subsequent values
    return ((df[:-1] <= 0) & (df[1:].values > 0)).rolling(window=window_size).sum()


def zerocrossingdev(df, window_size):
    window_size -= 1
    diff = df.diff()
    # norm = diff - diff.rolling(window=window_size).mean()
    return ((diff[:-1] <= 0) & (diff[1:].values > 0)).rolling(window=window_size).sum()
