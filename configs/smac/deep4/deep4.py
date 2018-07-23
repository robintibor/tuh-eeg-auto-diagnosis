from contextlib import contextmanager
import importlib
import os.path
import sys
import logging
import time
from hyperoptim.parse import cartesian_dict_of_lists_product, \
    product_of_list_of_lists_of_dicts
from hyperoptim.util import save_pkl_artifact
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.models.util import to_dense_prediction_model
from braindecode.torch_ext.util import set_random_seeds, np_to_var
import numpy as np
from torch.nn.functional import elu, relu, relu6, tanh
from braindecode.torch_ext.functions import identity, square, safe_log


@contextmanager
def add_to_path(p):
    import sys
    old_path = sys.path
    sys.path = sys.path[:]
    sys.path.insert(0, p)
    try:
        yield
    finally:
        sys.path = old_path

def path_import(absolute_path):
   '''implementation taken from https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly'''
   with add_to_path(os.path.dirname(absolute_path)):
       spec = importlib.util.spec_from_file_location(absolute_path, absolute_path)
       module = importlib.util.module_from_spec(spec)
       spec.loader.exec_module(module)
       return module

common = path_import('/home/schirrmr/code/auto-diagnosis/configs/common.py')
log = logging.getLogger(__name__)

def get_templates():
    return {}

def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product
    default_params = [{
        'save_folder': './data/models/pytorch/auto-diag/smac-test/',
        'only_return_exp': False,
    }]

    load_params = [{
        'max_recording_mins': 35,
        'n_recordings': 5,
    }]

    clean_defaults = {
        'max_min_threshold': None,
        'shrink_val': None,
        'max_min_expected': None,
        'max_abs_val': None,
        'batch_set_zero_val': None,
        'batch_set_zero_test': None,
        'max_min_remove': None,
    }

    clean_variants = [
        #{},
        #{'batch_set_zero_val': 500, 'batch_set_zero_test': True},
        {'max_abs_val' : 800},
        #{'max_abs_val' : 500},
        #{'shrink_val': 200},
        #{'shrink_val': 500},
        # {'shrink_val': 800},
    ]

    clean_params = product_of_list_of_lists_of_dicts(
        [[clean_defaults], clean_variants])


    preproc_params = dictlistprod({
        'sec_to_cut': [60],
        'duration_recording_mins': [3],
        'sampling_freq': [100],
        'low_cut_hz': [None,],
        'high_cut_hz': [None,],
        'divisor': [10],
    })

    standardizing_defaults = {
        'exp_demean': False,
        'exp_standardize': False,
        'moving_demean': False,
        'moving_standardize': False,
        'channel_demean': False,
        'channel_standardize': False,
    }

    standardizing_variants = [
        {},
    ]

    standardizing_params = product_of_list_of_lists_of_dicts(
        [[standardizing_defaults], standardizing_variants])

    split_params = dictlistprod({
        'n_folds': [5],
        'i_test_fold': [0],
    })

    model_params = [
    {
        'input_time_length': 1200,
        'final_conv_length': 40,
        'model_name': 'shallow',
    },
    ]

    model_constraint_params = dictlistprod({
        'model_constraint': ['defaultnorm', None],

    })

    iterator_params = [{
        'batch_size':  64
    }]

    stop_params = [{
        'max_epochs': 3,
    }]



    grid_params = product_of_list_of_lists_of_dicts([
        default_params,
        load_params,
        clean_params,
        preproc_params,
        split_params,
        model_params,
        iterator_params,
        standardizing_params,
        stop_params,
        model_constraint_params
    ])

    return grid_params

def sample_config_params(rng, params):
    return params


def run_exp(
        max_recording_mins, n_recordings,
        sec_to_cut, duration_recording_mins, max_abs_val,
        shrink_val,
        sampling_freq,
        divisor,
        n_folds, i_test_fold,
        final_conv_length,
        model_constraint,
        batch_size, max_epochs,
        n_filters_start,
        n_filters_factor,
        filter_time_length,
        first_nonlin,
        first_pool_mode,
        first_pool_nonlin,
        pool_time_stride,
        pool_time_length,
        drop_prob,
        filter_length_2,
        later_nonlin,
        later_pool_mode,
        later_pool_nonlin,
        filter_length_3,
        filter_length_4,
        double_time_convs,
        split_first_layer,
        do_batch_norm,
        stride_before_pool,
        input_time_length,
        only_return_exp,
        time_cut_off_sec,
        start_time):
    kwargs = locals()
    for model_param in [
        'final_conv_length',
        'n_filters_start',
        'n_filters_factor',
        'filter_time_length',
        'first_nonlin',
        'first_pool_mode',
        'first_pool_nonlin',
        'pool_time_stride',
        'pool_time_length',
        'drop_prob',
        'filter_length_2',
        'later_nonlin',
        'later_pool_mode',
        'later_pool_nonlin',
        'filter_length_3',
        'filter_length_4',
        'double_time_convs',
        'split_first_layer',
        'do_batch_norm',
        'stride_before_pool'
    ]:
        kwargs.pop(model_param)
    nonlin_dict = {
        'elu': elu,
        'relu': relu,
        'relu6': relu6,
        'tanh': tanh,
        'square': square,
        'identity': identity,
        'log': safe_log,
    }
    assert input_time_length == 6000

    n_classes = 2
    in_chans = 21
    cuda = True
    set_random_seeds(seed=20170629, cuda=cuda)
    model = Deep4Net(
        in_chans=in_chans, n_classes=n_classes,
        input_time_length=input_time_length,
        final_conv_length=final_conv_length,
        n_filters_time=n_filters_start,
        n_filters_spat=n_filters_start,
        filter_time_length=filter_time_length,
        pool_time_length=pool_time_length,
        pool_time_stride=pool_time_stride,
        n_filters_2= int(n_filters_start * n_filters_factor),
        filter_length_2=filter_length_2,
        n_filters_3=int(n_filters_start * (n_filters_factor ** 2.0)),
        filter_length_3=filter_length_3,
        n_filters_4= int(n_filters_start * (n_filters_factor ** 3.0)),
        filter_length_4=filter_length_4,
        first_nonlin=nonlin_dict[first_nonlin],
        first_pool_mode=first_pool_mode,
        first_pool_nonlin=nonlin_dict[first_pool_nonlin],
        later_nonlin=nonlin_dict[later_nonlin],
        later_pool_mode=later_pool_mode,
        later_pool_nonlin=nonlin_dict[later_pool_nonlin],
        drop_prob=drop_prob,
        double_time_convs=double_time_convs,
        split_first_layer=split_first_layer,
        batch_norm=do_batch_norm,
        batch_norm_alpha=0.1,
        stride_before_pool=stride_before_pool).create_network()

    to_dense_prediction_model(model)
    if cuda:
        model.cuda()
    test_input = np_to_var(
        np.ones((2, in_chans, input_time_length, 1), dtype=np.float32))
    if cuda:
        test_input = test_input.cuda()
    try:
        out = model(test_input)
    except:
        raise ValueError("Model receptive field too large...")
    n_preds_per_input = out.cpu().data.numpy().shape[2]
    n_receptive_field= input_time_length - n_preds_per_input
    if n_receptive_field > 6000:
        raise ValueError("Model receptive field ({:d}) too large...".format(
            n_receptive_field
        ))
        # For future, here optionally add input time length instead

    model = Deep4Net(
        in_chans=in_chans, n_classes=n_classes,
        input_time_length=input_time_length,
        final_conv_length=final_conv_length,
        n_filters_time=n_filters_start,
        n_filters_spat=n_filters_start,
        filter_time_length=filter_time_length,
        pool_time_length=pool_time_length,
        pool_time_stride=pool_time_stride,
        n_filters_2= int(n_filters_start * n_filters_factor),
        filter_length_2=filter_length_2,
        n_filters_3=int(n_filters_start * (n_filters_factor ** 2.0)),
        filter_length_3=filter_length_3,
        n_filters_4= int(n_filters_start * (n_filters_factor ** 3.0)),
        filter_length_4=filter_length_4,
        first_nonlin=nonlin_dict[first_nonlin],
        first_pool_mode=first_pool_mode,
        first_pool_nonlin=nonlin_dict[first_pool_nonlin],
        later_nonlin=nonlin_dict[later_nonlin],
        later_pool_mode=later_pool_mode,
        later_pool_nonlin=nonlin_dict[later_pool_nonlin],
        drop_prob=drop_prob,
        double_time_convs=double_time_convs,
        split_first_layer=split_first_layer,
        batch_norm=do_batch_norm,
        batch_norm_alpha=0.1,
        stride_before_pool=stride_before_pool).create_network()
    return common.run_exp(model=model, **kwargs)


def run(ex, max_recording_mins, n_recordings,
        sec_to_cut, duration_recording_mins, max_abs_val,
        shrink_val,
        sampling_freq,
        divisor,
        n_folds, i_test_fold,
        model_constraint,
        batch_size, max_epochs,
        final_conv_length,
        n_filters_start,
        n_filters_factor,
        filter_time_length,
        first_nonlin,
        first_pool_mode,
        first_pool_nonlin,
        pool_time_stride,
        pool_time_length,
        drop_prob,
        filter_length_2,
        later_nonlin,
        later_pool_mode,
        later_pool_nonlin,
        filter_length_3,
        filter_length_4,
        double_time_convs,
        split_first_layer,
        do_batch_norm,
        stride_before_pool,
        input_time_length,
        time_cut_off_sec,
        start_time,
        only_return_exp):
    i_test_fold = int(i_test_fold)
    kwargs = locals()
    kwargs.pop('ex')
    import sys
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.DEBUG, stream=sys.stdout)
    ex.info['finished'] = False


    exp = run_exp(**kwargs)
    # in case of too large model
    if exp is None:
        ex.score = 1
        return ex
    end_time = time.time()
    run_time = end_time - start_time
    ex.info['finished'] = True

    if not only_return_exp:
        last_row = exp.epochs_df.iloc[-1]
        for key, val in last_row.iteritems():
            ex.info[key] = float(val)
    ex.info['runtime'] = run_time
    if not only_return_exp:
        save_pkl_artifact(ex, exp.epochs_df, 'epochs_df.pkl')
        save_pkl_artifact(ex, exp.before_stop_df, 'before_stop_df.pkl')
    ex.score = float(exp.epochs_df.iloc[-1]['test_misclass'])
    return ex

