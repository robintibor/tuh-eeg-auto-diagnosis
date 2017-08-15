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
        max_min_threshold, max_min_expected, shrink_val,
        max_min_remove, batch_set_zero_val, batch_set_zero_test,
        sampling_freq,
        low_cut_hz, high_cut_hz,
        exp_demean, exp_standardize,
        moving_demean, moving_standardize,
        channel_demean, channel_standardize,
        divisor,
        n_folds, i_test_fold,
        final_conv_length,
        model_constraint,
        batch_size, max_epochs,
        n_filters_time,
        n_filters_spat,
        filter_time_length,
        conv_nonlin,
        pool_time_length,
        pool_time_stride,
        pool_mode,
        pool_nonlin,
        split_first_layer,
        do_batch_norm,
        drop_prob,
        only_return_exp):
    kwargs = locals()
    for model_param in [
        'final_conv_length',
        'n_filters_time',
        'n_filters_spat',
        'filter_time_length',
        'conv_nonlin',
        'pool_time_length',
        'pool_time_stride',
        'pool_mode',
        'pool_nonlin',
        'split_first_layer',
        'do_batch_norm',
        'drop_prob',
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
    input_time_length = 12000

    # copy over from early seizure
    # make proper
    n_classes = 2
    in_chans = 21
    cuda = True
    set_random_seeds(seed=20170629, cuda=cuda)
    model = ShallowFBCSPNet(
        in_chans=in_chans, n_classes=n_classes,
        input_time_length=input_time_length,
        final_conv_length=final_conv_length,
        n_filters_time=n_filters_time,
        filter_time_length=filter_time_length,
        n_filters_spat=n_filters_spat,
        pool_time_length=pool_time_length,
        pool_time_stride=pool_time_stride,
        conv_nonlin=nonlin_dict[conv_nonlin],
        pool_mode=pool_mode,
        pool_nonlin=nonlin_dict[pool_nonlin],
        split_first_layer=split_first_layer,
        batch_norm=do_batch_norm,
        batch_norm_alpha=0.1,
        drop_prob=drop_prob).create_network()

    to_dense_prediction_model(model)
    if cuda:
        model.cuda()
    test_input = np_to_var(
        np.ones((2, in_chans, input_time_length, 1), dtype=np.float32))
    if cuda:
        test_input = test_input.cuda()

    try:
        out = model(test_input)
    except RuntimeError:
        raise ValueError("Model receptive field too large...")
    n_preds_per_input = out.cpu().data.numpy().shape[2]
    n_receptive_field= input_time_length - n_preds_per_input
    if n_receptive_field > 6000:
        raise ValueError("Model receptive field ({:d}) too large...".format(
            n_receptive_field
        ))
    else:
        input_time_length = 2 * n_receptive_field

    model = ShallowFBCSPNet(
        in_chans=in_chans, n_classes=n_classes,
        input_time_length=input_time_length,
        final_conv_length=final_conv_length,
        n_filters_time=n_filters_time,
        filter_time_length=filter_time_length,
        n_filters_spat=n_filters_spat,
        pool_time_length=pool_time_length,
        pool_time_stride=pool_time_stride,
        conv_nonlin=nonlin_dict[conv_nonlin],
        pool_mode=pool_mode,
        pool_nonlin=nonlin_dict[pool_nonlin],
        split_first_layer=split_first_layer,
        batch_norm=do_batch_norm,
        batch_norm_alpha=0.1,
        drop_prob=drop_prob).create_network()
    return common.run_exp(model=model, input_time_length=input_time_length,
                          **kwargs)


def run(ex, max_recording_mins, n_recordings,
        sec_to_cut, duration_recording_mins, max_abs_val,
        max_min_threshold, max_min_expected, shrink_val,
        max_min_remove, batch_set_zero_val, batch_set_zero_test,
        sampling_freq,
        low_cut_hz, high_cut_hz,
        exp_demean, exp_standardize,
        moving_demean, moving_standardize,
        channel_demean, channel_standardize,
        divisor,
        n_folds, i_test_fold,
        model_constraint,
        batch_size, max_epochs,
        final_conv_length,
        n_filters_time,
        n_filters_spat,
        filter_time_length,
        conv_nonlin,
        pool_time_length,
        pool_time_stride,
        pool_mode,
        pool_nonlin,
        split_first_layer,
        do_batch_norm,
        drop_prob,
        only_return_exp):
    i_test_fold = int(i_test_fold)
    kwargs = locals()
    kwargs.pop('ex')
    import sys
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.DEBUG, stream=sys.stdout)
    start_time = time.time()
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

