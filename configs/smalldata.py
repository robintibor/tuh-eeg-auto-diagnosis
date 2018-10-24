import os

os.sys.path.insert(0, '/home/schirrmr/braindecode/code/')
os.sys.path.insert(0, '/home/schirrmr/braindecode/code/braindecode/')
os.sys.path.insert(0, '/home/schirrmr/code/auto-diagnosis/')
import logging
import time

import numpy as np
import torch as th
from torch import nn

from hyperoptim.parse import cartesian_dict_of_lists_product, \
    product_of_list_of_lists_of_dicts
from hyperoptim.util import save_pkl_artifact, save_npy_artifact
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.util import np_to_var
from braindecode.torch_ext.util import set_random_seeds
from braindecode.torch_ext.modules import Expression
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.torch_ext.util import var_to_np, confirm_gpu_availability

from autodiag.monitors import compute_preds_per_trial
from autodiag.threepathnet import create_multi_start_path_net

import importlib
from contextlib import contextmanager

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
log.setLevel('DEBUG')

def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product
    default_params = [{
        'save_folder': '/data/schirrmr/schirrmr/models/auto-diag/small-data-comparison-resampy-0.2.1-mne-0.16.2-correct-seeds/max-time-ensure-n-rec/',
        'only_return_exp': False,
    }]

    seed_params = dictlistprod({
        'np_th_seed': list(range(0,100))#[0,1,2,3,4]
    })

    save_params = [{
        'save_predictions': False,
        'save_crop_predictions': False,
    }]

    load_params = [{
        'max_recording_mins': 35,
        'n_recordings': 500,
    }]

    clean_params = [{
        'max_abs_val': 800,
    }]

    sensor_params = [{
        'n_chans': 21,
        'sensor_types': ['EEG'],
    },
    ]


    preproc_params = dictlistprod({
        'sec_to_cut_at_start': [60],
        'sec_to_cut_at_end': [0],
        'duration_recording_mins': [3],
        'test_recording_mins': [None],
        'sampling_freq': [100],
        'divisor': [None,], # 10 before
        'clip_before_resample': [False],#False,
    })

    # this differentiates train/test also.
    split_params = dictlistprod({
        'test_on_eval': [False],
        'n_folds': [5],
        'i_test_fold': [4],
        'shuffle': [False],
    })

    model_params = [
    # {
    #     'input_time_length': 6000,
    #     'final_conv_length': 35,
    #     'model_name': 'shallow',
    #     'n_start_chans': 40,
    #     'n_chan_factor': None,
    #     'model_constraint': 'defaultnorm',
    #     'stride_before_pool': None,
    #     'scheduler': None,
    #     'optimizer': 'adam',
    #     'learning_rate': 1e-3,
    #     'weight_decay': 0,
    #     'merge_train_valid': False,
    # },
    # {
    #     'input_time_length': 6000,
    #     'final_conv_length': 1,
    #     'model_name': 'deep',
    #     'n_start_chans': 25,
    #     'n_chan_factor': 2,
    #     'model_constraint': 'defaultnorm',
    #     'stride_before_pool': True,
    #     'scheduler': None,
    #     'optimizer': 'adam',
    #     'learning_rate': 1e-3,
    #     'weight_decay': 0,
    #     'merge_train_valid': False,
    # },
    # {
    #     'input_time_length': 6000,
    #     'final_conv_length': 35,
    #     'model_name': 'shallow',
    #     'n_start_chans': 40,
    #     'n_chan_factor': None,
    #     'model_constraint': None,
    #     'stride_before_pool': None,
    #     'scheduler': 'cosine',
    #     'optimizer': 'adamw',
    #     'learning_rate': 0.0625 * 0.01,
    #     'weight_decay': 0,
    #     'merge_train_valid': True,
    # },
    {
        'input_time_length': 6000,
        'final_conv_length': 1,
        'model_name': 'deep',
        'n_start_chans': 25,
        'n_chan_factor': 2,
        'model_constraint': None,
        'stride_before_pool': True,
        'scheduler': 'cosine',
        'optimizer': 'adamw',
        'learning_rate': 1*0.01,
        'weight_decay': 0.5*0.001,
        'merge_train_valid': True,
    },
    ]


    iterator_params = [{
        'batch_size':  64
    }]

    stop_params = [{
        'max_epochs': 35,
    }]


    grid_params = product_of_list_of_lists_of_dicts([
        default_params,
        seed_params,
        save_params,
        load_params,
        clean_params,
        preproc_params,
        sensor_params,
        split_params,
        model_params,
        iterator_params,
        stop_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    return params


def create_set(X, y, inds):
    """
    X list and y nparray
    :return: 
    """
    new_X = []
    for i in inds:
        new_X.append(X[i])
    new_y = y[inds]
    return SignalAndTarget(new_X, new_y)


def run_exp(test_on_eval,
            sensor_types,
            n_chans,
            max_recording_mins,
            test_recording_mins,
            n_recordings,
            sec_to_cut_at_start,
            sec_to_cut_at_end,
            duration_recording_mins, max_abs_val,
            clip_before_resample,
            sampling_freq,
            divisor,
            n_folds, i_test_fold,
            shuffle,
            merge_train_valid,
            model_name,
            n_start_chans, n_chan_factor,
            input_time_length, final_conv_length,
            stride_before_pool,
            optimizer,
            learning_rate,
            weight_decay,
            scheduler,
            model_constraint,
            batch_size, max_epochs,
            log_dir,
            only_return_exp,
            np_th_seed):

    cuda = True
    if ('smac' in model_name) and (input_time_length is None):
        input_time_length = 12000
        fix_input_length_for_smac = True
    else:
        fix_input_length_for_smac = False
    set_random_seeds(seed=np_th_seed, cuda=cuda)
    n_classes = 2
    if model_name == 'shallow':
        model = ShallowFBCSPNet(in_chans=n_chans, n_classes=n_classes,
                                n_filters_time=n_start_chans,
                                n_filters_spat=n_start_chans,
                                input_time_length=input_time_length,
                                final_conv_length=final_conv_length).create_network()
    elif model_name == 'deep':
        model = Deep4Net(n_chans, n_classes,
                         n_filters_time=n_start_chans,
                         n_filters_spat=n_start_chans,
                         input_time_length=input_time_length,
                         n_filters_2 = int(n_start_chans * n_chan_factor),
                         n_filters_3 = int(n_start_chans * (n_chan_factor ** 2.0)),
                         n_filters_4 = int(n_start_chans * (n_chan_factor ** 3.0)),
                         final_conv_length=final_conv_length,
                        stride_before_pool=stride_before_pool).create_network()
    elif (model_name == 'deep_smac') or (model_name == 'deep_smac_bnorm'):
        if model_name == 'deep_smac':
            do_batch_norm = False
        else:
            assert model_name == 'deep_smac_bnorm'
            do_batch_norm = True
        double_time_convs = False
        drop_prob = 0.244445
        filter_length_2 = 12
        filter_length_3 = 14
        filter_length_4 = 12
        filter_time_length = 21
        final_conv_length = 1
        first_nonlin = elu
        first_pool_mode = 'mean'
        first_pool_nonlin = identity
        later_nonlin = elu
        later_pool_mode = 'mean'
        later_pool_nonlin = identity
        n_filters_factor = 1.679066
        n_filters_start = 32
        pool_time_length = 1
        pool_time_stride = 2
        split_first_layer = True
        n_chan_factor = n_filters_factor
        n_start_chans = n_filters_start
        model = Deep4Net(n_chans, n_classes,
                 n_filters_time=n_start_chans,
                 n_filters_spat=n_start_chans,
                 input_time_length=input_time_length,
                 n_filters_2=int(n_start_chans * n_chan_factor),
                 n_filters_3=int(n_start_chans * (n_chan_factor ** 2.0)),
                 n_filters_4=int(n_start_chans * (n_chan_factor ** 3.0)),
                 final_conv_length=final_conv_length,
                 batch_norm=do_batch_norm,
                 double_time_convs=double_time_convs,
                 drop_prob=drop_prob,
                 filter_length_2=filter_length_2,
                 filter_length_3=filter_length_3,
                 filter_length_4=filter_length_4,
                 filter_time_length=filter_time_length,
                 first_nonlin=first_nonlin,
                 first_pool_mode=first_pool_mode,
                 first_pool_nonlin=first_pool_nonlin,
                 later_nonlin=later_nonlin,
                 later_pool_mode=later_pool_mode,
                 later_pool_nonlin=later_pool_nonlin,
                 pool_time_length=pool_time_length,
                 pool_time_stride=pool_time_stride,
                 split_first_layer=split_first_layer,
                 stride_before_pool=True).create_network()
    elif model_name == 'shallow_smac':
        conv_nonlin = identity
        do_batch_norm = True
        drop_prob = 0.328794
        filter_time_length = 56
        final_conv_length = 22
        n_filters_spat = 73
        n_filters_time = 24
        pool_mode = 'max'
        pool_nonlin = identity
        pool_time_length = 84
        pool_time_stride = 3
        split_first_layer = True
        model = ShallowFBCSPNet(in_chans=n_chans, n_classes=n_classes,
                                n_filters_time=n_filters_time,
                                n_filters_spat=n_filters_spat,
                                input_time_length=input_time_length,
                                final_conv_length=final_conv_length,
                                conv_nonlin=conv_nonlin,
                                batch_norm=do_batch_norm,
                                drop_prob=drop_prob,
                                filter_time_length=filter_time_length,
                                pool_mode=pool_mode,
                                pool_nonlin=pool_nonlin,
                                pool_time_length=pool_time_length,
                                pool_time_stride=pool_time_stride,
                                split_first_layer=split_first_layer,
                                ).create_network()
    elif model_name == 'deep_smac_new':
        from torch.nn.functional import elu, relu, relu6, tanh
        from braindecode.torch_ext.functions import identity, square, safe_log
        n_filters_factor = 1.9532637176784269
        n_filters_start = 61

        deep_kwargs = {
            "batch_norm": False,
            "double_time_convs": False,
            "drop_prob": 0.3622676569047184,
            "filter_length_2": 9,
            "filter_length_3": 6,
            "filter_length_4": 10,
            "filter_time_length": 17,
            "final_conv_length": 5,
            "first_nonlin": elu,
            "first_pool_mode": "max",
            "first_pool_nonlin": identity,
            "later_nonlin": relu6,
            "later_pool_mode": "max",
            "later_pool_nonlin": identity,
            "n_filters_time": n_filters_start,
            "n_filters_spat": n_filters_start,
            "n_filters_2": int(n_filters_start * n_filters_factor),
            "n_filters_3": int(n_filters_start * (n_filters_factor ** 2.0)),
            "n_filters_4": int(n_filters_start * (n_filters_factor ** 3.0)),
            "pool_time_length": 1,
            "pool_time_stride": 4,
            "split_first_layer": True,
            "stride_before_pool": True,
        }

        model = Deep4Net(n_chans, n_classes, input_time_length=input_time_length,
                         **deep_kwargs).create_network()
    elif model_name == 'shallow_smac_new':
        from torch.nn.functional import elu, relu, relu6, tanh
        from braindecode.torch_ext.functions import identity, square, safe_log
        shallow_kwargs = {
            "conv_nonlin": square,
            "batch_norm": True,
            "drop_prob": 0.10198630723385381,
            "filter_time_length": 51,
            "final_conv_length": 1,
            "n_filters_spat": 200,
            "n_filters_time": 76,
            "pool_mode": "max",
            "pool_nonlin": safe_log,
            "pool_time_length": 139,
            "pool_time_stride": 49,
            "split_first_layer": True,
        }

        model = ShallowFBCSPNet(in_chans=n_chans, n_classes=n_classes,
                                input_time_length=input_time_length,
                                **shallow_kwargs
                                ).create_network()
    elif model_name == 'linear':
        model = nn.Sequential()
        model.add_module("conv_classifier",
                         nn.Conv2d(n_chans, n_classes, (600,1)))
        model.add_module('softmax', nn.LogSoftmax())
        model.add_module('squeeze', Expression(lambda x: x.squeeze(3)))
    elif model_name == '3path':
        virtual_chan_1x1_conv = True
        mean_across_features = False
        drop_prob = 0.5
        n_start_filters = 10
        early_bnorm = False
        n_classifier_filters = 100
        later_kernel_len = 5
        extra_conv_stride = 4
        # dont forget to reset n_preds_per_blabla
        model = create_multi_start_path_net(
            in_chans=n_chans,
            virtual_chan_1x1_conv=virtual_chan_1x1_conv,
            n_start_filters=n_start_filters, early_bnorm=early_bnorm,
            later_kernel_len=later_kernel_len,
            extra_conv_stride=extra_conv_stride,
            mean_across_features=mean_across_features,
            n_classifier_filters=n_classifier_filters, drop_prob=drop_prob)
    else:
        assert False, "unknown model name {:s}".format(model_name)
    if not model_name == '3path':
        to_dense_prediction_model(model)
    log.info("Model:\n{:s}".format(str(model)))
    time_cut_off_sec = np.inf
    start_time = time.time()

    # fix input time length in case of smac models
    if fix_input_length_for_smac:
        assert ('smac' in model_name) and (input_time_length == 12000)
        if cuda:
            model.cuda()
        test_input = np_to_var(
            np.ones((2, n_chans, input_time_length, 1), dtype=np.float32))
        if cuda:
            test_input = test_input.cuda()
        try:
            out = model(test_input)
        except:
            raise ValueError("Model receptive field too large...")
        n_preds_per_input = out.cpu().data.numpy().shape[2]
        n_receptive_field = input_time_length - n_preds_per_input
        input_time_length = 2 * n_receptive_field

    exp = common.run_exp(
        max_recording_mins, n_recordings,
        sec_to_cut_at_start, sec_to_cut_at_end,
        duration_recording_mins, max_abs_val,
        clip_before_resample,
        sampling_freq,
        divisor,
        n_folds, i_test_fold,
        shuffle,
        merge_train_valid,
        model,
        input_time_length,
        optimizer,
        learning_rate,
        weight_decay,
        scheduler,
        model_constraint,
        batch_size, max_epochs,
        only_return_exp,
        time_cut_off_sec,
        start_time,
        test_on_eval,
        test_recording_mins,
        sensor_types,
        log_dir,
        np_th_seed,)

    return exp


def save_torch_artifact(ex, obj, filename):
    """Uses tempfile and file lock to safely store a pkl object as artefact"""
    import tempfile
    import fasteners
    log.info("Saving torch artifact")
    with tempfile.NamedTemporaryFile(suffix='.pkl') as tmpfile:
        lockname = tmpfile.name + '.lock'
        file_lock = fasteners.InterProcessLock(lockname)
        file_lock.acquire()
        th.save(obj, open(tmpfile.name, 'wb'))
        ex.add_artifact(tmpfile.name, filename)
        file_lock.release()
    log.info("Saved torch artifact")


def run(ex,
        test_on_eval,
        sensor_types,
        n_chans,
        max_recording_mins, n_recordings,
        sec_to_cut_at_start, sec_to_cut_at_end,
        duration_recording_mins,
        test_recording_mins,
        max_abs_val,
        clip_before_resample,
        sampling_freq,
        divisor,
        n_folds, i_test_fold,
        shuffle,
        merge_train_valid,
        model_name, input_time_length, final_conv_length,
        stride_before_pool,
        n_start_chans, n_chan_factor,
        optimizer,
        learning_rate,
        weight_decay,
        scheduler,
        model_constraint,
        batch_size, max_epochs,
        save_predictions,
        save_crop_predictions,
        np_th_seed,
        only_return_exp):
    log_dir =  ex.observers[0].dir
    kwargs = locals()
    kwargs.pop('ex')
    kwargs.pop('save_predictions')
    kwargs.pop('save_crop_predictions')
    import sys
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.DEBUG, stream=sys.stdout)
    start_time = time.time()
    ex.info['finished'] = False
    confirm_gpu_availability()

    exp = run_exp(**kwargs)
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
        save_torch_artifact(ex, exp.model.state_dict(), 'model_params.pkl')
        if save_predictions:
            exp.model.eval()
            for setname in ('train', 'valid', 'test'):
                log.info("Compute and save predictions for {:s}...".format(
                    setname))
                dataset = exp.datasets[setname]
                log.info("Save labels for {:s}...".format(
                    setname))
                save_npy_artifact(ex, dataset.y,
                                  '{:s}_trial_labels.npy'.format(setname))
                preds_per_batch = [var_to_np(exp.model(np_to_var(b[0]).cuda()))
                          for b in exp.iterator.get_batches(dataset, shuffle=False)]
                preds_per_trial = compute_preds_per_trial(
                    preds_per_batch, dataset,
                    input_time_length=exp.iterator.input_time_length,
                    n_stride=exp.iterator.n_preds_per_input)
                mean_preds_per_trial = [np.mean(preds, axis=(0, 2)) for preds in
                                            preds_per_trial]
                mean_preds_per_trial = np.array(mean_preds_per_trial)
                log.info("Save trial predictions for {:s}...".format(
                    setname))
                save_npy_artifact(ex, mean_preds_per_trial,
                                  '{:s}_trial_preds.npy'.format(setname))
                if save_crop_predictions:
                    log.info("Save crop predictions for {:s}...".format(
                        setname))
                    save_npy_artifact(ex, preds_per_trial,
                                      '{:s}_crop_preds.npy'.format(setname))

    else:
        return exp
