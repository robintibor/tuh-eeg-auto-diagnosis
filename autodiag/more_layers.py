from braindecode.torch_ext.modules import Expression
from torch import nn


def net_with_more_layers(net, n_blocks_to_add,
                        later_pool_class):
    def add_conv_pool_block(model, n_filters_before,
                            n_filters, filter_length, block_nr):
        suffix = '_{:d}'.format(block_nr)
        model.add_module('drop' + suffix,
                         nn.Dropout(p=net.drop_prob))
        model.add_module('conv' + suffix.format(block_nr),
                         nn.Conv2d(n_filters_before, n_filters,
                                   (filter_length, 1),
                                   stride=1, bias=not net.batch_norm))
        if net.batch_norm:
            model.add_module('bnorm' + suffix,
                         nn.BatchNorm2d(n_filters,
                                        momentum=net.batch_norm_alpha,
                                        affine=True,
                                        eps=1e-5))
        model.add_module('nonlin' + suffix,
                         Expression(net.later_nonlin))

        model.add_module('pool' + suffix,
                         later_pool_class(
                             kernel_size=(net.pool_time_length, 1),
                             stride=(net.pool_time_stride, 1)))
        model.add_module('pool_nonlin' + suffix,
                         Expression(net.later_pool_nonlin))

    model = net.create_network()
    from torch import nn
    larger_model = nn.Sequential()
    for name, module in model.named_children():
            if name == 'conv_classifier':
                break
            larger_model.add_module(name, module)
    for i_block in range(n_blocks_to_add):
        add_conv_pool_block(larger_model, net.n_filters_4, net.n_filters_4,
                            10,i_block+5)

    after_classifier = False
    for name, module in model.named_children():
            if name == 'conv_classifier':
                after_classifier = True
            if after_classifier:
                larger_model.add_module(name, module)
    return larger_model