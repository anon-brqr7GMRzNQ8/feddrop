'''
Copyright (C) 2019 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import sys
from functools import partial
import copy
import numpy as np
import torch
import torch.nn as nn
from ..models import LayerDropout
from ..pretty import log
from collections import OrderedDict

def get_model_complexity_info(
        model, input_res, print_per_layer_stat=False, as_strings=True,
        input_constructor=None, ost=sys.stdout,
        ignore_modules=[], custom_modules_hooks={}):
    assert type(input_res) is tuple
    assert len(input_res) >= 1
    assert isinstance(model, nn.Module)
    global CUSTOM_MODULES_MAPPING
    CUSTOM_MODULES_MAPPING = custom_modules_hooks

    global drop_prob  # this global variable won't be shared among multiple processings
    global prob_hooks # gather hooks so that we can remove them
    global ignore_list
    global orig_flops
    drop_prob = 1. # init the input dropout prob for the first conv layer
    prob_hooks = []
    ignore_list = []
    orig_flops = []

    model.apply(get_ignore_modules)
    ignore_modules = ignore_list
    # log.debug(f"ignore modules: {ignore_modules}")

    model.apply(register_get_convdrop_probs_hook)

    flops_model = add_flops_counting_methods(model)
    flops_model.eval()
    flops_model.start_flops_count(ost=ost, ignore_list=ignore_modules)

    if input_constructor:
        input = input_constructor(input_res)
        _ = flops_model(**input)
    else:
        try:
            batch = torch.ones(()).new_empty(
                (1, *input_res),
                dtype=next(flops_model.parameters()).dtype,
                device=next(flops_model.parameters()).device)
        except StopIteration:
            batch = torch.ones(()).new_empty((1, *input_res))

        _ = flops_model(batch)

    macs_count, params_count = flops_model.compute_average_flops_cost()
    if print_per_layer_stat:
        print_model_with_flops(flops_model, macs_count, params_count, ost=ost)
    flops_model.stop_flops_count()
    CUSTOM_MODULES_MAPPING = {}
    remove_convdrop_probs_hook(prob_hooks)
    orig_flops_result = get_orig_flops_result(model, orig_flops)
    if as_strings:
        return flops_to_string(macs_count), params_to_string(params_count), orig_flops_result
    return macs_count, params_count, orig_flops_result


def get_orig_flops_result(model, flops_list):
    flops_list.reverse()
    pre = None
    results = []
    for k in model.state_dict().keys():
        if 'probs' in k:
            kk = (k, ) if pre is None else (pre, k)
            results.append((kk, flops_list.pop()))
            pre = k
    return results


def get_ignore_modules(module):
    """ if replace conv with dropconv, make sure conv is in ignore list
    """
    global ignore_list
    if isinstance(module, LayerDropout):
        if len(ignore_list) == 0:
           ignore_list.append(torch.nn.Conv2d)
           ignore_list.append(torch.nn.ReLU)
           ignore_list.append(torch.nn.AdaptiveAvgPool2d)
           ignore_list.append(torch.nn.MaxPool2d)



def check_modules(module, droplist):
    if isinstance(module, LayerDropout):
        droplist.append(module.__name__)


def flops_to_string(flops, units='GMac', precision=2):
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GMac'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MMac'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'


def params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10 ** 6 > 0:
            return str(round(params_num / 10 ** 6, 2)) + ' M'
        elif params_num // 10 ** 3:
            return str(round(params_num / 10 ** 3, 2)) + ' k'
        else:
            return str(params_num)
    else:
        if units == 'M':
            return str(round(params_num / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(params_num / 10.**3, precision)) + ' ' + units
        else:
            return str(params_num)


def accumulate_flops(self):
    if is_supported_instance(self):
        return self.__flops__
    else:
        sum = 0
        for m in self.children():
            sum += m.accumulate_flops()
        return sum


def print_model_with_flops(
        model, total_flops, total_params, units='GMac',
        precision=3, ost=sys.stdout):

    def accumulate_params(self):
        if is_supported_instance(self):
            return self.__params__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_params()
            return sum

    def flops_repr(self):
        accumulated_params_num = self.accumulate_params()
        accumulated_flops_cost = self.accumulate_flops() / model.__batch_counter__
        probs = self.probs_repr()
        return ', '.join([
            params_to_string(
                accumulated_params_num, units='M', precision=precision),
            '{:.3%} Params'.format(accumulated_params_num / total_params),
            flops_to_string(
                accumulated_flops_cost, units=units, precision=precision),
            '{:.3%} MACs'.format(accumulated_flops_cost / total_flops),
            probs,
            self.original_extra_repr()])

    def probs_repr(self):
        probs = ''
        if is_supported_instance(self):
            if isinstance(self, LayerDropout):
                probs = 'pre_prob:{:.3%}, cur_prob:{:.3%}'.format(self.__pre_prob__, self.__cur_prob__)
        return probs

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        m.accumulate_params = accumulate_params.__get__(m)
        m.probs_repr = probs_repr.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        repr_list = ['original_extra_repr', 'accumulate_flops', 'probs_repr']
        for r in repr_list:
            if hasattr(m, r):
                if r == 'original_extra_repr':
                    m.extra_repr = m.original_extra_repr
                delattr(m, r)

    model.apply(add_extra_repr)
    log.debug(repr(model))
    model.apply(del_extra_repr)


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def add_flops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = \
        compute_average_flops_cost.__get__(net_main_module)

    net_main_module.reset_flops_count()

    return net_main_module


def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Returns current mean flops consumption per image.

    """

    for m in self.modules():
        m.accumulate_flops = accumulate_flops.__get__(m)
    flops_sum = self.accumulate_flops()
    for m in self.modules():
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops
    params_sum = get_model_parameters_number(self)
    return flops_sum / self.__batch_counter__, params_sum


def start_flops_count(self, **kwargs):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network.

    """
    add_batch_counter_hook_function(self)

    seen_types = set()

    def add_flops_counter_hook_function(module, ost, ignore_list):
        if type(module) in ignore_list:
            seen_types.add(type(module))
            if is_supported_instance(module):
                module.__params__ = 0
        elif is_supported_instance(module):
            if hasattr(module, '__flops_handle__'):
                return
            if type(module) in CUSTOM_MODULES_MAPPING:
                handle = module.register_forward_hook(
                    CUSTOM_MODULES_MAPPING[type(module)])
            else:
                handle = module.register_forward_hook(
                    MODULES_MAPPING[type(module)])
            module.__flops_handle__ = handle
            seen_types.add(type(module))
        elif False:  # FIXME too verbose
            if not type(module) in (nn.Sequential, nn.ModuleList) and \
               not type(module) in seen_types and \
               log.is_enabled('debug'):
                log.warn(
                    'Module ' + type(module).__name__ +
                    ' is treated as a zero-op.')
            seen_types.add(type(module))

    self.apply(partial(add_flops_counter_hook_function, **kwargs))


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.

    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)
    self.apply(remove_flops_counter_params)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.
    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


# ---- Internal functions
def empty_flops_counter_hook(module, input, output):
    module.__flops__ += 0


def upsample_flops_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__flops__ += int(output_elements_count)


def relu_flops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__flops__ += int(active_elements_count)


def linear_flops_counter_hook(module, input, output):
    input = input[0]
    # pytorch checks dimensions, so here we don't care much
    output_last_dim = output.shape[-1]
    bias_flops = output_last_dim if module.bias is not None else 0
    module.__flops__ += int(np.prod(input.shape) * output_last_dim + bias_flops)


def pool_flops_counter_hook(module, input, output):
    input = input[0]
    module.__flops__ += int(np.prod(input.shape))


def bn_flops_counter_hook(module, input, output):
    input = input[0]

    batch_flops = np.prod(input.shape)
    if module.affine:
        batch_flops *= 2
    module.__flops__ += int(batch_flops)


def conv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    # log.debug("counting flops of conv layer")
    input = input[0]
    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])
    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims)) * \
        in_channels * filters_per_channel
    active_elements_count = batch_size * int(np.prod(output_dims))
    overall_conv_flops = conv_per_position_flops * active_elements_count
    bias_flops = 0
    if conv_module.bias is not None:
        bias_flops = out_channels * active_elements_count
    overall_flops = overall_conv_flops + bias_flops
    conv_module.__flops__ += int(overall_flops)

def get_orig_flops(overall_flops):
    global orig_flops
    orig_flops.append(overall_flops)


def convdrop_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    # log.debug("counting flops of conv drop layer")
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    droplayer = conv_module.droplayer
    kernel_dims = list(droplayer.kernel_size)
    in_channels = droplayer.in_channels
    out_channels = droplayer.out_channels
    groups = droplayer.groups
    cur_prob, pre_prob = conv_module.__cur_prob__, conv_module.__pre_prob__

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims)) * \
        in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if droplayer.bias is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops
    get_orig_flops(overall_flops)
    overall_flops = overall_flops * cur_prob * pre_prob
    conv_module.__flops__ += int(overall_flops)

def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        log.warn(
            'No positional inputs found for a module,'
            ' assuming batch size is 1.')
    module.__batch_counter__ += batch_size


def rnn_flops(flops, rnn_module, w_ih, w_hh, input_size):
    # matrix matrix mult ih state and internal state
    flops += w_ih.shape[0]*w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    flops += w_hh.shape[0]*w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        flops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        flops += rnn_module.hidden_size
        # adding operations from both states
        flops += rnn_module.hidden_size*3
        # last two hadamard product and add
        flops += rnn_module.hidden_size*3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        flops += rnn_module.hidden_size*4
        # two hadamard product and add for C state
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # final hadamard
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return flops


def rnn_flops_counter_hook(rnn_module, input, output):
    """
    Takes into account batch goes at first position, contrary
    to pytorch common rule (but actually it doesn't matter).
    IF sigmoid and tanh are made hard, only a comparison FLOPS should be accurate
    """
    flops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__('weight_ih_l' + str(i))
        w_hh = rnn_module.__getattr__('weight_hh_l' + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        flops = rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__('bias_ih_l' + str(i))
            b_hh = rnn_module.__getattr__('bias_hh_l' + str(i))
            flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    flops *= seq_length
    if rnn_module.bidirectional:
        flops *= 2
    rnn_module.__flops__ += int(flops)


def rnn_cell_flops_counter_hook(rnn_cell_module, input, output):
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__('weight_ih')
    w_hh = rnn_cell_module.__getattr__('weight_hh')
    input_size = inp.shape[1]
    flops = rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__('bias_ih')
        b_hh = rnn_cell_module.__getattr__('bias_hh')
        flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    rnn_cell_module.__flops__ += int(flops)


def add_batch_counter_variables_or_reset(module):

    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return
    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops__') or hasattr(module, '__params__'):
            log.warn(
                'Variables __flops__ or __params__ are already '
                'defined for the module' + type(module).__name__ +
                ' ptflops can affect your code!')
        module.__flops__ = 0
        module.__params__ = get_model_parameters_number(module)


def get_convdrop_probs_hook(module, input, output):
    """ collect probs for convdrop module """
    global drop_prob
    if isinstance(module, LayerDropout):
        if hasattr(module, '__cur_prob__') or hasattr(module, '__pre_prob__'):
            log.warn(
                'Variables __cur_prob__ or __pre_prob__ are already '
                'defined for the module' + type(module).__name__ +
                ' ptflops can affect your code!')
        cur_prob = copy.deepcopy(module.probs).to("cpu").numpy().mean().tolist()
        module.__cur_prob__ = cur_prob
        module.__pre_prob__ = drop_prob
        drop_prob = cur_prob  # update global variable drop_prob

def register_get_convdrop_probs_hook(module):
    global prob_hooks
    if isinstance(module, LayerDropout):
        hook = module.register_forward_hook(get_convdrop_probs_hook)
        prob_hooks.append(hook)


def remove_convdrop_probs_hook(hooks):
    for h in hooks:
        h.remove()

def is_supported_instance(module):
    if type(module) in MODULES_MAPPING or type(module) in CUSTOM_MODULES_MAPPING:
        return True
    return False


def remove_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__

def remove_flops_counter_params(module):
    """ remove the added attributes from model to gurantee
        the training in the next round would not be affected
    """
    params = ['__flops__', '__params__', '__cur_prob__', '__pre_prob__']
    if is_supported_instance(module):
        for p in params:
            if hasattr(module, p):
                delattr(module, p)


CUSTOM_MODULES_MAPPING = {}
MODULES_MAPPING = {
    # convolutions
    nn.Conv1d: conv_flops_counter_hook,
    nn.Conv2d: conv_flops_counter_hook,
    nn.Conv3d: conv_flops_counter_hook,
    LayerDropout: convdrop_flops_counter_hook,
    # activations
    nn.ReLU: relu_flops_counter_hook,
    nn.PReLU: relu_flops_counter_hook,
    nn.ELU: relu_flops_counter_hook,
    nn.LeakyReLU: relu_flops_counter_hook,
    nn.ReLU6: relu_flops_counter_hook,
    # poolings
    nn.MaxPool1d: pool_flops_counter_hook,
    nn.AvgPool1d: pool_flops_counter_hook,
    nn.AvgPool2d: pool_flops_counter_hook,
    nn.MaxPool2d: pool_flops_counter_hook,
    nn.MaxPool3d: pool_flops_counter_hook,
    nn.AvgPool3d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool1d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool1d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool2d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool2d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool3d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool3d: pool_flops_counter_hook,
    # BNs
    nn.BatchNorm1d: bn_flops_counter_hook,
    nn.BatchNorm2d: bn_flops_counter_hook,
    nn.BatchNorm3d: bn_flops_counter_hook,
    # FC
    nn.Linear: linear_flops_counter_hook,
    # Upscale
    nn.Upsample: upsample_flops_counter_hook,
    # Deconvolution
    nn.ConvTranspose1d: conv_flops_counter_hook,
    nn.ConvTranspose2d: conv_flops_counter_hook,
    nn.ConvTranspose3d: conv_flops_counter_hook,
    # RNN
    nn.RNN: rnn_flops_counter_hook,
    nn.GRU: rnn_flops_counter_hook,
    nn.LSTM: rnn_flops_counter_hook,
    nn.RNNCell: rnn_cell_flops_counter_hook,
    nn.LSTMCell: rnn_cell_flops_counter_hook,
    nn.GRUCell: rnn_cell_flops_counter_hook
}
