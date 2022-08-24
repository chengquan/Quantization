from inspect import signature
from typing import Iterable

import torch
import torch.nn as nn
from .quant_utils import quantize, TensorQuantifyAttr, ModuleQuantifyAttr

"""
    while doing quant:
    all tensor including weights and input need to hasattr(tensor, 'quant.bit_width')
    and while doing forward each time, the tensor.data update quantize each time according to 'bit_width'
"""

GATHER = True
gathered = {}

def _quantize_module_parameters(module: nn.Module):
    for param_key in module._parameters.keys():
        param = module._parameters[param_key]
        if hasattr(param, 'quant_attr'):
            quant_attr = param.quant_attr
            if not quant_attr.quantified:
                continue
            if quant_attr.bit_width is None:
                continue
            bit_width = quant_attr.bit_width
            scale = quant_attr.scale
            module._parameters[param_key] = quantize(param, bit_width, scale)
    # TODO: removing recursive here, but recursive register pre_forward hook in
    #  _register_quantize_pre_forward_hook instead
    # for module_key in module._modules.keys():
    #     _quantize_module_parameters(module._modules[module_key])
    # raise NotImplementedError()


def _recover_module_parameters(module: nn.Module):
    for param_key in module._parameters.keys():
        param = module._parameters[param_key]
        if hasattr(param, 'quant_attr'):
            quant_attr = param.quant_attr
            if quant_attr.ori_data is not None:
                param.data = quant_attr.ori_data


def _quantize_input(quant_input_attrs, input):
    # if not hasattr(module, 'quant_attr'):
    #     return input
    # module_quant_attr = module.quant_attr
    # if module_quant_attr.output_attrs is not None:
    #     raise UserWarning("usage of output attrs is not allowed now, and will have no effect!")
    input_list = list(input)
    for idx in range(len(input_list)):
        e = input_list[idx]
        if isinstance(e, torch.Tensor):
            # if hasattr(e, 'quant_attr'):
            #     quant_attr = e.quant_attr
            if quant_input_attrs[idx] is not None:
                quant_attr = quant_input_attrs[idx]
                if not quant_attr.quantified:
                    continue
                if quant_attr.bit_width is None or quant_attr.bit_width <= 0:
                    continue
                bit_width = quant_attr.bit_width
                scale = quant_attr.scale
                #scale = None
                input_list[idx] = quantize(e, bit_width, scale)
        elif isinstance(e, tuple):
            input_list[idx] = _quantize_input(quant_input_attrs[idx], e)
        else:
            input_list[idx] = e
    return tuple(input_list)


def _quantize_pre_forward_hook(module: nn.Module, input):
    # print('quant_pre_forward_hook', module)
    # quantify module parameters
    global gathered
    gathered_local = list()
    # revised by qcheng
    _quantize_module_parameters(module)
    if GATHER:
        for k, v in module.named_parameters():
            if hasattr(v, 'quant_attr'):
                quant_attr = getattr(v, 'quant_attr', None)
                if quant_attr.quantified:
                    gathered_local.append((k, quant_attr.bit_width, quant_attr.scale, quant_attr.int_data, quant_attr.ori_data))
        #print('module is', quant_attr.bit_width, quant_attr.scale)
    # quantify input
    input_quant_attrs = None
    if hasattr(module, 'quant_attr'):
        input_quant_attrs = module.quant_attr.input_attrs
    input = _quantize_input(input_quant_attrs, input)
    if GATHER:
        for i, v in enumerate(input):
            if hasattr(v, 'quant_attr'):
                quant_attr = getattr(v, 'quant_attr', None)
                if quant_attr.quantified:
                    gathered_local.append(('input_%d' % i, quant_attr.bit_width, quant_attr.scale.clone(), quant_attr.int_data, quant_attr.ori_data))
        gathered[module.named_modules().__next__()[1]] = gathered_local
        #print('input is',  quant_attr.bit_width, quant_attr.scale.clone())
    if len(gathered) > 100:
        gathered = gathered[-50:]
    return input


def _get_gather():
    global gathered
    return gathered


def _quantize_post_forward_hook(module, input, result):
    global gathered
    gathered_local = gathered[module.named_modules().__next__()[1]]
    _recover_module_parameters(module)
    if hasattr(module, 'quant_attr'):
        output_quant_attrs = module.quant_attr.output_attrs
        _quantize_input(output_quant_attrs, (result,))
    if GATHER:
        for i, v in enumerate((result,)):
            if hasattr(v, 'quant_attr'):
                quant_attr = getattr(v, 'quant_attr', None)
                if quant_attr.quantified:
                    gathered_local.append(
                        ('output_%d' % i, quant_attr.bit_width, quant_attr.scale.clone(), quant_attr.int_data, quant_attr.ori_data))
        gathered[module.named_modules().__next__()[1]] = gathered_local
    if len(gathered) > 100:
        gathered = gathered[-50:]


def _set_params_quantize(params, bit_width=8, scale=None):
    if bit_width <= 0:
        return
    if isinstance(params, Iterable) and (not isinstance(params, torch.Tensor)):
        for p in params:
            quant_attr = TensorQuantifyAttr(quantified=True,
                                            bit_width=bit_width,
                                            scale=scale)
            setattr(p, 'quant_attr', quant_attr)
    elif isinstance(params, torch.Tensor):
        quant_attr = TensorQuantifyAttr(quantified=True,
                                        bit_width=bit_width,
                                        scale=scale)
        setattr(params, 'quant_attr', quant_attr)


def _set_module_quantize(module, bit_width=8, scale=None, scale_out=None):
    for child in module._modules.values():
        _set_module_quantize(child, bit_width, scale, scale_out)
    sig = signature(module.forward)
    inputs = sig.parameters
    in_quants = tuple()
    for i in range(len(inputs)):
        in_quants = in_quants + (TensorQuantifyAttr(quantified=True, bit_width=bit_width, scale=scale),)
    out_quants = (TensorQuantifyAttr(quantified=True, bit_width=bit_width, scale=scale_out),)  # scale
    quant_attr = ModuleQuantifyAttr(input_attrs=in_quants, output_attrs=out_quants)
    setattr(module, 'quant_attr', quant_attr)


def _register_quantize_pre_forward_hook(module):
    for child in module._modules.values():
        _register_quantize_pre_forward_hook(child)
    module.register_forward_pre_hook(_quantize_pre_forward_hook)


def _register_quantize_post_forward_hook(module):
    module.register_forward_hook(_quantize_post_forward_hook)


def _try_get_del(d, k, default_v=None):
    if k in d:
        ret = d[k]
        del d[k]
        return ret
    else:
        return default_v


class QuantifiedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        input_bit_width = _try_get_del(kwargs, 'input_bit_width')
        input_scale = _try_get_del(kwargs, 'input_scale')
        output_scale = _try_get_del(kwargs, 'output_scale')
        weight_bit_width = _try_get_del(kwargs, 'weight_bit_width')
        weight_scale = _try_get_del(kwargs, 'weight_scale')
        bias_bit_width = _try_get_del(kwargs, 'bias_bit_width')
        bias_scale = _try_get_del(kwargs, 'bias_bit_scale')

        super(QuantifiedLinear, self).__init__(*args, **kwargs)

        _set_params_quantize(self.weight, weight_bit_width, weight_scale)
        if getattr(self, 'bias', None) is not None:
            _set_params_quantize(self.bias, bias_bit_width, bias_scale)
        _set_module_quantize(self, input_bit_width, input_scale, output_scale)

        _register_quantize_pre_forward_hook(self)
        _register_quantize_post_forward_hook(self)


class QuantifiedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        input_bit_width = _try_get_del(kwargs, 'input_bit_width')
        input_scale = _try_get_del(kwargs, 'input_scale')
        output_scale = _try_get_del(kwargs, 'output_scale')
        weight_bit_width = _try_get_del(kwargs, 'weight_bit_width')
        weight_scale = _try_get_del(kwargs, 'weight_scale')
        bias_bit_width = _try_get_del(kwargs, 'bias_bit_width')
        bias_scale = _try_get_del(kwargs, 'bias_bit_scale')

        super(QuantifiedConv2d, self).__init__(*args, **kwargs)

        _set_params_quantize(self.weight, weight_bit_width, weight_scale)
        if getattr(self, 'bias', None) is not None:
            _set_params_quantize(self.bias, bias_bit_width, bias_scale)
        _set_module_quantize(self, input_bit_width, input_scale, output_scale)

        _register_quantize_pre_forward_hook(self)
        _register_quantize_post_forward_hook(self)


class QuantifiedConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        input_bit_width = _try_get_del(kwargs, 'input_bit_width')
        input_scale = _try_get_del(kwargs, 'input_scale')
        output_scale = _try_get_del(kwargs, 'output_scale')
        weight_bit_width = _try_get_del(kwargs, 'weight_bit_width')
        weight_scale = _try_get_del(kwargs, 'weight_scale')
        bias_bit_width = _try_get_del(kwargs, 'bias_bit_width')
        bias_scale = _try_get_del(kwargs, 'bias_bit_scale')

        super(QuantifiedConv3d, self).__init__(*args, **kwargs)

        _set_params_quantize(self.weight, weight_bit_width, weight_scale)
        if getattr(self, 'bias', None) is not None:
            _set_params_quantize(self.bias, bias_bit_width, bias_scale)
        _set_module_quantize(self, input_bit_width, input_scale, output_scale)

        _register_quantize_pre_forward_hook(self)
        _register_quantize_post_forward_hook(self)


if __name__ == '__main__':
    # import torch
    # import torch.nn as nn
    # import numpy as np
    # from torchvision.models import resnet18
    # from model.TTLayer import TTLayer
    # hook = 0
    # rm = resnet18()
    # tm = TTLayer((1, 2, 3, 4), (5, 6, 7, 8), (1, 4, 4, 4, 1))
    # hook = 1
    # _set_params_quantize(rm.parameters())
    # _set_module_quantize(rm)
    # hook = 2
    # _set_params_quantize(tm.parameters())
    # _set_module_quantize(tm)
    # hook = 3
    # rx = torch.from_numpy(np.random.randn(2, 3, 120, 160)).float()
    # tx = torch.from_numpy(np.random.randn(2, 1, 2, 3, 4)).float()
    # hook = 4
    # _register_quantize_pre_forward_hook(tm)
    # _register_quantize_pre_forward_hook(rm)
    # hook = 4.5
    # ry = rm(rx)
    # hook = 5
    # ty = tm(tx)
    # hook = 6
    pass
