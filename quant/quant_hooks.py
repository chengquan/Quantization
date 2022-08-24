__VERSION__ = "4.0"
__DATE__ = "20220101"

import torch
import torch.nn as nn

from .quant_utils import quantize


def _quantize_module_parameters(module):
    for param in module.parameters():
        if hasattr(param, 'quant_attr'):
            quant_attr = param.quant_attr
            if not quant_attr.quantified:
                continue
            if quant_attr.bit_width is None:
                continue
            bit_width = quant_attr.bit_width
            scale = None
            quantize(param, bit_width, scale)


def _quantize_input(quant_attrs, inputs):
    inputs = list(inputs)
    for i, input in enumerate(inputs):
        if isinstance(input, torch.Tensor):
            if quant_attrs[i] is not None:
                quant_attr = quant_attrs[i]
                if not quant_attr.quantified:
                    continue
                if quant_attr.bit_width is None or quant_attr.bit_width <= 0:
                    continue
                bit_width = quant_attr.bit_width
                scale = None
                quantize(input, bit_width, scale)
        elif isinstance(input, tuple):
            inputs[i] = _quantize_input(quant_attrs[i], inputs[i])
        else:
            pass
    return tuple(inputs)


def _recover_module_parameters(module):
    for parameter in module.parameters():
        if hasattr(parameter, 'quant_attr'):
            quant_attr = parameter.quant_attr
            if quant_attr.ori_data is not None:
                parameter.data = quant_attr.ori_data


def _quantize_pre_forward_hook(module: nn.Module, inputs):
    _quantize_module_parameters(module)
    input_quant_attrs = None
    if hasattr(module, 'quant_attr'):
        input_quant_attrs = module.quant_attr.input_attrs
    return _quantize_input(input_quant_attrs, inputs)


def _quantize_forward_hook(module: nn.Module, input, result):
    _recover_module_parameters(module)


def register_quantize_pre_forward_hook(module: nn.Module):
    for child in module.children():
        register_quantize_pre_forward_hook(child)
    module.register_forward_pre_hook(_quantize_pre_forward_hook)


def register_quantize_forward_hook(module: nn.Module):
    for child in module.children():
        register_quantize_forward_hook(child)
    module.register_forward_hook(_quantize_forward_hook)

