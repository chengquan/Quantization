from .mixed_op import MultiArchMixedOP
from quant.quant_layers import QuantifiedLinear, QuantifiedConv2d


def _try_get_del(d, k, default_v=None):
    if k in d:
        ret = d[k]
        del d[k]
        return ret
    else:
        return default_v


class SymMixedPrecisionConv2d(MultiArchMixedOP):
    def __init__(self, *args, **kwargs):
        bit_width_scan_field = _try_get_del(
            kwargs,
            'bit_width_scan_field',
            [(2, 2, None), (4, 4, None), (8, 8, None)]
        )
        op_list = list()
        for input_bitwidth, weight_bitwidth, bias_bitwidth in bit_width_scan_field:
            kwargs['input_bit_width'] = input_bitwidth
            kwargs['weight_bit_width'] = weight_bitwidth
            kwargs['bias_bit_width'] = bias_bitwidth
            op_list.append(QuantifiedConv2d(*args, **kwargs))
        super(SymMixedPrecisionConv2d, self).__init__(op_list)

    def branch_cost(self, no_quantize_cost=64):
        normalized_alpha = self._gumbel_softmax(self.alpha)
        cost = None
        for i, op in enumerate(self.op_list):
            cc = [no_quantize_cost, no_quantize_cost, no_quantize_cost]
            if hasattr(op, 'quant_attr'):
                if hasattr(op.quant_attr, 'input_attrs'):
                    if len(op.quant_attr.input_attrs) == 1:
                        if op.quant_attr.input_attrs[0].quantified and op.quant_attr.input_attrs[0].bit_width > 0:
                            cc[0] = op.quant_attr.input_attrs[0].bit_width / 8
            if hasattr(op.weight, 'quant_attr'):
                if op.weight.quant_attr.quantified and op.weight.quant_attr.bit_width > 0:
                    cc[1] = op.weight.quant_attr.bit_width
            if hasattr(op, 'bias'):
                if op.bias is not None:
                    if hasattr(op.bias, 'quant_attr'):
                        if op.bias.quant_attr.quantified and op.bias.quant_attr.bit_width > 0:
                            cc[2] = op.bias.quant_attr.bit_width
            cc = cc[0]
            if i == 0:
                cost = cc*normalized_alpha[i]
            else:
                cost += cc*normalized_alpha[i]
        return cost

    def forward(self, *args, **kwargs):
        x = None
        normalized_alpha = self._gumbel_softmax(self.alpha)
        for i, op in enumerate(self.op_list):
            if i == 0:
                x = op(*args, **kwargs)
                self.flops = x.shape[1] * args[0].shape[1] * x.shape[2] * x.shape[3] * op.kernel_size[0]
            else:
                xx = op(*args, **kwargs)
                x = self._model_output_add(x, xx*normalized_alpha[i])
        return x


class SymMixedPrecisionLinear(MultiArchMixedOP):
    def __init__(self, *args, **kwargs):
        bit_width_scan_field = _try_get_del(
            kwargs,
            'bit_width_scan_field',
            [(2, 2, None), (4, 4, None), (8, 8, None)]
        )
        op_list = list()
        for input_bitwidth, weight_bitwidth, bias_bitwidth in bit_width_scan_field:
            kwargs['input_bit_width'] = input_bitwidth
            kwargs['weight_bit_width'] = weight_bitwidth
            kwargs['bias_bit_width'] = bias_bitwidth
            op_list.append(QuantifiedLinear(*args, **kwargs))
        super(SymMixedPrecisionLinear, self).__init__(op_list)

    def branch_cost(self, no_quantize_cost=64):
        normalized_alpha = self._gumbel_softmax(self.alpha)
        cost = None
        for i, op in enumerate(self.op_list):
            cc = [no_quantize_cost, no_quantize_cost, no_quantize_cost]
            if hasattr(op, 'quant_attr'):
                if hasattr(op.quant_attr, 'input_attrs'):
                    if len(op.quant_attr.input_attrs) == 1:
                        if op.quant_attr.input_attrs[0].quantified and op.quant_attr.input_attrs[0].bit_width > 0:
                            cc[0] = op.quant_attr.input_attrs[0].bit_width / 8
            if hasattr(op.weight, 'quant_attr'):
                if op.weight.quant_attr.quantified and op.weight.quant_attr.bit_width > 0:
                    cc[1] = op.weight.quant_attr.bit_width
            if hasattr(op, 'bias'):
                if op.bias is not None:
                    if hasattr(op.bias, 'quant_attr'):
                        if op.bias.quant_attr.quantified and op.bias.quant_attr.bit_width > 0:
                            cc[2] = op.bias.quant_attr.bit_width
            cc = cc[0]
            if i == 0:
                cost = cc*normalized_alpha[i]
            else:
                cost += cc*normalized_alpha[i]
        return cost

    def forward(self, *args, **kwargs):
        x = None
        normalized_alpha = self._gumbel_softmax(self.alpha)
        for i, op in enumerate(self.op_list):
            if i == 0:
                x = op(*args, **kwargs)
                self.flops = x.shape[1] * args[0].shape[1]
            else:
                xx = op(*args, **kwargs)
                x = self._model_output_add(x, xx*normalized_alpha[i])
        return x


