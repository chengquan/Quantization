import torch
import numpy as np

GATHER = True

#35ge34  35ge35
# def quantize(tensor: torch.Tensor, bit_width=8, scale=None):
#     tensor_copy = tensor.clone().detach()
#     mean = tensor_copy.mean()
#     tensor_copy = tensor_copy - mean
#     if scale is None:
#         scale = tensor_copy.view((-1,)).abs().sort()[0][-1]
#     step = scale / (2**(bit_width-1))
#     quantified_integer = torch.round(tensor_copy.data/step)
#     tensor.data = quantified_integer*step+mean
#     return tensor, (quantified_integer, scale, mean)

# TD:   this module set the input/output quantization of the module
#       first while register the pre forward hook, the register checks the params signature of input/output
#       after checking sig, then generate the quantify attr for each in/out
#       then register the pre-hook and post-hook
#       in the pre-hook and post-hook, the hook quantify the input and output,
#       also set the quantify attr for debug usage
class ModuleQuantifyAttr:
    def __init__(self, input_attrs=None, output_attrs=None):
        self.input_attrs = input_attrs
        self.output_attrs = output_attrs


class TensorQuantifyAttr:
    def __init__(self, quantified=None, bit_width=None, scale=None, ori_data=None, int_data=None, mean=None):
        super(TensorQuantifyAttr, self).__init__()
        self.quantified = quantified
        self.bit_width = bit_width
        self.scale = scale
        self.ori_data = ori_data
        self.int_data = int_data
        self.mean = mean
        self.child = tuple()


def quantize(tensor: torch.Tensor, bit_width=8, scale=None):
    ori_data = tensor.data.clone()
    qut_data = tensor.data.clone()
    # mean = ori_data.mean()
    # ori_data = ori_data - mean
    # if str(mean.cpu().numpy()) == 'nan':
    #     hook = 0
    if scale is None:
        scale = ori_data.view((-1,)).abs_()
        scale = scale.sort()[0][int(0.99*scale.shape[-1])]
        # scale = torch.log2(scale)-(bit_width-1)
        scale = torch.ceil_(torch.log2_(scale))-(bit_width-1)
        # scale = torch.floor(torch.log2(scale))-(bit_width-1)
    else:
        scale = np.array(scale, dtype=float)
        scale = torch.from_numpy(scale)
        scale = scale.cuda()
        # print("scale is", scale)
    data_range = torch.pow(2, scale+(bit_width-1))
    data_step = torch.pow(2, scale)
    # data_range = 2**(scale+(bit_width-1))
    # data_step = 2**scale
    qut_data.clamp_(-data_range, data_range-data_step)
    # step = scale / (2**(bit_width-1))
    # quantizied_data = torch.floor_(ori_data/data_step)
    quantizied_data = torch.round_(qut_data/data_step + 1e-6)

    if GATHER:
        quant = TensorQuantifyAttr(quantified=True,
                                   bit_width=bit_width,
                                   scale=scale.cpu(),
                                   ori_data=tensor.data.clone(),
                                   int_data=quantizied_data.detach().cpu(),
                                   mean=None)
    else:
        quant = TensorQuantifyAttr(quantified=True,
                                   bit_width=bit_width,
                                   scale=scale.cpu(),
                                   ori_data=tensor.data.clone(),
                                   int_data=None,
                                   mean=None)
    tensor.data = quantizied_data*data_step
    if hasattr(tensor, 'quant_attr'):
        del tensor.quant_attr.ori_data
        delattr(tensor, 'quant_attr')
    setattr(tensor, 'quant_attr', quant)
    del ori_data
    del qut_data
    del quantizied_data
    del scale
    return tensor
