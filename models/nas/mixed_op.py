import abc
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstractMixedOP(nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(AbstractMixedOP, self).__init__()
        self.op_list = nn.ModuleList()
        self.build_op_list()
        self.flops = 0
        self.n_op = len(self.op_list)
        self.alpha = nn.Parameter(torch.empty((self.n_op,)), requires_grad=True)
        self.reset_parameters()

    @abstractmethod
    def build_op_list(self):
        pass

    def reset_parameters(self, a=0, b=1e-3):
        nn.init.uniform_(self.alpha, a, b)

    def forward(self, *args, **kwargs):
        x = None
        normalized_alpha = self._gumbel_softmax(self.alpha)
        for i, op in enumerate(self.op_list):
            if i == 0:
                x = op(*args, **kwargs)
            else:
                xx = op(*args, **kwargs)
                x = self._model_output_add(x, xx*normalized_alpha[i])
        return x

    def _model_output_add(self, a, b):
        if isinstance(a, tuple):
            x = ()
            for i in range(len(a)):
                x += (self._model_output_add(a[i], b[i]),)
            return x
        else:
            return a+b

    def _branch_cost(self):
        return torch.zeros_like(self.alpha)

    @staticmethod
    def _gumbel_softmax(x, hard=False, temperature=0.05, eps=1e-20):
        if not hard:
            return F.softmax(x, dim=0)
        else:
            uniform = torch.rand(x.size()[-1])
            sample = -torch.log(-torch.log(uniform + eps) + eps).to(x.device)
            y = x + sample
            return F.softmax(y/temperature, dim=0)


class MultiParamMixedOP(AbstractMixedOP):
    def __init__(self, op, param_scan_field, *args, **kwargs):
        # assert isinstance(op, nn.Module)
        assert isinstance(param_scan_field, dict)
        self._build_param = (op, param_scan_field, args, kwargs)
        super(MultiParamMixedOP, self).__init__()

    def build_op_list(self):
        op, param_scan_field, args, kwargs = self._build_param
        op_list = nn.ModuleList()
        param_scan_field_keys = list(param_scan_field.keys())
        if len(param_scan_field_keys) == 0:
            op_list.append(op(*args, **kwargs))
        elif len(param_scan_field_keys) == 1:
            for p in param_scan_field[param_scan_field_keys[0]]:
                new_kwargs = kwargs.copy()
                new_kwargs[param_scan_field_keys[0]] = p
                op_list.append(op(*args, **new_kwargs))
        else:
            new_param_scan_field = param_scan_field.copy()
            del new_param_scan_field[param_scan_field_keys[0]]
            for p in param_scan_field[param_scan_field_keys[0]]:
                new_kwargs = kwargs.copy()
                new_kwargs[param_scan_field_keys[0]] = p
                op_list.append(MultiParamMixedOP(op, new_param_scan_field, *args, **new_kwargs))
        self.op_list = op_list


class MultiArchMixedOP(AbstractMixedOP):
    def __init__(self, op_list):
        self._build_param = op_list
        super(MultiArchMixedOP, self).__init__()

    def build_op_list(self):
        self.op_list = nn.ModuleList(self._build_param)
