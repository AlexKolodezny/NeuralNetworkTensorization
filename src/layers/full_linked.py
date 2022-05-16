from typing import Tuple, Optional, List
import torch
from torch import Tensor
from torch import nn
from typing import Tuple
from torch.nn import init
import math
import numpy as np
from copy import deepcopy

import tensornetwork as tn

from torch.nn.parameter import Parameter
from torch.nn.init import _no_grad_normal_


class FullLinkedLayer(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        ranks: List[List[int]],
        edge_mask: List[List[int]] = None,
        bias: bool = True,
        gain: float = 1,
        device=None,
        dtype=None
    ) -> None:
        assert np.all(np.array(ranks) == np.array(ranks).T)
        assert np.all(np.array(edge_mask) == np.array(edge_mask).T)
        assert len(ranks) == len(input_shape) + len(output_shape)

        if edge_mask == None:
            edge_mask = [[1] * len(ranks) for i in range(len(ranks))]

        factory_kwargs = {"device": device, "dtype": dtype}
        super(FullLinkedLayer, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        ranks_copy = deepcopy(ranks)
        self.shape = self.input_shape + self.output_shape
        self.gain = gain
        self.edge_mask = deepcopy(edge_mask)
        for i, dim in enumerate(self.shape):
            ranks_copy[i][i] = dim
            self.edge_mask[i][i] = 1
        if bias:
            self.bias = Parameter(torch.empty(self.output_shape, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.factors = nn.ParameterList([
            Parameter(torch.empty(list(np.array(ranks_copy[i])[np.array(self.edge_mask[i])==1]),
                **factory_kwargs))
            for i in range(len(self.shape))
        ])
        self.masks = {}
        self.reset_parameters()
    
    def ranks(self):
        ranks = [[1] * len(self.shape) for i in range(len(self.shape))]
        for i, _ in enumerate(self.factors):
            for j, dim in zip(np.arange(len(self.shape))[np.array(self.edge_mask[i])==1], self.factors[i].shape):
                ranks[i][j] = dim
        for i, dim in enumerate(self.shape):
            ranks[i][i] = dim
        return ranks
    
    def clear_masks(self):
        self.masks = {}

    def masked_tensor(self, tensor, name):
        mask = self.masks.get(name, None)
        if mask is None:
            return tensor
        res = tensor * mask.detach()
        res += tensor.detach() * mask.detach().logical_not()
        return res
    
    def construct_network(self, input=None):
        if input is not None:
            tensor = tn.Node(
                self.masked_tensor(input, "input"),
                name="input",
                axis_names=["batch"] + [f"in_{i}" for i, _ in enumerate(self.input_shape)],
                backend="pytorch")
        factors = [
            tn.Node(
                self.masked_tensor(self.factors[i], f"factor_{i}"),
                name=f"factor_{i}",
                axis_names=[f"rank_{i}{j}" for j in np.arange(len(self.shape))[np.array(self.edge_mask[i])==1]],
                backend="pytorch")
            for i, _ in enumerate(self.factors)
        ]
        result_cores = factors
        for i, _ in enumerate(factors):
            for j in np.arange(len(self.shape))[np.array(self.edge_mask[i])==1]:
                if j < i:
                    factors[i][f"rank_{i}{j}"] ^ factors[j][f"rank_{j}{i}"]
        if input is not None:
            for i, _ in enumerate(self.input_shape):
                tensor[f"in_{i}"] ^ factors[i][f"rank_{i}{i}"]
            output_edges = [factors[i][f"rank_{i}{i}"] for i in range(len(self.input_shape), len(self.shape))]
            return result_cores + [tensor], [tensor["batch"]] + output_edges
        return result_cores


    def calculate_std(self) -> float:
        network = self.construct_network()
        N = len(network)
        fan_in = np.prod(self.input_shape)
        fan_out = np.prod(self.output_shape)
        matrix_std = self.gain * (2 / (fan_in + fan_out))**0.5
        rank_prod = np.prod([edge.dimension for edge in list(tn.get_all_nondangling(network))])
        return (matrix_std**2 / rank_prod) ** (0.5 / (N))

    def reset_parameters(self) -> None:
        std = self.calculate_std()
        for core in self.construct_network():
            _no_grad_normal_(core.tensor, 0, std)
        if self.bias is not None:
            fan_in = np.prod(self.input_shape)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: Tensor) -> Tensor:
        batch_size = input.shape[0]
        cores, dangling_edges = self.construct_network(input=input)
        tensor = tn.contractors.greedy(cores, output_edge_order=dangling_edges)
        torch_tensor = tensor.tensor.reshape((batch_size,) + self.output_shape)
        if self.bias is None:
            return torch_tensor
        return torch.add(torch_tensor, self.bias)

    def extra_repr(self) -> str:
        return 'input_shape={}, output_shape={}, ranks={}, bias={}'.format(
            self.input_shape, self.output_shape, self.ranks, self.bias is not None
        )
