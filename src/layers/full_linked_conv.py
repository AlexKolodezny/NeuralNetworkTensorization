import enum
from typing import Tuple, List
import torch
from torch import Tensor
from torch import nn
from typing import Tuple
from torch.nn import init
import math

from torch.nn import functional as F

import tensornetwork as tn

import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.init import _no_grad_normal_
import gc


class FullLinkedConv(nn.Module):
    def __init__(
        self,
        input_chanel_shape: Tuple[int, ...],
        output_chanel_shape: Tuple[int, ...],
        kernel_size: Tuple[int, int],
        ranks: List[List[int]],
        bias: bool = True,
        padding=0,
        stride=1,
        device=None,
        dtype=None
    ) -> None:
        assert len(input_chanel_shape) == len(output_chanel_shape)
        assert len(input_chanel_shape) == len(ranks) - 1
        assert np.all(np.array(ranks) == np.array(ranks).T)

        factory_kwargs = {"device": device, "dtype": dtype}
        super(FullLinkedConv, self).__init__()

        self.input_chanel_shape = input_chanel_shape
        self.output_chanel_shape = output_chanel_shape
        self.kernel_size = kernel_size
        self.space_ranks = ranks[0][1:]
        self.channel_ranks = [ranks[i][1:] for i in range(1, len(ranks))]
        for i, dim in enumerate(self.space_ranks):
            self.channel_ranks[i][i] = dim
        self.padding = padding
        self.stride = stride

        if bias:
            self.bias = Parameter(torch.empty(np.prod(self.output_chanel_shape), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.space_factor = Parameter(torch.empty(tuple(self.space_ranks) + self.kernel_size, **factory_kwargs))
        self.factors = nn.ParameterList([
            Parameter(torch.empty(tuple(r) + (self.input_chanel_shape[i], self.output_chanel_shape[i]), **factory_kwargs))
            for i, r in enumerate(self.channel_ranks)
        ])
        self.reset_parameters()


    def construct_network(self, input=None):
        if input is not None:
            tensor = tn.Node(
                input,
                name="input",
                axis_names=["batch"] + [f"in_{i}" for i, _ in enumerate(self.input_chanel_shape)] + \
                    [f"space_rank_{i}" for i, _ in enumerate(self.space_ranks)] + ["H", "W"],
                backend="pytorch")
        else:
            space_factor = tn.Node(
                self.space_factor,
                name="space_factor",
                axis_names=[f"space_rank_{i}" for i, _ in enumerate(self.space_ranks)] + ["kH", "kW"],
                backend="pytorch",
            )
        factors = [
            tn.Node(
                self.factors[i],
                name=f"factor_{i}",
                axis_names=[f"rank_{i},{j}" for j, _ in enumerate(self.factors)] + [f"in_{i}", f"out_{i}"],
                backend="pytorch")
            for i, _ in enumerate(self.factors)
        ]
        out_edges = []
        for i in range(len(factors)):
            if input is not None:
                tensor[f"in_{i}"] ^ factors[i][f"in_{i}"]
                tensor[f"space_rank_{i}"] ^ factors[i][f"rank_{i},{i}"]
            else:
                space_factor[f"space_rank_{i}"] ^ factors[i][f"rank_{i},{i}"]
            out_edges.append(factors[i][f"out_{i}"])
            for j in range(i):
                factors[i][f"rank_{i},{j}"] ^ factors[j][f"rank_{j},{i}"]
        if input is not None:
            return factors + [tensor], [tensor["batch"]] + out_edges + [tensor["H"], tensor["W"]]
        return factors +[space_factor]

    def calculate_std(self) -> float:
        network = self.construct_network()
        N = len(network)
        fan_in = np.prod(self.input_chanel_shape) * np.prod(self.kernel_size)
        fan_out = np.prod(self.output_chanel_shape) * np.prod(self.kernel_size)
        matrix_std = (2 / (fan_in + fan_out))**0.5
        rank_prod = np.prod([edge.dimension for edge in list(tn.get_all_nondangling(network))])
        return (matrix_std**2 / rank_prod) ** (0.5 / (N))

    def reset_parameters(self) -> None:
        std = self.calculate_std()
        # for i, _ in enumerate(self.factors):
        #     _no_grad_normal_(
        #         self.factors[i],
        #         0,
        #         (2 / (np.prod(self.factors[i].shape[:2]) + np.prod(self.factors[i].shape[2:])))**0.5)
        # init.xavier_normal_(self.space_factor)
        # _no_grad_normal_(
        #     self.last_factor,
        #     0,
        #     (2 / (np.prod(self.factors[i].shape[:2]) + np.prod(self.factors[i].shape[2])))**0.5)
        for i, _ in enumerate(self.factors):
            _no_grad_normal_(self.factors[i], 0, std)
        _no_grad_normal_(self.space_factor, 0, std)
        if self.bias is not None:
            fan_in = np.prod(self.input_chanel_shape) * np.prod(self.kernel_size)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: Tensor) -> Tensor:
        batch_size, channel, H, W = input.shape
        tensor = input
        tensor = F.conv2d(
            tensor,
            self.space_factor.reshape((np.prod(self.space_ranks), 1) + self.kernel_size).repeat((channel, 1, 1, 1)),
            padding=self.padding,
            stride=self.stride,
            groups=channel,
        )
        _, _, H, W = tensor.shape
        tensor=tensor.reshape((batch_size,) + self.input_chanel_shape + tuple(self.space_ranks) + (H, W))
        
        cores, dangling_edges = self.construct_network(
            input=tensor,
        )
        tensor = tn.contractors.optimal(cores, output_edge_order=dangling_edges)
        torch_tensor = tensor.tensor.reshape((batch_size, np.prod(self.output_chanel_shape), H, W))
        del tensor
        del cores
        del dangling_edges
        gc.collect(generation=0)
        if self.bias is None:
            return torch_tensor
        return torch.add(torch_tensor, self.bias[None, :, None, None])

    def extra_repr(self) -> str:
        return 'input_channel_shape={}, output_channel_shape={}, ranks={}, space_rank={}, bias={}'.format(
            self.input_chanel_shape, self.output_chanel_shape, self.ranks, self.space_rank, self.bias is not None
        )
