import enum
from typing import Tuple
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


class TuckerConv(nn.Module):
    def __init__(
        self,
        input_chanel_shape: Tuple[int, ...],
        output_chanel_shape: Tuple[int, ...],
        kernel_size: Tuple[int, int],
        ranks: Tuple[int, ...],
        space_rank: int,
        bias: bool = True,
        padding=0,
        device=None,
        dtype=None
    ) -> None:
        assert len(input_chanel_shape) == len(output_chanel_shape)
        assert len(input_chanel_shape) == len(ranks)

        factory_kwargs = {"device": device, "dtype": dtype}
        super(TuckerConv, self).__init__()

        self.input_chanel_shape = input_chanel_shape
        self.output_chanel_shape = output_chanel_shape
        self.kernel_size = kernel_size
        self.ranks = ranks
        self.space_rank = space_rank
        self.padding = padding

        if bias:
            self.bias = Parameter(torch.empty(np.prod(self.output_chanel_shape), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.space_factor = Parameter(torch.empty((self.space_rank, *self.kernel_size), **factory_kwargs))
        self.factors = nn.ParameterList([
            Parameter(torch.empty((self.input_chanel_shape[i], self.ranks[i], self.output_chanel_shape[i]), **factory_kwargs))
            for i in range(len(self.input_chanel_shape))
        ])
        self.core = Parameter(torch.empty(self.ranks + (self.space_rank,), **factory_kwargs))
        self.reset_parameters()


    def construct_network(self, input=None):
        if input is not None:
            tensor = tn.Node(
                input,
                name="input",
                axis_names=["batch"] + [f"in_{i}" for i, _ in enumerate(self.input_chanel_shape)] + ["space_rank", "H", "W"],
                backend="pytorch")
        else:
            space_factor = tn.Node(
                self.space_factor,
                name="space_factor",
                axis_names=["space_rank", "kH", "kW"],
                backend="pytorch",
            )
        factors = [
            tn.Node(
                self.factors[i],
                name=f"factor_{i}",
                axis_names=[f"in_{i}", f"rank_{i}", f"out_{i}"],
                backend="pytorch")
            for i, _ in enumerate(self.factors)
        ]
        core = tn.Node(
            self.core,
            name="core",
            axis_names=[f"rank_{i}" for i in range(len(self.input_chanel_shape))] + ["space_rank"],
            backend="pytorch",
        )
        out_edges = []
        for i in range(len(factors)):
            if input is not None:
                tensor[f"in_{i}"] ^ factors[i][f"in_{i}"]
            factors[i][f"rank_{i}"] ^ core[f"rank_{i}"]
            out_edges.append(factors[i][f"out_{i}"])
        if input is None:
            space_factor["space_rank"] ^ core["space_rank"]
        else:
            tensor["space_rank"] ^ core["space_rank"]
        if input is not None:
            return factors + [core, tensor], [tensor["batch"]] + out_edges + [tensor["H"], tensor["W"]]
        return factors + [core, space_factor]

    def calculate_std(self) -> float:
        network = self.construct_network()
        N = len(network)
        fan_in = np.prod(self.input_chanel_shape) * np.prod(self.kernel_size)
        fan_out = np.prod(self.output_chanel_shape)
        matrix_std = (2 / (fan_in + fan_out))**0.5
        rank_prod = np.prod([edge.dimension for edge in list(tn.get_all_nondangling(network))])
        return (matrix_std**2 / rank_prod) ** (0.5 / (N))

    def reset_parameters(self) -> None:
        std = self.calculate_std()
        for i, _ in enumerate(self.factors):
            _no_grad_normal_(self.factors[i], 0, std)
        _no_grad_normal_(self.core, 0, std)
        _no_grad_normal_(self.space_factor, 0, std)
        if self.bias is not None:
            fan_in = np.prod(self.input_chanel_shape) * np.prod(self.kernel_size)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: Tensor) -> Tensor:
        batch_size, channel, H, W = input.shape
        tensor = input
        # tensor = input.reshape(batch_size * channel, 1, H, W)
        tensor = F.conv2d(
            tensor,
            self.space_factor.reshape((self.space_factor.shape[0], 1) + self.space_factor.shape[1:]).repeat((channel, 1, 1, 1)),
            padding=self.padding,
            groups=channel,
        )
        _, _, H, W = tensor.shape
        tensor=tensor.reshape((batch_size,) + self.input_chanel_shape + (self.space_rank, H, W))
        # tensor = torch.tensordot(tensor, self.core, dims=([1 + len(self.input_chanel_shape)], [len(self.input_chanel_shape)]))
        # for i, _ in enumerate(self.input_chanel_shape):
        #     tensor = torch.tensordot(
        #         tensor,
        #         self.factors[i],
        #         dims=([1, 3 + len(self.input_chanel_shape) - i], [0, 1]))
        # tensor = tensor.permute([0] + list(range(3, len(tensor.shape))) + [1, 2])
        
        cores, dangling_edges = self.construct_network(
            input=tensor.reshape((batch_size,) + self.input_chanel_shape + (self.space_rank, H, W))
        )
        # tensor = cores[-1]
        # tensor = tn.contract_between(tensor, cores[-2])
        # for node in cores[:-2]:
        #     tensor = tn.contract_between(tensor, node)
        # tensor = tn.contractors.greedy(cores, output_edge_order=dangling_edges)
        tensor = tn.contractors.optimal(cores, output_edge_order=dangling_edges)
        # torch_tensor = tensor.tensor_from_edge_order(dangling_edges)\
        #     .reshape((batch_size, np.prod(self.output_chanel_shape), H, W))
        torch_tensor = tensor.tensor.reshape((batch_size, np.prod(self.output_chanel_shape), H, W))
        # torch_tensor = tensor.reshape((batch_size, np.prod(self.output_chanel_shape), H, W))
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
