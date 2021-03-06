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


class TTConv(nn.Module):
    def __init__(
        self,
        input_chanel_shape: Tuple[int, ...],
        output_chanel_shape: Tuple[int, ...],
        kernel_size: Tuple[int, int],
        ranks: Tuple[int, ...],
        space_rank: int,
        bias: bool = True,
        padding=0,
        stride=1,
        device=None,
        dtype=None
    ) -> None:
        assert len(input_chanel_shape) == len(output_chanel_shape)
        assert len(input_chanel_shape) == len(ranks) + 1

        factory_kwargs = {"device": device, "dtype": dtype}
        super(TTConv, self).__init__()

        self.input_chanel_shape = input_chanel_shape
        self.output_chanel_shape = output_chanel_shape
        self.kernel_size = kernel_size
        self.ranks = (space_rank,) + ranks
        self.space_rank = space_rank
        self.padding = padding
        self.stride = stride

        if bias:
            self.bias = Parameter(torch.empty(np.prod(self.output_chanel_shape), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.space_factor = Parameter(torch.empty((self.ranks[0], 1, *self.kernel_size), **factory_kwargs))
        self.factors = nn.ParameterList([
            Parameter(torch.empty((self.input_chanel_shape[i], self.ranks[i], self.ranks[i + 1], self.output_chanel_shape[i]), **factory_kwargs))
            for i in range(len(self.input_chanel_shape) - 1)
        ])
        self.last_factor = Parameter(torch.empty((self.input_chanel_shape[-1], self.ranks[-1], self.output_chanel_shape[-1]), **factory_kwargs))
        self.masks = {}
        self.reset_parameters()

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
                axis_names=["batch"] + [f"in_{i}" for i, _ in enumerate(self.input_chanel_shape)] + ["rank_0", "H", "W"],
                backend="pytorch")
        else:
            space_factor = tn.Node(
                self.masked_tensor(self.space_factor, "space_factor"),
                name="space_factor",
                axis_names=["rank_0", "space_in", "kH", "kW"],
                backend="pytorch",
            )
        factors = [
            tn.Node(
                self.masked_tensor(self.factors[i], f"factor_{i}"),
                name=f"factor_{i}",
                axis_names=[f"in_{i}", f"rank_{i}", f"rank_{i + 1}", f"out_{i}"],
                backend="pytorch")
            for i, _ in enumerate(self.factors)
        ]
        last = len(self.input_chanel_shape) - 1
        factors = factors + [tn.Node(
                self.masked_tensor(self.last_factor, f"factor_{last}"),
                name=f"factor_{last}",
                axis_names=[f"in_{last}", f"rank_{last}", f"out_{last}"],
                backend="pytorch")]
        out_edges = []
        for i in range(len(factors)):
            if input is not None:
                tensor[f"in_{i}"] ^ factors[i][f"in_{i}"]
            out_edges.append(factors[i][f"out_{i}"])
        for i in range(1, len(factors)):
            factors[i - 1][f"rank_{i}"] ^ factors[i][f"rank_{i}"]
        if input is None:
            space_factor["rank_0"] ^ factors[0]["rank_0"]
        else:
            tensor["rank_0"] ^ factors[0]["rank_0"]
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
        # std = self.calculate_std()
        for i, _ in enumerate(self.factors):
            _no_grad_normal_(
                self.factors[i],
                0,
                (2 / (np.prod(self.factors[i].shape[:2]) + np.prod(self.factors[i].shape[2:])))**0.5)
        init.xavier_normal_(self.space_factor)
        _no_grad_normal_(
            self.last_factor,
            0,
            (2 / (np.prod(self.factors[i].shape[:2]) + np.prod(self.factors[i].shape[2])))**0.5)
        # for i, _ in enumerate(self.factors):
        #     _no_grad_normal_(self.factors[i], 0, std)
        # _no_grad_normal_(self.space_factor, 0, std)
        # _no_grad_normal_(self.last_factor, 0, std)
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
            self.masked_tensor(self.space_factor, "space_factor").repeat((channel, 1, 1, 1)),
            padding=self.padding,
            stride=self.stride,
            groups=channel,
        )
        # tensor = F.conv2d(
        #     tensor,
        #     self.space_factor,
        #     padding=self.padding,
        #     stride=self.stride,
        # )
        _, _, H, W = tensor.shape
        tensor=tensor.reshape((batch_size,) + self.input_chanel_shape + (self.space_factor.shape[0], H, W))
        # tensor = torch.tensordot(tensor, self.core, dims=([1 + len(self.input_chanel_shape)], [len(self.input_chanel_shape)]))
        # for i, _ in enumerate(self.input_chanel_shape):
        #     tensor = torch.tensordot(
        #         tensor,
        #         self.factors[i],
        #         dims=([1, 3 + len(self.input_chanel_shape) - i], [0, 1]))
        # tensor = tensor.permute([0] + list(range(3, len(tensor.shape))) + [1, 2])
        
        cores, dangling_edges = self.construct_network(
            input=tensor.reshape((batch_size,) + self.input_chanel_shape + (self.space_factor.shape[0], H, W))
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
