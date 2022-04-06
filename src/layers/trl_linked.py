from typing import Tuple, Optional
import torch
from torch import Tensor
from torch import nn
from typing import Tuple
from torch.nn import init
import math
import numpy as np

import tensornetwork as tn

from torch.nn.parameter import Parameter
from torch.nn.init import _no_grad_normal_


class TRLMasked(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        input_ranks: Tuple[int, ...],
        link_ranks: Tuple[int, ...],
        output_features: int,
        output_rank: Optional[int]=None,
        bias: bool = True,
        gain: float = 1,
        device=None,
        dtype=None
    ) -> None:
        assert len(input_shape) == len(input_ranks)

        factory_kwargs = {"device": device, "dtype": dtype}
        super(TRLMasked, self).__init__()

        self.input_shape = input_shape
        self.input_ranks = input_ranks
        self.output_features = output_features
        self.output_rank = output_rank
        self.gain = gain
        self.link_ranks = link_ranks[-1:] + link_ranks
        if bias:
            self.bias = Parameter(torch.empty(self.output_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.factors = nn.ParameterList([
            Parameter(torch.empty((
                self.input_shape[i],
                self.link_ranks[i],
                self.link_ranks[i + 1],
                self.input_ranks[i]),
                **factory_kwargs))
            for i in range(len(self.input_shape))
        ])
        self.core = Parameter(
            torch.empty(self.input_ranks + (self.output_rank if self.output_rank is not None else self.output_features,),
            **factory_kwargs))
        self.masks = {}
        if output_rank is not None:
            self.output_factor = Parameter(torch.empty((self.output_rank, self.output_features), **factory_kwargs))
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
                axis_names=["batch"] + [f"in_{i}" for i, _ in enumerate(self.input_shape)],
                backend="pytorch")
        factors = [
            tn.Node(
                self.masked_tensor(self.factors[i], f"factor_{i}"),
                name=f"factor_{i}",
                axis_names=[f"in_{i}", f"link_left", f"link_right", f"rank_{i}"],
                backend="pytorch")
            for i, _ in enumerate(self.factors)
        ]
        core = tn.Node(
            self.masked_tensor(self.core, "core"),
            name="core",
            axis_names=[f"rank_{i}" for i in range(len(self.input_shape))] + ["out_rank"],
            backend="pytorch",
        )
        result_coures = factors + [core]
        for i in range(len(factors)):
            if input is not None:
                tensor[f"in_{i}"] ^ factors[i][f"in_{i}"]
            factors[i][f"rank_{i}"] ^ core[f"rank_{i}"]
            if i != 0:
                factors[i - 1]["link_right"] ^ factors[i]["link_left"]
        factors[-1]["link_right"] ^ factors[0]["link_left"]
        out_edge = core["out_rank"]
        if self.output_rank is not None:
            out_factor = tn.Node(
                self.masked_tensor(self.output_factor, "out_factor"),
                name="out_factor",
                axis_names=["out_rank", "out"],
                backend="pytorch",
            )
            out_edge = out_factor["out"]
            core["out_rank"] ^ out_factor["out_rank"]
            result_coures = result_coures + [out_factor]
        if input is not None:
            return result_coures + [tensor], [tensor["batch"], out_edge]
        return result_coures


    def calculate_std(self) -> float:
        network = self.construct_network()
        N = len(network)
        fan_in = np.prod(self.input_shape)
        fan_out = self.output_features
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
        torch_tensor = tensor.tensor.reshape((batch_size, self.output_features))
        if self.bias is None:
            return torch_tensor
        return torch.add(torch_tensor, self.bias)

    def extra_repr(self) -> str:
        return 'input_shape={}, output_features={}, input_ranks={}, output_rank={}, bias={}'.format(
            self.input_shape, self.output_features, self.input_ranks, self.output_rank, self.bias is not None
        )
