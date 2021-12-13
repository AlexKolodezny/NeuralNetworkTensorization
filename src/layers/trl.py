from typing import Tuple
import torch
from torch import Tensor
from torch import nn
from typing import Tuple
from functools import reduce
import operator
from torch.nn import init
import math

import tensornetwork as tn

from torch.nn.parameter import Parameter
from src.utils import mul, xavier_normal


class TRL(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        input_ranks: Tuple[int, ...],
        output_features: int,
        output_rank: int,
        bias: bool = True,
        device=None,
        dtype=None
    ) -> None:
        assert len(input_shape) == len(input_ranks)

        factory_kwargs = {"device": device, "dtype": dtype}
        super(TRL, self).__init__()

        self.input_shape = input_shape
        self.input_ranks = input_ranks
        self.output_features = output_features
        self.output_rank = output_rank
        if bias:
            self.bias = Parameter(torch.empty(self.output_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.factors = nn.ParameterList([
            Parameter(torch.empty((self.input_shape[i], self.input_ranks[i]), **factory_kwargs))
            for i in range(len(self.input_shape))
        ])
        self.core = Parameter(torch.empty(self.input_ranks + (self.output_rank,), **factory_kwargs))
        self.output_factor = Parameter(torch.empty((self.output_rank, self.output_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i, _ in enumerate(self.factors):
            xavier_normal(self.factors[i], [0], [1])
        xavier_normal(self.core, list(range(len(self.input_shape))), [len(self.input_shape)])
        xavier_normal(self.output_factor, [0], [1])
        if self.bias is not None:
            fan_in = mul(self.input_shape)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: Tensor) -> Tensor:
        batch_size = input.shape[0]
        tensor = tn.Node(
            input,
            name="input",
            axis_names=["batch"] + [f"in_{i}" for i, _ in enumerate(self.input_shape)],
            backend="pytorch")
        factors = [
            tn.Node(
                self.factors[i],
                name=f"factor_{i}",
                axis_names=[f"in_{i}", f"rank_{i}"],
                backend="pytorch")
            for i, _ in enumerate(self.factors)
        ]
        core = tn.Node(
            self.core,
            name="core",
            axis_names=[f"rank_{i}" for i in range(len(self.input_shape))] + ["out_rank"],
            backend="pytorch",
        )
        out_factor = tn.Node(
            self.output_factor,
            name="out_factor",
            axis_names=["out_rank", "out"],
            backend="pytorch",
        )
        for i in range(len(factors)):
            tensor[f"in_{i}"] ^ factors[i][f"in_{i}"]
            factors[i][f"rank_{i}"] ^ core[f"rank_{i}"]
        core["out_rank"] ^ out_factor["out_rank"]
        for node in factors:
            tensor = tn.contract_between(tensor, node)
        tensor = tn.contract_between(tensor, core)
        tensor = tn.contract_between(tensor, out_factor)
        torch_tensor = tensor.tensor.reshape((batch_size, self.output_features))
        if self.bias is None:
            return torch_tensor
        return torch.add(torch_tensor, self.bias)

    def extra_repr(self) -> str:
        return 'input_shape={}, output_features={}, input_ranks={}, output_rank={}, bias={}'.format(
            self.input_shape, self.output_features, self.input_ranks, self.output_rank, self.bias is not None
        )
