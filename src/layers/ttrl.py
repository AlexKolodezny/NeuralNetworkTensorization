from typing import Tuple
import torch
from torch import Tensor
from torch import nn
from typing import Tuple
from torch.nn import init
import math

import tensornetwork as tn

from torch.nn.parameter import Parameter
from src.utils import mul, xavier_normal


class TTRL(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[Tuple[int, ...], ...],
        input_ranks: Tuple[Tuple[int, ...], ...],
        output_shape: Tuple[int, ...],
        output_rank: Tuple[int, ...],
        bias: bool = True,
        device=None,
        dtype=None
    ) -> None:
        assert len(input_shape) == len(input_ranks)
        for input, ranks in zip(input_shape, input_ranks):
            assert len(input) == len(ranks)
        assert len(output_shape) == len(output_rank)

        factory_kwargs = {"device": device, "dtype": dtype}
        super(TTRL, self).__init__()

        self.input_shape = input_shape
        self.input_ranks = input_ranks
        self.output_shape = output_shape
        self.output_rank = output_rank
        if bias:
            self.bias = Parameter(torch.empty(self.output_shape, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.factors = nn.ParameterList([
            Parameter(torch.empty((1 if i == 0 else ranks[i - 1], shapes[i], ranks[i]), **factory_kwargs))
            for shapes, ranks in zip(self.input_shape, self.input_ranks) for i in range(len(shapes))
        ])
        self.core = Parameter(
            torch.empty(tuple(ranks[-1] for ranks in self.input_ranks) + (self.output_rank[-1],), **factory_kwargs)
        )
        self.output_factor = nn.ParameterList([
            Parameter(torch.empty(
                (1 if i == 0 else self.output_rank[i - 1], self.output_shape[i], self.output_rank[i]),
                **factory_kwargs
            ))
            for i in range(len(self.output_shape))
        ])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i, _ in enumerate(self.factors):
            xavier_normal(self.factors[i], [0, 1], [2])
        xavier_normal(self.core, list(range(len(self.input_shape))), [len(self.input_shape)])
        for j, _ in enumerate(self.output_factor):
            xavier_normal(self.output_factor[j], [0, 1], [2])
        if self.bias is not None:
            fan_in = mul([mul(shape) for shape in self.input_shape])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: Tensor) -> Tensor:
        batch_size = input.shape[0]
        tensor = tn.Node(
            input.reshape(input.shape + (1,) * len(self.input_shape)),
            name="input",
            axis_names=["batch"] + \
                [f"in_{i}_{j}" for i, shape in enumerate(self.input_shape) for j, _ in enumerate(shape)] + \
                [f"rank_{i}_{-1}" for i, _ in enumerate(self.input_shape)],
            backend="pytorch")
        factors = []
        cnt = 0
        for i, shapes in enumerate(self.input_shape):
            factors.append([])
            for j, _ in enumerate(shapes):
                factors[-1].append(
                    tn.Node(
                        self.factors[cnt],
                        name=f"factor_{i}_{j}",
                        axis_names=[f"rank_{i}_{j - 1}", f"in_{i}_{j}", f"rank_{i}_{j}"],
                        backend="pytorch")
                )
                cnt += 1

        core = tn.Node(
            self.core,
            name="core",
            axis_names=[f"rank_{i}" for i in range(len(self.input_shape))] + ["out_rank"],
            backend="pytorch",
        )
        out_factors = [
            tn.Node(
                self.output_factor[i],
                name="out_factor",
                axis_names=[f"rank_{i - 1}", f"out_{i}", f"rank_{i}"],
                backend="pytorch",
            ) for i, _ in enumerate(self.output_factor)
        ]
        for i, fact in enumerate(factors):
            tensor[f"rank_{i}_{-1}"] ^ factors[i][0][f"rank_{i}_{-1}"]
            tensor[f"in_{i}_{0}"] ^ factors[i][0][f"in_{i}_{0}"]
            for j in range(1, len(fact)):
                tensor[f"in_{i}_{j}"] ^ factors[i][j][f"in_{i}_{j}"]
                factors[i][j - 1][f"rank_{i}_{j - 1}"] ^ factors[i][j][f"rank_{i}_{j - 1}"]
            factors[i][-1][f"rank_{i}_{len(fact) - 1}"] ^ core[f"rank_{i}"]
        core["out_rank"] ^ out_factors[-1][f"rank_{len(out_factors) - 1}"]
        for j in range(1, len(out_factors)):
            out_factors[j - 1][f"rank_{j - 1}"] ^ out_factors[j][f"rank_{j - 1}"]
        for nodes in factors:
            for node in nodes:
                tensor = tn.contract_between(tensor, node)
        tensor = tn.contract_between(tensor, core)
        for node in out_factors[::-1]:
            tensor = tn.contract_between(tensor, node)
        assert tensor.tensor.shape[0] == input.shape[0]
        torch_tensor = tensor.tensor.reshape((batch_size,) + self.output_shape)
        if self.bias is None:
            return torch_tensor
        return torch.add(torch_tensor, self.bias)

    def extra_repr(self) -> str:
        return 'input_shape={}, output_features={}, input_ranks={}, output_rank={}, bias={}'.format(
            self.input_shape, self.output_shape, self.input_ranks, self.output_rank, self.bias is not None
        )
