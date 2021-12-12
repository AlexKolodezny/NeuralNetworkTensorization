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
    

class TT(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int,...],
        ranks: Tuple[int, ...],
        bias: bool = True,
        device=None,
        dtype=None
    ) -> None:
        assert len(input_shape) == len(output_shape)
        assert len(input_shape) - 1 == len(ranks)

        factory_kwargs = {"device": device, "dtype": dtype}
        super(TT, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.ranks = (1,) + ranks + (1,)
        if bias:
            self.bias = Parameter(torch.empty(mul(self.output_shape), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.cores = nn.ParameterList([
            Parameter(torch.empty((self.input_shape[i], self.output_shape[i], self.ranks[i], self.ranks[i + 1]), **factory_kwargs))
            for i in range(len(self.ranks) - 1)
        ])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i, _ in enumerate(self.cores):
            xavier_normal(self.cores[i], [0, 2], [1, 3])
        if self.bias is not None:
            fan_in = mul(self.input_shape)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: Tensor) -> Tensor:
        batch_size = input.shape[0]
        tensor = tn.Node(
            input.reshape((batch_size,) + self.input_shape),
            name="input",
            axis_names=["batch"] + [f"in_{i}" for i, _ in enumerate(self.input_shape)],
            backend="pytorch")
        nodes = [
            tn.Node(
                self.cores[i],
                name=f"core_{i}",
                axis_names=[f"in_{i}", f"out_{i}", "rank_left", "rank_right"],
                backend="pytorch")
            for i, _ in enumerate(self.cores)
        ]
        for i in range(len(nodes) - 1):
            nodes[i]["rank_right"] ^ nodes[i + 1]["rank_left"]
        for i, _ in enumerate(self.input_shape):
            tensor[f"in_{i}"] ^ nodes[i][f"in_{i}"]
        for node in nodes:
            tensor = tn.contract_between(tensor, node)
        torch_tensor = tensor.tensor.reshape((batch_size, mul(self.output_shape)))
        if self.bias is None:
            return torch_tensor
        return torch.add(torch_tensor, self.bias)

    def extra_repr(self) -> str:
        return 'in_shape={}, out_shape={}, ranks={}, bias={}'.format(
            self.input_shape, self.output_shape, self.ranks, self.bias is not None
        )

        

        
