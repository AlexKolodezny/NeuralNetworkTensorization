from typing import Tuple
from tensornetwork.network_components import contract_between
import torch
from torch import Tensor
from torch import nn
from typing import Tuple
from torch.nn import init
import math

import tensornetwork as tn

from torch.nn.parameter import Parameter
from src.utils import mul, xavier_normal


class HTSubTree(nn.Module):
    def __init__(
        self,
        is_left_child: bool,
        interval: Tuple[int, int],
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        rank: int,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(HTSubTree, self).__init__()

        self.interval = interval
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.rank = rank
        self.is_left_child = is_left_child

        if self._is_factor():
            pos = interval[0]
            self.root = Parameter(torch.empty((self.input_shape[pos], self.output_shape[pos], rank), **factory_kwargs))
        else:
            self.root = Parameter(torch.empty((rank,) * 3, **factory_kwargs))
            m = (interval[0] + interval[1]) // 2
            self.left_child = HTSubTree(True, (interval[0], m), self.input_shape, self.output_shape, self.rank, **factory_kwargs)
            self.right_child = HTSubTree(False, (m, interval[1]), self.input_shape, self.output_shape, self.rank, **factory_kwargs)
        
        self.reset_parameters()

    def _is_factor(self) -> bool:
        return self.interval[0] == self.interval[1] - 1

    def reset_parameters(self) -> None:
        if self._is_factor():
            xavier_normal(self.root, [0], [1, 2])
        else:
            if self.is_left_child:
                xavier_normal(self.root, [0], [1, 2])
            else:
                xavier_normal(self.root, [0, 2], [1])
            self.left_child.reset_parameters()
            self.right_child.reset_parameters()
    
    def build_network(self, input: tn.Node) -> tn.Node:
        if self._is_factor():
            pos = self.interval[0]
            self.node = tn.Node(
                self.root,
                name=f"factor{pos}",
                axis_names=[f"in_{pos}", f"out_{pos}", "parent"],
                backend="pytorch"
            )
            input[f"in_{pos}"] ^ self.node[f"in_{pos}"]
            return self.node
        else:
            self.node = tn.Node(
                self.root,
                name=f"core{self.interval}",
                axis_names=["left", "right", "parent"],
                backend="pytorch",
            )
            left_node = self.left_child.build_network(input)
            right_node = self.right_child.build_network(input)
            left_node["parent"] ^ self.node["left"]
            right_node["parent"] ^ self.node["right"]
            return self.node
    
    def forward(self, tensor: tn.Node) -> tn.Node:
        if self._is_factor():
            return tn.contract_between(tensor, self.node)
        else:
            tensor = self.left_child.forward(tensor)
            tensor = tn.contract_between(tensor, self.node)
            return self.right_child.forward(tensor)


class HT(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        rank: int,
        bias: bool = True,
        device=None,
        dtype=None
    ) -> None:
        assert len(input_shape) == len(output_shape)

        factory_kwargs = {"device": device, "dtype": dtype}
        super(HT, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.rank = rank
        if bias:
            self.bias = Parameter(torch.empty(mul(self.output_shape), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        interval_l = 0
        interval_r = len(input_shape)
        interval_m = (interval_l + interval_r) // 2
        self.left_subtree = HTSubTree(True, (interval_l, interval_m), input_shape, output_shape, rank, **factory_kwargs)
        self.right_subtree = HTSubTree(False, (interval_m, interval_r), input_shape, output_shape, rank, **factory_kwargs)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.left_subtree.reset_parameters()
        self.right_subtree.reset_parameters()
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
        left_node = self.left_subtree.build_network(tensor)
        right_node = self.right_subtree.build_network(tensor)
        left_node["parent"] ^ right_node["parent"]
        tensor = self.left_subtree.forward(tensor)
        tensor = self.right_subtree.forward(tensor)

        torch_tensor = tensor.tensor.reshape((batch_size, mul(self.output_shape)))
        if self.bias is None:
            return torch_tensor
        return torch.add(torch_tensor, self.bias)

    def extra_repr(self) -> str:
        return 'input_shape={}, output_features={}, input_ranks={}, output_rank={}, bias={}'.format(
            self.input_shape, self.output_features, self.input_ranks, self.output_rank, self.bias is not None
        )
