import copy
from typing import List

import numpy as np
import torch
from torch import nn


class TreeNode:
    def __init__(self, value: int, prev: "TreeNode" = None):
        self.prev = prev
        self.value = value
        self._length = 1
        if prev is not None:
            self._length = len(prev) + 1

    def to_list(self) -> List[int]:
        sequence = []
        node = self
        while node is not None:
            sequence.append(node.value)
            node = node.prev

        return sequence[::-1]

    @staticmethod
    def from_list(sequence: List[int]) -> "TreeNode":
        node = None
        for s in sequence:
            if isinstance(s, int):
                new_node = TreeNode(s, node)
                node = new_node
            else:
                raise ValueError(f"Node value has to be int not {type(s)}")

        return node

    def append(self, value: int) -> "TreeNode":
        return TreeNode(value, self)

    def sequence_equals(self, other) -> bool:
        if isinstance(other, TreeNode):
            return self.to_list() == other.to_list()

        return False

    def __len__(self) -> int:
        return self._length

    def __repr__(self):
        nodes_str = ", ".join(map(str, self.to_list()[:3]))
        return f"{self.__class__.__name__} ({hex(id(self))}) with {len(self)} nodes ({nodes_str}, ...)"


def init_net(net: nn.Module, g, sigma=None):
    for layer in net.children():
        if isinstance(layer, nn.Linear):
            if sigma is None:
                layer.weight.copy_(torch.randn(layer.weight.shape, generator=g) * (1 / layer.in_features))
            else:
                layer.weight.copy_(torch.randn(layer.weight.shape, generator=g) * sigma)

            nn.init.zeros_(layer.bias)


@torch.no_grad()
def apply_seeds(net: nn.Module, seeds: List[int], sigma: float = 1, reset_net=False, copy_net=False) -> nn.Module:
    if copy_net:
        net = copy.deepcopy(net)

    if reset_net:
        for p in net.parameters():
            p.mul_(0)

    g = torch.Generator()
    g.manual_seed(seeds[0])

    init_net(net, g, sigma=None)

    size = sum(p.view(-1).shape[0] for p in net.parameters())
    noise = torch.zeros(size)
    for seed in seeds[1:]:
        g.manual_seed(seed)
        noise += sigma * torch.randn(size, generator=g)

    i = 0
    for p in net.parameters():
        p = p.view(-1)
        p.add_(noise[i:i + len(p)])
        i += len(p)

    return net


class RandomSeedGenerator:
    def __init__(self, seed: int, max_seed: int):
        self.rng = np.random.default_rng(seed)
        self.max_seed = max_seed

    def __call__(self) -> int:
        return self.rng.integers(self.max_seed).item()


class TreeNodeInitialiser:
    def __init__(self, seed_generator: RandomSeedGenerator):
        self.seed_generator = seed_generator

    def __call__(self) -> TreeNode:
        return TreeNode(self.seed_generator())


def reproduce(node: TreeNode, seed_generator: RandomSeedGenerator) -> TreeNode:
    return node.append(seed_generator())


class ReproductionOP:
    def __init__(self, seed_generator: RandomSeedGenerator):
        self.seed_generator = seed_generator

    def __call__(self, node: TreeNode) -> TreeNode:
        return reproduce(node, self.seed_generator)
