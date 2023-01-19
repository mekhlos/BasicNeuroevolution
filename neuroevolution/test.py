import copy
import logging

import numpy as np
import pytest
import torch

from neuroevolution.baseline import GAHparams, SimpleGa
from neuroevolution.nn import Network
from neuroevolution.seed_encoded_nets import RandomSeedGenerator, TreeNodeInitialiser, ReproductionOP, apply_seeds
from neuroevolution.utils import accuracy


@pytest.fixture
def seed_generator():
    max_seed = 1000000
    seed_generator = RandomSeedGenerator(0, max_seed)
    return seed_generator


@pytest.fixture
def tree_init(seed_generator):
    return TreeNodeInitialiser(seed_generator)


@pytest.fixture
def reproduce(seed_generator):
    return ReproductionOP(seed_generator)


@pytest.fixture
def hparams():
    return GAHparams(pop_size=30, n_generations=5, n_parents=5, n_elite_evals=3, n_elites=3)


@pytest.fixture
def ga(hparams, tree_init, reproduce):
    rng = np.random.default_rng(0)

    x = rng.random((200, 2)).round()
    y = (x.sum(1).astype(int) == 1).astype(int)

    def calculate_fitness(node) -> float:
        net = apply_seeds(Network(2, 10, 1), node.to_list(), reset_net=False, sigma=0.1)
        y_pred = net(torch.from_numpy(x).float()).numpy()
        return accuracy(y, y_pred.round())

    logger = logging.getLogger()
    return SimpleGa(logger, rng, tree_init, reproduce, calculate_fitness, hparams)


def test_init(tree_init):
    node1 = tree_init()
    node2 = tree_init()

    assert node1 != node2


def test_reproduce(tree_init, reproduce):
    node = tree_init()
    node2 = reproduce(node)

    assert node != node2


def test_net_init(tree_init):
    node = tree_init()
    net1 = apply_seeds(Network(2, 10, 1), node.to_list(), sigma=1)
    net2 = apply_seeds(Network(2, 10, 1), node.to_list(), sigma=1)

    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        assert torch.equal(p1, p2)


def test_net_reproduce(tree_init, reproduce):
    node1 = tree_init()
    node2 = reproduce(node1)
    net1 = apply_seeds(Network(2, 10, 1), node1.to_list(), sigma=1)
    net2 = apply_seeds(Network(2, 10, 1), node2.to_list(), sigma=1)
    net3 = apply_seeds(Network(2, 10, 1), node2.to_list(), sigma=1)

    for p1, p2, p3 in zip(net1.parameters(), net2.parameters(), net3.parameters()):
        assert not torch.equal(p1, p2)
        assert torch.equal(p2, p3)


def test_net_reproduce2():
    net1 = apply_seeds(Network(2, 10, 1), [1, 3, 2, 12], sigma=1)
    net2 = apply_seeds(Network(2, 10, 1), [1, 12, 2, 3], sigma=1)
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        torch.testing.assert_allclose(p1, p2)


def test_mutation_does_not_change_original(tree_init, reproduce):
    node = tree_init()
    node1 = copy.deepcopy(node)
    node2 = reproduce(node)
    assert node1.sequence_equals(node)
    assert not node2.sequence_equals(node)


def test_params(tree_init, reproduce):
    node = tree_init()
    n = 100
    for i in range(n):
        node = reproduce(node)

    assert len(node.to_list()) == n + 1
    net = apply_seeds(Network(2, 1000, 1), node.to_list(), sigma=0.1)
    params = torch.cat([p.view(-1) for p in net.parameters()])
    print(params.mean(), params.std())


def test_ga(ga):
    ga.step()
    assert ga.gen == 1
    assert len(ga.pop) == ga.hparams.pop_size
    assert ga.pop[0] == ga.elite
    ga.step()
    nodes = [tuple(node.to_list()[:-1]) for node in ga.pop[1:]]
    assert len(set(nodes)) == ga.hparams.n_parents
