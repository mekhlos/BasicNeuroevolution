import logging

import numpy as np
import torch

from neuroevolution.baseline import SimpleGa, GAHparams
from neuroevolution.nn import Network
from neuroevolution.seed_encoded_nets import RandomSeedGenerator, TreeNodeInitialiser, ReproductionOP, TreeNode, \
    apply_seeds
from neuroevolution.utils import mse, accuracy

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

rng = np.random.default_rng(0)
max_seed = 1000
seed_generator = RandomSeedGenerator(0, max_seed)
tree_init = TreeNodeInitialiser(seed_generator)
reproduce = ReproductionOP(seed_generator)
x = rng.random((200, 2)).round()
y = (x.sum(1).astype(int) == 1).astype(int)


def node_to_net(node: TreeNode):
    return apply_seeds(Network(2, 10, 1), node.to_list(), sigma=0.1)


def calculate_fitness(node: TreeNode) -> float:
    net = node_to_net(node)
    y_pred = net(torch.from_numpy(x).float()).numpy().flatten()
    mse_score = mse(y, y_pred)
    acc_score = accuracy(y, y_pred.round())
    return -mse_score


if __name__ == '__main__':
    hparams = GAHparams(pop_size=1000, n_generations=20, n_parents=20, n_elite_evals=10, n_elites=10)
    ga = SimpleGa(logger, rng, tree_init, reproduce, calculate_fitness, hparams)

    ga.run()
