from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GAHparams:
    pop_size: Optional[int] = None
    n_generations: Optional[int] = None
    n_parents: Optional[int] = None
    n_elite_evals: Optional[int] = None
    n_elites: Optional[int] = None


# See https://arxiv.org/pdf/1712.06567.pdf
class SimpleGa:

    def __init__(
            self,
            logger,
            rng: np.random.Generator,
            init_individual,
            mutation_op,
            calculate_fitness,
            hparams: GAHparams
    ):
        self.logger = logger
        self.rng = rng
        self.init_individual = init_individual
        self.mutation_op = mutation_op
        self.calculate_fitness = calculate_fitness
        self.hparams = hparams

        self.pop = None
        self.elite = None
        self.fitness = None
        self.max_fitness_hist = []
        self.gen = 0

    def run(self):
        for gen in range(self.hparams.n_generations):
            self.step()
            max_fitness = np.max(self.fitness)
            self.logger.debug(f"Max fitness: {max_fitness}")
            self.max_fitness_hist.append(max_fitness)

        return self.pop[0]

    def evaluate(self):
        pass

    def step(self):
        def get_parent_index():
            return self.rng.integers(self.hparams.n_parents)

        self.logger.debug("calculating fitness...")

        if self.gen == 0:
            self.pop = [self.init_individual() for _ in range(self.hparams.pop_size)]
        else:
            self.pop = [self.mutation_op(self.pop[get_parent_index()]) for _ in range(self.hparams.pop_size)]

        self.fitness = [self.calculate_fitness(ind) for ind in self.pop]

        ids = np.argsort(self.fitness)[::-1]
        self.pop = [self.pop[i] for i in ids]

        self.logger.debug("finding elites...")
        if self.gen == 0:
            elites = self.pop[:self.hparams.n_elites]
        else:
            elites = [self.elite] + [p for p in self.pop[:self.hparams.n_elites] if p is not self.elite]
            elites = elites[:self.hparams.n_elites]

        self.logger.debug("finding elite...")
        elites_fitness = []
        for i in range(self.hparams.n_elites):
            fitness = np.mean([self.calculate_fitness(elites[i]) for _ in range(self.hparams.n_elite_evals)])
            elites_fitness.append(fitness)

        new_elite = elites[np.argmax(elites_fitness)]
        if new_elite != self.elite:
            self.logger.debug(f"Replacing elite")
            self.elite = new_elite

        if self.elite in self.pop:
            self.pop.remove(self.elite)
        self.pop.insert(0, self.elite)
        self.gen += 1
