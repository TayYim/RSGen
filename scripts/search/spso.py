import itertools
import math
import operator
import random
import pickle
import os

import numpy as np

try:
    from itertools import imap
except:
    # Python 3 nothing to do
    pass
else:
    map = imap

from deap import base
from deap import creator
from deap import tools


class SPSO:
    def __init__(self, func, dim, pop, max_iter, lb, ub, w, c1, c2, output_path):
        self.pop = pop
        self.dim = dim
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.func = func
        self.particles = []
        self.output_path = output_path

        self.bounds = np.array([lb, ub])

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create(
            "Particle",
            list,
            fitness=creator.FitnessMin,
            speed=list,
            best=None,
            bestfit=creator.FitnessMin,
        )

        toolbox = base.Toolbox()
        toolbox.register(
            "particle",
            self.generate,
            creator.Particle,
            dim=self.dim,
            pmin=self.bounds[0],
            pmax=self.bounds[1],
            smin=-(self.bounds[1] - self.bounds[0]) / 2.0,
            smax=(self.bounds[1] - self.bounds[0]) / 2.0,
        )
        toolbox.register("swarm", tools.initRepeat, list, toolbox.particle)
        toolbox.register("update", self.updateParticle, chi=0.729843788, c=2.05)
        toolbox.register("convert", self.convert_quantum)
        toolbox.register("evaluate", self.func)
        self.toolbox = toolbox

    def generate(self, pclass, dim, pmin, pmax, smin, smax):
        part = pclass(random.uniform(pmin[i], pmax[i]) for i in range(dim))
        part.speed = [random.uniform(smin[i], smax[i]) for i in range(dim)]
        return part

    def convert_quantum(self, swarm, rcloud, centre):
        dim = len(swarm[0])
        for part in swarm:
            position = [random.gauss(0, 1) for _ in range(dim)]
            dist = math.sqrt(sum(x**2 for x in position))

            # Gaussian distribution
            # u = abs(random.gauss(0, 1.0/3.0))
            # part[:] = [(rcloud * x * u**(1.0/dim) / dist) + c for x, c in zip(position, centre)]

            # UVD distribution
            # u = random.random()
            # part[:] = [(rcloud * x * u**(1.0/dim) / dist) + c for x, c in zip(position, centre)]

            # NUVD distribution
            u = abs(random.gauss(0, 1.0 / 3.0))
            part[:] = [(rcloud * x * u / dist) + c for x, c in zip(position, centre)]

            del part.fitness.values
            del part.bestfit.values
            part.best = None

        return swarm

    def updateParticle(self, part, best, chi, c):
        ce1 = (c * random.uniform(0, 1) for _ in range(len(part)))
        ce2 = (c * random.uniform(0, 1) for _ in range(len(part)))
        ce1_p = map(operator.mul, ce1, map(operator.sub, best, part))
        ce2_g = map(operator.mul, ce2, map(operator.sub, part.best, part))
        a = map(
            operator.sub,
            map(operator.mul, itertools.repeat(chi), map(operator.add, ce1_p, ce2_g)),
            map(operator.mul, itertools.repeat(1 - chi), part.speed),
        )
        part.speed = list(map(operator.add, part.speed, a))
        part[:] = list(map(operator.add, part, part.speed))

    def run(self, verbose=True, checkpoint=None):
        RS = (self.bounds[1] - self.bounds[0]) / (
            50 ** (1.0 / self.dim)
        )  # between 1/20 and 1/10 of the domain's range
        print(RS)
        PMAX = 10
        RCLOUD = 1.0  # 0.5 times the move severity

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        if checkpoint:
            # A file name has been given, then load the data from the file
            with open(checkpoint, "rb") as cp_file:
                cp = pickle.load(cp_file)

            generation = cp["generation"]
            swarm = cp["swarm"]
            logbook = cp["logbook"]
        else:
            # Start a new evolution

            logbook = tools.Logbook()
            logbook.header = ["gen", "nswarm", "evals"] + stats.fields

            swarm = self.toolbox.swarm(n=self.pop)

            generation = 0

        while generation < self.max_iter:
            # Evaluate each particle in the swarm
            for part in swarm:
                part.fitness.values = self.toolbox.evaluate(part)
                if not part.best or part.bestfit < part.fitness:
                    part.best = self.toolbox.clone(part[:])  # Get the position
                    part.bestfit.values = part.fitness.values  # Get the fitness

            # Sort swarm into species, best individual comes first
            sorted_swarm = sorted(swarm, key=lambda ind: ind.bestfit, reverse=True)
            species = []
            while sorted_swarm:
                found = False
                for s in species:
                    # dists = ((x1 - x2)**2 for x1, x2 in zip(sorted_swarm[0].best, s[0].best))
                    dists = np.array(sorted_swarm[0].best) - np.array(s[0].best)

                    far = False
                    for i in range(self.dim):
                        if dists[i] > RS[i]:
                            far = True
                            break

                    # dist = math.sqrt(sum((x1 - x2)**2 for x1, x2 in zip(sorted_swarm[0].best, s[0].best)))
                    # if dist <= RS:
                    if not far:
                        found = True
                        s.append(sorted_swarm[0])
                        break
                if not found:
                    species.append([sorted_swarm[0]])
                sorted_swarm.pop(0)

            record = stats.compile(swarm)
            logbook.record(
                gen=generation, nswarm=len(species), evals=len(swarm), **record
            )

            if verbose:
                print(logbook.stream)

            # Detect change
            if any(
                s[0].bestfit.values != self.toolbox.evaluate(s[0].best) for s in species
            ):
                # Convert particles to quantum particles
                for s in species:
                    s[:] = self.toolbox.convert(s, rcloud=RCLOUD, centre=s[0].best)

            else:
                # Replace exceeding particles in a species with new particles
                for s in species:
                    if len(s) > PMAX:
                        n = len(s) - PMAX
                        del s[PMAX:]
                        s.extend(self.toolbox.swarm(n=n))

                # Update particles that have not been reinitialized
                for s in species[:-1]:
                    for part in s[:PMAX]:
                        self.toolbox.update(part, s[0].best)
                        del part.fitness.values

            # Return all but the worst species' updated particles to the swarm
            # The worst species is replaced by new particles
            swarm = list(
                itertools.chain(self.toolbox.swarm(n=len(species[-1])), *species[:-1])
            )
            generation += 1

            # save checkpoint
            cp = dict(
                generation=generation,
                swarm=swarm,
                logbook=logbook,
            )
            self._save_pickle(self.output_path, "checkpoint.pkl", cp)

        return species, logbook

    def _save_pickle(self, path, filename, data):
        """
        save dict data to csv
        """
        file_path = os.path.join(path, filename)
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(data, file)


# main test
if __name__ == "__main__":

    def rosenbrock(particle):
        return ((1 - particle[0]) ** 2 + 100 * (particle[1] - particle[0] ** 2) ** 2,)

    pso = SPSO(
        func=rosenbrock,
        dim=2,
        pop=100,
        max_iter=500,
        lb=[-500, -100],
        ub=[500, 100],
        w=1,
        c1=2,
        c2=2,
        output_path=".",
    )
    species, logbook = pso.run()

    # print(len(species))
    # for s in species:
    #     if len(s) == 1:
    #         average = s[0].bestfit.values[0]
    #     else:
    #         average = sum(p.bestfit.values[0] for p in s if p.best is not None) / len(s)
    #     print(s[0].bestfit.values[0], s[0].best, len(s), average)
