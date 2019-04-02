#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.


#    example which maximizes the sum of a list of integers
#    each of which can be 0 or 1

import random
import math
import matplotlib as mpl
import matplotlib.pylab as plt

from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, 16)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# the goal ('fitness') function to be maximized


def decode(target):
    ss = 0
    addtive = 1
    for i in range(7, -1, -1):
        ss += addtive * target[i]
        addtive *= 2
    ss = ss / 256.0 * 60
    ss = ss - 30
    return ss


def griewank(x, y):
    ret = ((x**2) + (y**2)) / 4000
    ret = ret - math.cos(x) * math.cos(y / (2**0.5)) + 1
    return ret


def evalOneMax(individual):
    a = list(individual[0:8])
    b = list(individual[8:16])
    aa = decode(a)
    bb = decode(b)
    return (-griewank(aa, bb)),


def distance(u, v):
    (x1, y1) = (decode(u[0:8]), decode(u[8:16]))
    (x2, y2) = (decode(v[0:8]), decode(v[8:16]))
    return math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))


def similar(val):
    nicheRadius = 1.0
    if(val < nicheRadius):
        ret = 1 - (val/nicheRadius)
    else:
        ret = 0
    return ret

# Operator registration
# register the goal / fitness function
toolbox.register("evaluate", evalOneMax)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)


def main():

    random.seed(64)
    f = open("result_pop1000_CX50_MUT50_G100_NR1.0.txt", 'w')
    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=1000)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.5
    f.write("Start of evolution\n")
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for ind in pop:
        divisor = 0
        for other in pop:
            divisor += similar(distance(ind, other))
        for vals in ind.fitness.values:
            vals /= divisor

    f.write("  Evaluated %i individuals\n" % len(pop))
    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    plt.xlim(-30, 30)
    plt.ylim(-30, 30)
    # Begin the evolution
    while max(fits) < 50 and g < 100:
        plt.clf()
        # A new generation
        g = g + 1
        f.write("-- Generation %i --\n" % g)
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        for ind in invalid_ind:
            divisor = 0
            for others in offspring:
                divisor += similar(distance(ind, others))
            for vals in ind.fitness.values:
                vals /= divisor
        f.write("  Evaluated %i individuals\n" % len(invalid_ind))
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        if(g % 5 == 1):
            px = []
            py = []
            for out in pop:
                a = decode(out[0:8])
                b = decode(out[8:16])
                px.append(a)
                py.append(b)
            plt.plot(px, py, 'ro')
            plt.show()
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        f.write("  Max %s\n" % -min(fits))
        f.write("  Min %s\n" % -max(fits))
        f.write("  Avg %s\n" % -mean)
        f.write("  Std %s\n" % std)
    f.write("-- End of (successful) evolution --\n")
    best_ind = tools.selBest(pop, 1)[0]
    f.write("Best individual is ( %f , %f ). In this case, f(x,y) = %f\n" %
            (decode(best_ind[0:8]), decode(best_ind[8:16]),
             -best_ind.fitness.values[0]))

    f.close()


if __name__ == "__main__":
    main()
