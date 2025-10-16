from GeneticAlgorithm.chromosome import CoffeeChromosome
import numpy as np


class Population:
    """
    Population of candidate coffee chromosomes for the genetic algorithm.

    This class manages a collection of `CoffeeChromosome` individuals,
    handling their evaluation, selection, crossover, and mutation during
    evolutionary optimization.

    Attributes
    ----------
    individuals : list of CoffeeChromosome
        List containing all individuals in the current generation.
    """

    def __init__(self, size=30):
        """
        Initialize a population with random individuals.

        Parameters
        ----------
        size : int, optional
            Number of individuals in the population (default is 30).
        """
        self.individuals = [CoffeeChromosome() for _ in range(size)]
