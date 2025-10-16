from typing import List

from GeneticAlgorithm.chromosome import CoffeeChromosome


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

    def __init__(self, size: int = 30) -> None:
        """
        Initialize a population with random individuals.

        Parameters
        ----------
        size : int, optional
            Number of individuals in the population (default is 30).
        """
        self.individuals: List[CoffeeChromosome] = [
            CoffeeChromosome() for _ in range(size)
        ]
