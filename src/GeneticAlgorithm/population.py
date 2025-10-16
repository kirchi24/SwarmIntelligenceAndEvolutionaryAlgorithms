import numpy as np
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

    VALID_SELECTION_METHODS = ("tournament", "roulette")

    def __init__(self, size: int = 30, selection_method: str = "tournament") -> None:
        """
        Initialize a population with random individuals.

        Parameters
        ----------
        size : int, optional
            Number of individuals in the population (default 30).
        selection_method : str, optional
            Method for parent selection ('tournament' or 'roulette'), default 'tournament'.

        Raises
        ------
        ValueError
            If `selection_method` is not 'tournament' or 'roulette'.
        """
        if selection_method not in self.VALID_SELECTION_METHODS:
            raise ValueError(
                f"Invalid selection_method '{selection_method}'. "
                f"Valid options are: {self.VALID_SELECTION_METHODS}"
            )

        self.individuals: List[CoffeeChromosome] = [
            CoffeeChromosome() for _ in range(size)
        ]
        self.selection_method: str = selection_method

    def evaluate(self) -> None:
        """
        Evaluate the fitness of all individuals in the population.

        Each individual's fitness is computed using the `coffee_fitness_4d`
        function defined inside the `CoffeeChromosome` class.
        """
        for ind in self.individuals:
            ind.evaluate()

    def best(self) -> CoffeeChromosome:
        """
        Return the best individual in the current population.

        Returns
        -------
        CoffeeChromosome
            The individual with the highest fitness score.
        """
        return max(self.individuals, key=lambda x: x.fitness)

    def select_parents(self, k: int = 3) -> List[CoffeeChromosome]:
        """
        Select parent individuals according to the population's selection method.

        Parameters
        ----------
        k : int, optional
            Tournament size if using tournament selection (default 3).

        Returns
        -------
        List[CoffeeChromosome]
            Selected parents for crossover.
        """
        if self.selection_method == "tournament":
            return self._tournament_selection(k)
        elif self.selection_method == "roulette":
            return self._roulette_selection()
        else:
            raise ValueError(
                "Unknown selection method. Use 'tournament' or 'roulette'."
            )

    def _tournament_selection(self, k: int) -> List[CoffeeChromosome]:
        """
        Perform tournament selection to choose parent individuals.

        In each tournament, `k` random individuals compete, and the one
        with the highest fitness is selected as a parent. This is repeated
        until the number of selected parents equals the population size.

        Parameters
        ----------
        k : int
            Tournament size (number of individuals competing per selection).

        Returns
        -------
        List[CoffeeChromosome]
            List of selected parent chromosomes.
        """
        parents: List[CoffeeChromosome] = []
        for _ in range(len(self.individuals)):
            contenders = np.random.choice(self.individuals, k)
            parents.append(max(contenders, key=lambda x: x.fitness))
        return parents


    def _roulette_selection(self) -> List[CoffeeChromosome]:
        """
        Perform roulette wheel (fitness-proportionate) selection.

        Each individual is assigned a probability proportional to its fitness.
        Parents are sampled according to these probabilities. Fitness values
        are shifted if negative to ensure all probabilities are non-negative.
        If all fitnesses are zero, selection is uniform.

        Returns
        -------
        List[CoffeeChromosome]
            List of selected parent chromosomes.
        """
        fitnesses = np.array([ind.fitness for ind in self.individuals])
        min_fit = fitnesses.min()
        if min_fit < 0:
            fitnesses -= min_fit  # shift so all fitnesses are >= 0

        total_fitness = fitnesses.sum()
        if total_fitness == 0:
            probabilities = np.full(len(self.individuals), 1 / len(self.individuals))
        else:
            probabilities = fitnesses / total_fitness

        selected_indices = np.random.choice(len(self.individuals), size=len(self.individuals), p=probabilities)
        parents = [self.individuals[i] for i in selected_indices]
        return parents


   
