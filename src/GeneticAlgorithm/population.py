import numpy as np
from typing import List, Callable
from src.GeneticAlgorithm.chromosome import CoffeeChromosome


class Population:
    """
    Population of candidate coffee chromosomes for the genetic algorithm.

    Manages a collection of `CoffeeChromosome` individuals, handling their
    evaluation, selection, crossover, and mutation during evolutionary optimization.
    """

    VALID_SELECTION_METHODS = ("tournament", "roulette")

    def __init__(
        self,
        size: int = 30,
        selection_method: str = "tournament",
        fitness_fn: Callable[[int, int, int, float], float] = None,
    ) -> None:
        """
        Initialize a population with random individuals.

        Parameters
        ----------
        size : int
            Number of individuals (default 30).
        selection_method : str
            'tournament' or 'roulette' (default 'tournament').
        fitness_fn : callable
            Fitness function that takes (roast, blend, grind, brew_time) and returns a float.

        Raises
        ------
        ValueError
            If selection_method is invalid or fitness_fn is None.
        """
        if selection_method not in self.VALID_SELECTION_METHODS:
            raise ValueError(
                f"Invalid selection_method '{selection_method}'. "
                f"Valid options are: {self.VALID_SELECTION_METHODS}"
            )
        if fitness_fn is None:
            raise ValueError("Population requires a fitness function.")

        self.selection_method: str = selection_method
        self.fitness_fn = fitness_fn
        self.individuals: List[CoffeeChromosome] = [
            CoffeeChromosome(fitness_fn=fitness_fn) for _ in range(size)
        ]

    def evaluate(self) -> None:
        """
        Evaluate the fitness of all individuals in the population.

        Each individual's fitness is computed using the population's fitness function.
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

    def _tournament_selection(self, k: int) -> List[CoffeeChromosome]:
        """
        Perform tournament selection to choose parent individuals.

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

        Returns
        -------
        List[CoffeeChromosome]
            List of selected parent chromosomes.
        """
        fitnesses = np.array([ind.fitness for ind in self.individuals])
        min_fit = fitnesses.min()
        if min_fit < 0:
            fitnesses -= min_fit  # shift so all fitnesses >= 0
        total_fitness = fitnesses.sum()
        if total_fitness == 0:
            probabilities = np.full(len(self.individuals), 1 / len(self.individuals))
        else:
            probabilities = fitnesses / total_fitness
        selected_indices = np.random.choice(
            len(self.individuals), size=len(self.individuals), p=probabilities
        )
        return [self.individuals[i] for i in selected_indices]

    def evolve(self, crossover_rate: float = 0.8, mutation_rate: float = 0.2) -> None:
        """
        Evolve the population through selection, crossover, and mutation.

        A new generation is created by:
          1. Selecting parents using the population's selection method.
          2. Applying crossover between parent pairs (with probability `crossover_rate`).
          3. Mutating offspring (with probability `mutation_rate`).
          4. Evaluating all new individuals.

        Parameters
        ----------
        crossover_rate : float, optional
            Probability of applying crossover to each parent pair (default 0.8).
        mutation_rate : float, optional
            Probability of mutating each offspring (default 0.2).
        """
        new_gen: List[CoffeeChromosome] = []

        # Selection
        parents = self.select_parents()

        # Crossover
        for i in range(0, len(parents), 2):
            if np.random.rand() < crossover_rate:
                c1, c2 = CoffeeChromosome.crossover(parents[i], parents[i + 1])
            else:
                c1, c2 = parents[i].copy(), parents[i + 1].copy()
            new_gen += [c1, c2]

        # Mutation
        for ind in new_gen:
            if np.random.rand() < mutation_rate:
                ind.mutate()

        # Update population and evaluate
        self.individuals = new_gen[: len(self.individuals)]
        self.evaluate()
