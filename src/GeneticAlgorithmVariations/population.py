import numpy as np
from typing import List, Callable
from src.GeneticAlgorithmVariations.chromosome import ImageChromosome


class Population:
    """
    Population of ImageChromosome individuals for a genetic algorithm.

    Handles evaluation, selection, crossover, mutation, and survivor selection.
    """

    VALID_PARENT_METHODS = ("tournament", "rank")
    VALID_SURVIVOR_METHODS = ("fitness", "age")

    def __init__(
        self,
        size: int = 30,
        parent_selection: str = "tournament",
        survivor_method: str = "fitness",
        fitness_fn: Callable[[np.ndarray], float] = None,
        mutation_method: str = "uniform_local",
        mutation_rate: float = 0.2,
        crossover_method: str = "arithmetic",
        alpha: float = 0.5,  # for arithmetic crossover
    ) -> None:
        """
        Initialize a population of ImageChromosome individuals with configurable GA parameters.

        Parameters
        ----------
        size : int
            Number of individuals in the population.
        parent_selection : str
            Selection method: "tournament" or "rank".
        survivor_method : str
            Survivor selection method: "fitness" or "age".
        fitness_fn : callable
            Fitness function accepting an np.ndarray of genes.
        mutation_method : str
            Mutation method for all chromosomes ("uniform_local" or "gaussian_adaptive").
        mutation_rate : float
            Probability of mutation for each chromosome.
        crossover_method : str
            Crossover method for all chromosomes ("arithmetic" or "global_uniform").
        alpha : float
            Weight for arithmetic crossover (default 0.5).

        Raises
        ------
        ValueError
            If parent_selection, survivor_selection, mutation_method, or crossover_method is invalid, or if fitness_fn is None.
        """
        self.size = size
        self.parent_selection = parent_selection
        self.fitness_fn = fitness_fn
        self.survivor_method = survivor_method
        self.mutation_method = mutation_method
        self.mutation_rate = mutation_rate
        self.crossover_method = crossover_method
        self.alpha = alpha

        if fitness_fn is None:
            raise ValueError("A fitness function must be provided.")
        if parent_selection not in self.VALID_PARENT_METHODS:
            raise ValueError(f"Invalid selection method: {parent_selection}")
        if mutation_method not in ImageChromosome.VALID_MUTATION_METHODS:
            raise ValueError(f"Invalid mutation method: {mutation_method}")
        if crossover_method not in ImageChromosome.VALID_CROSSOVER_METHODS:
            raise ValueError(f"Invalid crossover method: {crossover_method}")

        # create initial random population
        self.individuals: List[ImageChromosome] = [
            ImageChromosome(
                fitness_fn=self.fitness_fn,
                mutation_method=self.mutation_method,
                crossover_method=self.crossover_method,
                alpha=self.alpha,
            )
            for _ in range(size)
        ]
        self.evaluate()

    def evaluate(self) -> None:
        """
        Evaluate the fitness of all individuals in the population.

        Each individual's fitness is computed using the population's fitness function.
        """
        for ind in self.individuals:
            try:
                ind.evaluate()
            except Exception as e:
                print(f"[WARN] Evaluation failed for {ind}: {e}")
                ind.fitness = 0  # return worst fitness on failure

    def best(self) -> ImageChromosome:
        """
        Return the best individual in the current population.

        Returns
        -------
        ImageChromosome
            The individual with the highest fitness score.
        """
        # All individuals should have a fitness (evaluate() sets fitness to 0 on error)
        fitness_values = np.array([ind.fitness for ind in self.individuals])
        best_index = np.argmax(fitness_values)

        return self.individuals[best_index]

    def _select_parents(
        self, k: int = 3, selection_pressure: float = 1.7
    ) -> List[ImageChromosome]:
        """
        Select parent individuals according to the population's selection method.

        Parameters
        ----------
        k : int, optional
            Tournament size if using tournament selection (default 3).
        selection_pressure : float, optional
            Selection pressure if using rank-based selection (default 1.7, valid range [1.0, 2.0]).

        Returns
        -------
        List[ImageChromosome]
            Selected parents for crossover.
        """
        if self.parent_selection == "tournament":
            return self._tournament_selection(k)
        elif self.parent_selection == "rank":
            return self._rank_selection(selection_pressure)

    def _tournament_selection(self, k: int) -> List[ImageChromosome]:
        """
        Perform tournament selection to choose parent individuals.

        Parameters
        ----------
        k : int
            Tournament size (number of individuals competing per selection).

        Returns
        -------
        List[ImageChromosome]
            List of selected parent chromosomes.
        """
        n = len(self.individuals)
        fitness = np.array([ind.fitness for ind in self.individuals])
        parents: List[ImageChromosome] = []

        for _ in range(n):
            idx = np.random.choice(n, size=k, replace=False)
            best_idx = idx[np.argmax(fitness[idx])]
            parents.append(self.individuals[best_idx])

        return parents

    def _rank_selection(self, selection_pressure: float = 1.7) -> List[ImageChromosome]:
        """
        Perform linear rank-based selection.

        Parameters
        ----------
        selection_pressure : float, optional
            Selection pressure in [1.0, 2.0]. Higher values favor fitter individuals.

        Returns
        -------
        List[ImageChromosome]
            List of selected parent chromosomes.
        """
        n = len(self.individuals)
        sorted_inds = sorted(self.individuals, key=lambda x: x.fitness)
        ranks = np.arange(1, n + 1)

        # Compute probabilities linearly from ranks
        probs = (2 - selection_pressure) / n + 2 * ranks * (selection_pressure - 1) / (
            n * (n - 1)
        )
        probs /= probs.sum()

        indices = np.random.choice(n, size=n, p=probs)
        return [sorted_inds[i] for i in indices]

    def _select_survivors(
        self, combined: list[ImageChromosome]
    ) -> list[ImageChromosome]:
        # update ages
        for ind in combined:
            ind.age = getattr(ind, "age", 0) + 1

        if self.survivor_method == "fitness":
            return self._fitness_based_survivor(combined)
        elif self.survivor_method == "age":
            return self._age_based_survivor(combined)

    def _fitness_based_survivor(
        self, combined: list[ImageChromosome]
    ) -> list[ImageChromosome]:
        combined.sort(key=lambda x: x.fitness, reverse=True)
        return combined[: self.size]

    def _age_based_survivor(
        self, combined: list[ImageChromosome]
    ) -> list[ImageChromosome]:
        combined.sort(key=lambda x: getattr(x, "age", 0))
        survivors = combined[-self.size :]
        return survivors

    def evolve(self) -> None:
        """
        Perform one generation of evolution: selection, crossover, mutation, and survivor selection.
        """
        parents = self._select_parents()
        n_parents = len(parents)
        offspring: List[ImageChromosome] = []

        max_fitness = 1.0

        # process pairs
        for i in range(0, n_parents - 1, 2):
            p1, p2 = parents[i], parents[i + 1]

            # crossover
            children = p1.crossover(p2)

            # mutation + evaluation
            for child in children:
                if np.random.rand() < self.mutation_rate:
                    child.mutate(max_fitness)
                child.evaluate()
                offspring.append(child)

        # handle odd parent
        if n_parents % 2 == 1:
            last = parents[-1].copy()
            if np.random.rand() < self.mutation_rate:
                last.mutate(max_fitness)
            last.evaluate()
            offspring.append(last)

        # combine old + new, and select survivors
        combined = self.individuals + offspring
        combined = [ind for ind in combined if ind.fitness is not None]
        self.individuals = self._select_survivors(combined)
