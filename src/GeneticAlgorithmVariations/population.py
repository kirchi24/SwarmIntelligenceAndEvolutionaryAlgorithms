import numpy as np
from typing import List, Callable
from src.GeneticAlgorithmVariations.chromosome import ImageChromosome


class Population:
    """
    Population of ImageChromosome individuals for a genetic algorithm.

    Handles evaluation, selection, crossover, mutation, and survivor selection.
    """

    VALID_SELECTION_METHODS = ("tournament", "roulette")

    def __init__(
        self,
        size: int = 30,
        selection_method: str = "tournament",
        fitness_fn: Callable[[np.ndarray], float] = None,
        mutation_method: str = "uniform_local",
        crossover_method: str = "arithmetic",
        alpha: float = 0.5,  # for arithmetic crossover
    ) -> None:
        """
        Initialize a population of ImageChromosome individuals with configurable GA parameters.

        Parameters
        ----------
        size : int
            Number of individuals in the population.
        selection_method : str
            Selection method: "tournament" or "roulette".
        fitness_fn : callable
            Fitness function accepting an np.ndarray of genes.
        mutation_method : str
            Mutation method for all chromosomes ("uniform_local" or "gaussian_adaptive").
        crossover_method : str
            Crossover method for all chromosomes ("arithmetic" or "global_uniform").
        alpha : float
            Weight for arithmetic crossover (default 0.5).

        Raises
        ------
        ValueError
            If selection_method, mutation_method, or crossover_method is invalid, or if fitness_fn is None.
        """
        self.size = size
        self.selection_method = selection_method
        self.fitness_fn = fitness_fn
        self.mutation_method = mutation_method
        self.crossover_method = crossover_method
        self.alpha = alpha

        if fitness_fn is None:
            raise ValueError("A fitness function must be provided.")
        if selection_method not in self.VALID_SELECTION_METHODS:
            raise ValueError(f"Invalid selection method: {selection_method}")
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

    def select_parents(self, k: int = 3) -> List[ImageChromosome]:
        """
        Select parent individuals according to the population's selection method.

        Parameters
        ----------
        k : int, optional
            Tournament size if using tournament selection (default 3).

        Returns
        -------
        List[ImageChromosome]
            Selected parents for crossover.
        """
        if self.selection_method == "tournament":
            return self._tournament_selection(k)
        elif self.selection_method == "roulette":
            return self._roulette_selection()

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

    def _roulette_selection(self) -> List[ImageChromosome]:
        """
        Perform roulette wheel (fitness-proportionate) selection.

        Returns
        -------
        List[ImageChromosome]
            List of selected parent chromosomes.
        """
        fitnesses = np.array([ind.fitness for ind in self.individuals], dtype=float)
        fitnesses = np.nan_to_num(fitnesses, nan=0.0)

        # shift fitnesses if there are negative values
        min_fit = fitnesses.min()
        if min_fit < 0:
            fitnesses -= min_fit
        total = fitnesses.sum()
        if total <= 0:
            # if all fitnesses are zero, select uniformly
            probabilities = np.full_like(fitnesses, 1 / len(fitnesses))
        else:
            probabilities = fitnesses / total
        indices = np.random.choice(
            len(self.individuals), size=len(self.individuals), p=probabilities
        )
        return [self.individuals[i] for i in indices]

    def evolve(self) -> None:
        """
        Perform one generation of evolution: selection, crossover, mutation, and survivor selection.
        """
        parents = self.select_parents()
        n_parents = len(parents)
        offspring: List[ImageChromosome] = []

        max_fitness = max(ind.fitness for ind in self.individuals)

        for i in range(0, n_parents - 1, 2):
            # crossover
            c1 = parents[i].crossover(parents[i + 1])
            c2 = parents[i + 1].crossover(parents[i])

            # mutation + evaluation
            for child in (c1, c2):
                child.mutate(max_fitness)
                child.evaluate()
                offspring.append(child)

        # handle odd parent
        if n_parents % 2 == 1:
            last = parents[-1].copy()
            last.mutate(max_fitness)
            last.evaluate()
            offspring.append(last)

        # combine and select top individuals
        combined = self.individuals + offspring
        combined = [ind for ind in combined if ind.fitness is not None]
        combined.sort(key=lambda x: x.fitness, reverse=True)
        self.individuals = combined[: self.size]
