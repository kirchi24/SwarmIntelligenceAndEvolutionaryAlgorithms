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
        if selection_method not in self.VALID_SELECTION_METHODS:
            raise ValueError(
                f"Invalid selection_method '{selection_method}'. "
                f"Valid options: {self.VALID_SELECTION_METHODS}"
            )
        if fitness_fn is None:
            raise ValueError("Population requires a valid fitness function.")

        self.size = size
        self.selection_method = selection_method
        self.fitness_fn = fitness_fn

        # --- create initial random population
        self.individuals: List[CoffeeChromosome] = [
            CoffeeChromosome(fitness_fn=self.fitness_fn) for _ in range(size)
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

    def best(self) -> CoffeeChromosome:
        """
        Return the best individual in the current population.

        Returns
        -------
        CoffeeChromosome
            The individual with the highest fitness score.
        """
        # All individuals should have a fitness (evaluate() sets fitness to 0 on error)
        fitness_values = np.array([ind.fitness for ind in self.individuals])
        best_index = np.argmax(fitness_values)

        return self.individuals[best_index]

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
        n = len(self.individuals)
        fitness = np.array([ind.fitness for ind in self.individuals])
        parents: List[CoffeeChromosome] = []

        for _ in range(n):
            idx = np.random.choice(n, size=k, replace=False)
            best_idx = idx[np.argmax(fitness[idx])]
            parents.append(self.individuals[best_idx])

        return parents

    def _roulette_selection(self) -> List[CoffeeChromosome]:
        """
        Perform roulette wheel (fitness-proportionate) selection.

        Returns
        -------
        List[CoffeeChromosome]
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
            len(self.individuals),
            size=len(self.individuals),
            p=probabilities,
        )
        return [self.individuals[i] for i in indices]

    def evolve(
        self,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        mutation_float_prob: float = 0.2,
        mutation_int_prob: float = 0.2,
    ) -> None:
        """
        Evolve the population through selection, crossover, mutation, and survivor selection.

        Steps:
            1. Parent selection
            2. Offspring creation (crossover and mutation)
            3. Combine old and new individuals and select survivors
            4. Update population and evaluate fitness

        Parameters
        ----------
        crossover_rate : float
            Probability of applying crossover to each parent pair.
        mutation_rate : float
            Probability of mutating each offspring.
        mutation_float_prob : float
            Probability of mutating continuous genes (e.g., brew_time).
        mutation_int_prob : float
            Probability of mutating integer genes (e.g., roast, blend, grind).
        """

        # parent selection
        parents = self.select_parents()
        n_parents = len(parents)

        # offspring creation (crossover & mutation)
        offspring: list[CoffeeChromosome] = []
        for i in range(0, n_parents - 1, 2):
            # crossover
            if np.random.rand() < crossover_rate:
                c1, c2 = CoffeeChromosome.crossover(parents[i], parents[i + 1])
            else:
                c1, c2 = parents[i].copy(), parents[i + 1].copy()

            # mutation + evaluation
            for child in (c1, c2):
                child.fitness_fn = self.fitness_fn
                if np.random.rand() < mutation_rate:
                    child.mutate(
                        p_float=mutation_float_prob,
                        p_int=mutation_int_prob,
                    )
                child.evaluate()
                offspring.append(child)

        # handle odd number of parents
        if n_parents % 2 == 1:
            last = parents[-1].copy()
            last.fitness_fn = self.fitness_fn
            if np.random.rand() < mutation_rate:
                last.mutate(
                    p_float=mutation_float_prob,
                    p_int=mutation_int_prob,
                )
            last.evaluate()
            offspring.append(last)

        # combine old + new individuals and select survivors
        combined = self.individuals + offspring
        combined = [ind for ind in combined if ind.fitness is not None]
        combined.sort(key=lambda x: x.fitness, reverse=True)

        # update population and evaluate fitness
        self.individuals = combined[: self.size]
        for ind in self.individuals:
            ind.evaluate()
