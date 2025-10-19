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

        # --- evaluate immediately so each has fitness
        self.evaluate()


    def evaluate(self) -> None:
        """
        Evaluate the fitness of all individuals in the population.

        Each individual's fitness is computed using the population's fitness function.
        """
        for ind in self.individuals:
            try:
                ind.evaluate()
                if ind.fitness is None:
                    # Fallback falls coffee_fitness_4d nichts zurückgibt
                    ind.fitness = float(
                        self.fitness_fn(
                            roast=float(ind.roast),
                            blend=float(ind.blend),
                            grind=float(ind.grind),
                            brew_time=float(ind.brew_time),
                        )
                    )
            except Exception as e:
                print(f"[WARN] Evaluation failed for {ind}: {e}")
                ind.fitness = -np.inf  # schlechtester Wert, aber kein None

    def best(self) -> CoffeeChromosome:
        """
        Return the best individual in the current population.

        Returns
        -------
        CoffeeChromosome
            The individual with the highest fitness score.
        """
        valid_inds = [ind for ind in self.individuals if ind.fitness is not None]
        if not valid_inds:
            return None
        return max(valid_inds, key=lambda x: x.fitness)

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
        if np.any(np.isnan(fitnesses)):
            fitnesses = np.nan_to_num(fitnesses, nan=0.0)

        min_fit = fitnesses.min()
        if min_fit < 0:
            fitnesses -= min_fit  # shift so all fitnesses >= 0
        total = fitnesses.sum()

        if total == 0:
            probabilities = np.ones(len(fitnesses)) / len(fitnesses)
        else:
            probabilities = fitnesses / total

        indices = np.random.choice(
            len(self.individuals), size=len(self.individuals), p=probabilities
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
        parents = self.select_parents()
        n_parents = len(parents)
        new_generation: List[CoffeeChromosome] = []
        offspring: List[CoffeeChromosome] = []

        # i = 0
        # while i < n_parents - 1:
        #     if np.random.rand() < crossover_rate:
        #         child1, child2 = CoffeeChromosome.crossover(parents[i], parents[i + 1])
        #     else:
        #         child1, child2 = parents[i].copy(), parents[i + 1].copy()

        #     child1.fitness_fn = self.fitness_fn
        #     child2.fitness_fn = self.fitness_fn

        #     if np.random.rand() < mutation_rate:
        #         child1.mutate(p_float=mutation_float_prob, p_int=mutation_int_prob)
        #     if np.random.rand() < mutation_rate:
        #         child2.mutate(p_float=mutation_float_prob, p_int=mutation_int_prob)

        #     new_generation += [child1, child2]
        #     i += 2

        # if n_parents % 2 == 1:
        #     last_child = parents[-1].copy()
        #     last_child.fitness_fn = self.fitness_fn
        #     if np.random.rand() < mutation_rate:
        #         last_child.mutate(p_float=mutation_float_prob, p_int=mutation_int_prob)
        #     new_generation.append(last_child)

        # combined_population = self.individuals + new_generation

        # for ind in combined_population:
        #     if ind.fitness is None:
        #         ind.evaluate()

        # combined_sorted = sorted(
        #     combined_population, key=lambda x: x.fitness, reverse=True
        # )
        # self.individuals = combined_sorted[: len(self.individuals)]

        for i in range(0, len(parents) - 1, 2):
            if np.random.rand() < crossover_rate:
                c1, c2 = CoffeeChromosome.crossover(parents[i], parents[i + 1])
            else:
                c1, c2 = parents[i].copy(), parents[i + 1].copy()

            for child in (c1, c2):
                child.fitness_fn = self.fitness_fn
                if np.random.rand() < mutation_rate:
                    child.mutate(p_float=mutation_float_prob, p_int=mutation_int_prob)
                child.evaluate()
                offspring.append(child)

        # Survivor selection: keep the best N individuals
        combined = self.individuals + offspring
        combined = [ind for ind in combined if ind.fitness is not None]
        combined.sort(key=lambda x: x.fitness, reverse=True)
        self.individuals = combined[: self.size]

        for ind in self.individuals:
            if ind.fitness is None:
                ind.evaluate()