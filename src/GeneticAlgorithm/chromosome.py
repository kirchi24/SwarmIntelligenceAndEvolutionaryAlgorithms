from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Callable


class CoffeeChromosome:
    """
    Represents one coffee configuration (CoffeeChromosome) for mixed-variable optimization.

    Attributes
    ----------
    roast : int
        Coffee roast level in [0, 20].
    blend : int
        Blend ratio in [0, 100].
    grind : int
        Grind coarseness in [0, 10].
    brew_time : float
        Brew time in minutes, in [0.0, 5.0].
    fitness : float or None
        Fitness score (quality) of this configuration. None if unevaluated.
    """

    INT_RANGES: dict[str, Tuple[int, int]] = {
        "roast": (0, 20),
        "blend": (0, 100),
        "grind": (0, 10),
    }
    FLOAT_RANGES: dict[str, Tuple[float, float]] = {"brew_time": (0.0, 5.0)}

    def __init__(
        self,
        fitness_fn: Optional[Callable[[int, int, int, float], float]],
        roast: Optional[int] = None,
        blend: Optional[int] = None,
        grind: Optional[int] = None,
        brew_time: Optional[float] = None,
    ) -> None:
        """
        Initialize a CoffeeChromosome with given or random parameters.

        Parameters
        ----------
        fitness_fn : callable
            Function to compute fitness. Should accept (roast, blend, grind, brew_time).
        roast : int, optional
            Roast level (default random integer in [0, 20]).
        blend : int, optional
            Blend ratio (default random integer in [0, 100]).
        grind : int, optional
            Grind coarseness (default random integer in [0, 10]).
        brew_time : float, optional
            Brew time in minutes (default random float in [0.0, 5.0]).
        """
        self.roast: int = np.random.randint(0, 21) if roast is None else roast
        self.blend: int = np.random.randint(0, 101) if blend is None else blend
        self.grind: int = np.random.randint(0, 11) if grind is None else grind
        self.brew_time: float = (
            np.random.uniform(0.0, 5.0) if brew_time is None else brew_time
        )
        self.fitness: Optional[float] = None
        if self.fitness_fn is None:
            raise ValueError("No fitness function provided for CoffeeChromosome.")
        self.fitness_fn = fitness_fn

    def evaluate(self) -> float:
        """
        Compute and store the coffee fitness score for this chromosome.

        Returns
        -------
        float
            Fitness (quality) value between 0 and 100.
        """
        self.fitness = self.fitness_fn(
            self.roast, self.blend, self.grind, self.brew_time
        )
        return self.fitness

    def mutate(self, p_int: float = 0.3, p_float: float = 0.3) -> None:
        if np.random.rand() < p_int:
            self.roast = int(np.clip(self.roast + np.random.choice([-1, 1]), 0, 20))
        if np.random.rand() < p_int:
            self.blend = int(
                np.clip(self.blend + np.random.choice([-1, 1, -2, 2]), 0, 100)
            )
        if np.random.rand() < p_int:
            self.grind = int(np.clip(self.grind + np.random.choice([-1, 1]), 0, 10))
        if np.random.rand() < p_float:
            self.brew_time = float(
                np.clip(self.brew_time + np.random.normal(0, 0.1), 0.0, 5.0)
            )

    @staticmethod
    def crossover(
        parent1: CoffeeChromosome, parent2: CoffeeChromosome
    ) -> Tuple[CoffeeChromosome, CoffeeChromosome]:

        child1 = CoffeeChromosome(fitness_fn=parent1.fitness_fn)
        child2 = CoffeeChromosome(fitness_fn=parent2.fitness_fn)

        # Integer genes
        child1.roast = np.random.choice([parent1.roast, parent2.roast])
        child2.roast = np.random.choice([parent1.roast, parent2.roast])
        child1.blend = np.random.choice([parent1.blend, parent2.blend])
        child2.blend = np.random.choice([parent1.blend, parent2.blend])
        child1.grind = np.random.choice([parent1.grind, parent2.grind])
        child2.grind = np.random.choice([parent1.grind, parent2.grind])

        # Continuous gene (blend crossover)
        alpha = np.random.rand()
        child1.brew_time = alpha * parent1.brew_time + (1 - alpha) * parent2.brew_time
        child2.brew_time = (1 - alpha) * parent1.brew_time + alpha * parent2.brew_time

        return child1, child2

    def copy(self) -> CoffeeChromosome:
        return CoffeeChromosome(
            self.roast, self.blend, self.grind, self.brew_time, self.fitness_fn
        )

    def __repr__(self) -> str:
        base = (
            f"CoffeeChromosome(roast={self.roast}, blend={self.blend}, grind={self.grind}, "
            f"brew_time={self.brew_time:.2f}"
        )
        if self.fitness is not None:
            return base + f", fitness={self.fitness:.2f})"
        return base + ")"
