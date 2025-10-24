from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Callable


class ImageChromosome:
    """
    Represents one coffee configuration (ImageChromosome) for mixed-variable optimization.

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

    def __init__(
        self,
        shape: Tuple[int, int] = (16, 16),
        fitness_fn: Callable[[np.ndarray], float] = None,
    ) -> None:
        """
        Initialize a ImageChromosome with given or random parameters.

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
        # genes as numpy array rather than individual class in order to increase speed
        self.genes = np.random.rand(*shape).astype(np.float32)
        self.fitness = 0.0
        if fitness_fn is None:
            raise ValueError("No fitness function provided for ImageChromosome.")
        self.fitness_fn = fitness_fn

    def evaluate(self) -> float:
        """
        Compute and store the coffee fitness score for this chromosome.

        Returns
        -------
        float
            Fitness (quality) value between 0 and 100.
        """
        self.fitness = self.fitness_fn(genes=self.genes)
        return self.fitness


    def copy(self) -> ImageChromosome:
        """
        Create an exact copy of this chromosome.

        Returns
        -------
        ImageChromosome
            Independent clone of the current chromosome.
        """
        new_chromosome = ImageChromosome(
            shape=self.genes.shape,
            fitness_fn=self.fitness_fn,
        )
        new_chromosome.fitness = self.fitness  # copy fitness to avoid re-evaluation
        return new_chromosome
