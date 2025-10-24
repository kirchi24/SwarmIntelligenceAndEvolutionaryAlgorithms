from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Callable


class ImageChromosome:
    """
    Represents one chromosome for image-based genetic algorithms.

    Attributes
    ----------
    genes : np.ndarray
        2D array of floats in [0,1] representing pixel intensities of the chromosome.
    fitness : float
        Fitness score of the chromosome. Initialized to 0.0.
    fitness_fn : Callable[[np.ndarray], float]
        Function to compute the fitness of the chromosome based on its genes.
    """

    def __init__(
        self,
        shape: Tuple[int, int] = (16, 16),
        fitness_fn: Callable[[np.ndarray], float] = None,
    ) -> None:
        """
        Initialize an ImageChromosome with a given shape and fitness function.

        Parameters
        ----------
        shape : tuple of int, optional
            Shape of the chromosome (height, width). Default is (16, 16).
        fitness_fn : callable
            Function to compute fitness. Should accept a single argument: the genes array.

        Raises
        ------
        ValueError
            If no fitness function is provided.
        """
        self.genes = np.random.rand(*shape).astype(np.float32)
        self.fitness = 0.0
        if fitness_fn is None:
            raise ValueError("No fitness function provided for ImageChromosome.")
        self.fitness_fn = fitness_fn

    def evaluate(self) -> float:
        """
        Compute and store the fitness score of this chromosome.

        Returns
        -------
        float
            Fitness value computed by the fitness function.
        """
        self.fitness = self.fitness_fn(self.genes)
        return self.fitness

    def copy(self) -> ImageChromosome:
        """
        Create a deep copy of this chromosome, including its genes and fitness.

        Returns
        -------
        ImageChromosome
            Independent clone of the current chromosome.
        """
        new_chromosome = ImageChromosome(
            shape=self.genes.shape,
            fitness_fn=self.fitness_fn,
        )
        new_chromosome.genes = self.genes.copy()
        new_chromosome.fitness = self.fitness
        return new_chromosome
