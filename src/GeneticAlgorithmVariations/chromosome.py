from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Callable

from pyparsing import Literal


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

    VALID_MUTATION_METHODS = ("uniform_local", "gaussian_adaptive")
    VALID_CROSSOVER_METHODS = ("arithmetic", "global_uniform")

    def __init__(
        self,
        shape: tuple[int, int] = (16, 16),
        fitness_fn: Callable[[np.ndarray], float] = None,
        mutation_method: Literal[
            "uniform_local", "gaussian_adaptive"
        ] = "uniform_local",
        crossover_method: Literal["arithmetic", "global_uniform"] = "arithmetic",
        mutation_rate: float = 0.01,
        mutation_width: float = 0.1,
        alpha: float = 0.5,  # for arithmetic crossover
    ) -> None:
        """
        Initialize an ImageChromosome with a given shape and fitness function.

        Parameters
        ----------
        shape : tuple[int, int]
            Shape of the image (default (16,16)).
        fitness_fn : callable
            Fitness function, must accept genes array.
        mutation_method : str
            Mutation strategy ("uniform_local" or "gaussian_adaptive").
        crossover_method : str
            Crossover strategy ("arithmetic" or "global_uniform").
        mutation_rate : float
            Probability of mutating each gene.
        mutation_width : float
            Base width for uniform mutation or base stddev for Gaussian.
        alpha : float
            Weight for arithmetic crossover (default 0.5).

        Raises
        ------
        ValueError
            If no fitness function, mutation or crossover method is provided.
        """
        self.genes = np.random.rand(*shape).astype(np.float32)
        self.fitness = 0.0
        self.fitness_fn = fitness_fn
        self.mutation_method = mutation_method
        self.crossover_method = crossover_method
        self.mutation_rate = mutation_rate
        self.mutation_width = mutation_width
        self.alpha = alpha

        if fitness_fn is None:
            raise ValueError("A fitness function must be provided.")
        if mutation_method not in self.VALID_MUTATION_METHODS:
            raise ValueError(f"Invalid mutation method: {mutation_method}")
        if crossover_method not in self.VALID_CROSSOVER_METHODS:
            raise ValueError(f"Invalid crossover method: {crossover_method}")

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

    # -------------------------
    # Mutation Methods
    # -------------------------

    def mutate(self, max_fitness: float = 1.0) -> None:
        """
        Apply selected mutation method to the chromosome.

        Parameters
        ----------
        max_fitness : float
            Maximum possible fitness, used for adaptive Gaussian mutation.
        """
        if self.mutation_method == "uniform_local":
            self._mutate_uniform_local()
        elif self.mutation_method == "gaussian_adaptive":
            self._mutate_gaussian_adaptive(max_fitness)

    def _mutate_uniform_local(self) -> None:
        """
        Uniform local mutation around current gene values.
        """
        mask = np.random.rand(*self.genes.shape) < self.mutation_rate
        perturb = np.random.uniform(
            -self.mutation_width / 2, self.mutation_width / 2, self.genes.shape
        )
        self.genes[mask] += perturb[mask]
        self.genes = np.clip(self.genes, 0.0, 1.0)

    def _mutate_gaussian_adaptive(self, max_fitness: float) -> None:
        """
        Gaussian mutation centered at current gene values, scaled by fitness.

        Parameters
        ----------
        max_fitness : float
            Maximum possible fitness, used to scale mutation size.
        """
        mask = np.random.rand(*self.genes.shape) < self.mutation_rate
        factor = 1.0 - (self.fitness / max_fitness)
        stddev = self.mutation_width * factor
        perturb = np.random.normal(0.0, stddev, self.genes.shape)
        self.genes[mask] += perturb[mask]
        self.genes = np.clip(self.genes, 0.0, 1.0)

    # -------------------------
    # Crossover Methods
    # -------------------------

    def crossover(self, other: ImageChromosome) -> ImageChromosome:
        """
        Perform crossover with another chromosome using the selected method.

        Parameters
        ----------
        other : ImageChromosome
            The other parent chromosome.

        Returns
        -------
        ImageChromosome
            Child chromosome resulting from crossover.

        Raises
        ------
        ValueError
            If the selected crossover method is not recognized.
        """
        if self.crossover_method == "arithmetic":
            child_genes = self._crossover_arithmetic(other)
        elif self.crossover_method == "global_uniform":
            child_genes = self._crossover_global_uniform(other)
        else:
            raise ValueError(f"Invalid crossover method: {self.crossover_method}")

        child = ImageChromosome(
            shape=self.genes.shape,
            fitness_fn=self.fitness_fn,
            mutation_method=self.mutation_method,
            crossover_method=self.crossover_method,
            mutation_rate=self.mutation_rate,
            mutation_width=self.mutation_width,
            alpha=self.alpha,
        )
        child.genes = child_genes
        return child

    def _crossover_arithmetic(self, other: ImageChromosome) -> np.ndarray:
        """
        Arithmetic crossover: weighted average of parent genes.

        Parameters
        ----------
        other : ImageChromosome
            The second parent chromosome.

        Returns
        -------
        np.ndarray
            Child gene array after arithmetic crossover.
        """
        return np.clip(
            self.alpha * self.genes + (1 - self.alpha) * other.genes, 0.0, 1.0
        )

    def _crossover_global_uniform(self, other: ImageChromosome) -> np.ndarray:
        """
        Global uniform crossover: each gene randomly chosen from one of the parents.

        Parameters
        ----------
        other : ImageChromosome
            The second parent chromosome.

        Returns
        -------
        np.ndarray
            Child gene array after global uniform crossover.
        """
        mask = np.random.rand(*self.genes.shape) < 0.5
        return np.where(mask, self.genes, other.genes)

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
