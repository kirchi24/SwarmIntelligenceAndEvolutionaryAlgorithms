from __future__ import annotations
import numpy as np
from typing import Callable


from pyparsing import Literal


class ImageChromosome:
    """
    Represents one chromosome for image-based genetic algorithms.
    """

    VALID_MUTATION_METHODS = ("uniform_local", "gaussian_adaptive")
    VALID_CROSSOVER_METHODS = ("arithmetic", "global_uniform")
    VALID_INITIALIZATION_METHODS = ("random", "expert_knowledge")

    def __init__(
        self,
        shape: tuple[int, int] = (16, 16),
        fitness_fn: Callable[[np.ndarray], float] = None,
        initialization_method: Literal["random", "expert_knowledge"] = "random",
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
        initialization_method : str
            Method for initializing genes ("random" or "expert_knowledge").
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
        if fitness_fn is None:
            raise ValueError("A fitness function must be provided.")
        if initialization_method not in self.VALID_INITIALIZATION_METHODS:
            raise ValueError(f"Invalid initialization method: {initialization_method}")
        if mutation_method not in self.VALID_MUTATION_METHODS:
            raise ValueError(f"Invalid mutation method: {mutation_method}")
        if crossover_method not in self.VALID_CROSSOVER_METHODS:
            raise ValueError(f"Invalid crossover method: {crossover_method}")

        # --- Initialize genes ---
        if initialization_method == "random":
            genes = np.random.rand(*shape).astype(np.float32)

        elif initialization_method == "expert_knowledge":
            # Assume mostly white background with darker center
            genes = np.ones(shape, dtype=np.float32)
            yy, xx = np.meshgrid(
                np.linspace(-1, 1, shape[0]),
                np.linspace(-1, 1, shape[1]),
                indexing="ij",
            )
            radius = np.sqrt(xx**2 + yy**2)
            center_mask = np.exp(-(radius**2) / (2 * (0.4**2)))  # Gaussian falloff
            noise = np.random.rand(*shape).astype(np.float32) * 0.3
            genes -= 0.6 * center_mask
            genes += noise
            genes = np.clip(genes, 0.0, 1.0)

        self.genes = genes
        self.fitness = 0.0
        self.fitness_fn = fitness_fn
        self.mutation_method = mutation_method
        self.crossover_method = crossover_method
        self.mutation_rate = mutation_rate
        self.mutation_width = mutation_width
        self.alpha = alpha
        self.age = 0  # age in generations

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

    def crossover(
        self, other: ImageChromosome
    ) -> tuple[ImageChromosome, ImageChromosome]:
        """
        Perform crossover with another chromosome and produce two offspring.

        Parameters
        ----------
        other : ImageChromosome
            The other parent chromosome.

        Returns
        -------
        tuple[ImageChromosome, ImageChromosome]
            Two child chromosomes resulting from crossover.

        Raises
        ------
        ValueError
            If the selected crossover method is not recognized.
        """
        if self.crossover_method == "arithmetic":
            child1_genes, child2_genes = self._crossover_arithmetic(other)
        elif self.crossover_method == "global_uniform":
            child1_genes, child2_genes = self._crossover_global_uniform(other)
        else:
            raise ValueError(f"Invalid crossover method: {self.crossover_method}")

        child1 = ImageChromosome(
            shape=self.genes.shape,
            fitness_fn=self.fitness_fn,
            mutation_method=self.mutation_method,
            crossover_method=self.crossover_method,
            mutation_rate=self.mutation_rate,
            mutation_width=self.mutation_width,
            alpha=self.alpha,
        )
        child2 = ImageChromosome(
            shape=self.genes.shape,
            fitness_fn=self.fitness_fn,
            mutation_method=self.mutation_method,
            crossover_method=self.crossover_method,
            mutation_rate=self.mutation_rate,
            mutation_width=self.mutation_width,
            alpha=self.alpha,
        )

        child1.genes = child1_genes
        child2.genes = child2_genes
        return child1, child2

    def _crossover_arithmetic(
        self, other: ImageChromosome
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Arithmetic crossover: two offspring produced by weighted averaging of parent genes.

        Parameters
        ----------
        other : ImageChromosome
            The second parent chromosome.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Two child gene arrays after arithmetic crossover.
        """
        child1 = np.clip(
            self.alpha * self.genes + (1 - self.alpha) * other.genes, 0.0, 1.0
        )
        child2 = np.clip(
            self.alpha * other.genes + (1 - self.alpha) * self.genes, 0.0, 1.0
        )
        return child1, child2

    def _crossover_global_uniform(
        self, other: ImageChromosome
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Global uniform crossover: each gene randomly chosen from one of the parents.

        Parameters
        ----------
        other : ImageChromosome
            The second parent chromosome.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Two child gene arrays after global uniform crossover.
        """
        mask = np.random.rand(*self.genes.shape) < 0.5
        child1 = np.where(mask, self.genes, other.genes)
        child2 = np.where(mask, other.genes, self.genes)
        return child1, child2

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
