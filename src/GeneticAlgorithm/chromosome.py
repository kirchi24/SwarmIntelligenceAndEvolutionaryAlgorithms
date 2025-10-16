import numpy as np



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

    INT_RANGES = {"roast": (0, 20), "blend": (0, 100), "grind": (0, 10)}
    FLOAT_RANGES = {"brew_time": (0.0, 5.0)}

    def __init__(self, roast=None, blend=None, grind=None, brew_time=None):
        """
        Initialize a CoffeeChromosome with given or random parameters.

        Parameters
        ----------
        roast : int, optional
            Roast level (default random integer in [0, 20]).
        blend : int, optional
            Blend ratio (default random integer in [0, 100]).
        grind : int, optional
            Grind coarseness (default random integer in [0, 10]).
        brew_time : float, optional
            Brew time in minutes (default random float in [0.0, 5.0]).
        """
        self.roast = np.random.randint(0, 21) if roast is None else roast
        self.blend = np.random.randint(0, 101) if blend is None else blend
        self.grind = np.random.randint(0, 11) if grind is None else grind
        self.brew_time = np.random.uniform(0.0, 5.0) if brew_time is None else brew_time
        self.fitness = None

   

    def copy(self):
        """
        Create an exact copy of this chromosome.

        Returns
        -------
        CoffeeChromosome
            Independent clone of the current chromosome.
        """
        return CoffeeChromosome(self.roast, self.blend, self.grind, self.brew_time)

    def __repr__(self):
        """
        Developer-friendly string representation of the chromosome.

        Returns
        -------
        str
            Text summary including parameters and (if evaluated) fitness value.
        """
        return (
            f"Chromosome(roast={self.roast}, blend={self.blend}, grind={self.grind}, "
            f"brew_time={self.brew_time:.2f}, fitness={self.fitness:.2f})"
            if self.fitness is not None
            else f"Chromosome(roast={self.roast}, blend={self.blend}, grind={self.grind}, "
            f"brew_time={self.brew_time:.2f})"
        )
