import numpy as np



class CoffeeChromosome:
    """Mixed-variable chromosome for the coffee optimization problem."""

    INT_RANGES = {"roast": (0, 20), "blend": (0, 100), "grind": (0, 10)}
    FLOAT_RANGES = {"brew_time": (0.0, 5.0)}

    def __init__(self, roast=None, blend=None, grind=None, brew_time=None):
        # Initialize randomly if no values are given
        self.roast = np.random.randint(0, 21) if roast is None else roast
        self.blend = np.random.randint(0, 101) if blend is None else blend
        self.grind = np.random.randint(0, 11) if grind is None else grind
        self.brew_time = np.random.uniform(0.0, 5.0) if brew_time is None else brew_time

        self.fitness = None

   

    def copy(self):
        return CoffeeChromosome(self.roast, self.blend, self.grind, self.brew_time)

    def __repr__(self):
        return (
            f"Chromosome(roast={self.roast}, blend={self.blend}, grind={self.grind}, "
            f"brew_time={self.brew_time:.2f}, fitness={self.fitness:.2f})"
            if self.fitness is not None
            else f"Chromosome(roast={self.roast}, blend={self.blend}, grind={self.grind}, "
            f"brew_time={self.brew_time:.2f})"
        )
