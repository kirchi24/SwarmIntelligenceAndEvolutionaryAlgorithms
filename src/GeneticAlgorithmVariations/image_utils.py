import numpy as np
from PIL import Image
import os
from typing import Callable


def load_and_resize_image(path: str, size: tuple[int, int]) -> np.ndarray:
    """
    Load an image, convert to grayscale, resize, and normalize to [0,1].

    Parameters
    ----------
    path : str
        Path to the image file.
    size : tuple[int, int]
        Target size (width, height).

    Returns
    -------
    np.ndarray
        Grayscale image as 2D array with values in [0,1].
    """
    img = Image.open(path).convert("L")
    img = img.resize(size, Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0  # normalize to [0,1]


def fitness_factory(target: np.ndarray, method: str = "manhattan") -> Callable[[np.ndarray], float]:
    """
    Create a fitness function comparing an individual to a fixed target.

    Parameters
    ----------
    target : np.ndarray
        Target image with values in [0,1].
    method : str
        Distance method: "manhattan" or "euclidean".

    Returns
    -------
    Callable[[np.ndarray], float]
        Function that takes an individual and returns fitness in [0,1].
    """
    if method not in ("manhattan", "euclidean"):
        raise ValueError(f"Invalid method '{method}'. Choose 'manhattan' or 'euclidean'.")

    def fitness(individual: np.ndarray) -> float:
        if method == "manhattan":
            diff = np.abs(individual - target)
            return 1.0 - float(np.mean(diff))  # 1.0 = perfect match
        else:  # euclidean
            diff = (individual - target) ** 2
            return 1.0 - float(np.sqrt(np.mean(diff)))  # 1.0 = perfect match

    return fitness


if __name__ == "__main__":
    # Load target image
    image_path = os.path.join(
        os.getcwd(), "src", "GeneticAlgorithmVariations", "data", "example_image.png"
    )

    if not os.path.exists(image_path):
        print(f"Please ensure the example image exists at '{image_path}'")
        exit()

    target = load_and_resize_image(image_path, (16, 16))

    # Create two fitness functions
    manhattan_fitness = fitness_factory(target, method="manhattan")
    euclidean_fitness = fitness_factory(target, method="euclidean")

    # Test
    print(f"Manhattan fitness (target vs target): {manhattan_fitness(target):.4f}")
    print(f"Euclidean fitness (target vs target): {euclidean_fitness(target):.4f}")
