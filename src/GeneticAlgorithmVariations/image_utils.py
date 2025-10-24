import numpy as np
from PIL import Image
import os


def load_and_resize_image(path: str, size: tuple[int, int]) -> np.ndarray:
    """
    Load an image from a file, convert it to grayscale, and resize it.

    Parameters
    ----------
    path : str
        Path to the image file.
    size : tuple of int
        Target size as (width, height).

    Returns
    -------
    np.ndarray
        Grayscale image as a 2D NumPy array with dtype uint8.
    """
    img = Image.open(path).convert("L")
    img = img.resize(size, Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def image_fitness(individual: np.ndarray, target: np.ndarray) -> float:
    """
    Compute the fitness of an individual image compared to the target image.

    The fitness is calculated as the mean absolute difference between the pixel values.

    Parameters
    ----------
    individual : np.ndarray
        The candidate image as a 2D NumPy array (dtype uint8).
    target : np.ndarray
        The target image to compare against as a 2D NumPy array (dtype uint8).

    Returns
    -------
    float
        The mean absolute difference between individual and target pixels.
    """
    diff = np.abs(individual.astype(np.int16) - target.astype(np.int16))
    return float(np.mean(diff))


if __name__ == "__main__":
    """
    Load a target image, resize it, and compute the fitness of the target
    itself (trivial case). Expects the image to exist at the specified path.
    """
    image_path = (
        f"{os.getcwd()}\\src\\GeneticAlgorithmVariations\\data\\example_image.png"
    )

    if not os.path.exists(image_path):
        print(f"Please ensure the example image exists at '{image_path}'")
        exit()

    target = load_and_resize_image(image_path, (16, 16))
