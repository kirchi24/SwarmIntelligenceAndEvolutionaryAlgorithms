import numpy as np
from PIL import Image
import os


def load_and_resize_image(path: str, size: tuple[int, int]) -> np.ndarray:
    """
    Load an image from a file, convert it to grayscale, resize it,
    and normalize pixel values to [0,1].

    Parameters
    ----------
    path : str
        Path to the image file.
    size : tuple of int
        Target size as (width, height).

    Returns
    -------
    np.ndarray
        Grayscale image as a 2D NumPy array with values in [0,1].
    """
    img = Image.open(path).convert("L")
    img = img.resize(size, Image.LANCZOS)
    array = np.array(img, dtype=np.float32) / 255.0  # normalize to [0,1]
    return array


def image_fitness(individual: np.ndarray, target: np.ndarray) -> float:
    """
    Compute the fitness of an individual image compared to the target image.

    Both images are expected to have values in [0,1].
    Fitness is 1.0 for a perfect match, 0.0 for maximal difference.

    Parameters
    ----------
    individual : np.ndarray
        Candidate image as a 2D NumPy array with values in [0,1].
    target : np.ndarray
        Target image as a 2D NumPy array with values in [0,1].

    Returns
    -------
    float
        Normalized fitness between 0 and 1.
    """
    diff = np.abs(individual - target)
    return 1.0 - float(np.mean(diff))  # 1.0 = perfect match


if __name__ == "__main__":
    """
    Load a target image, resize it, and compute the fitness of the target
    itself (trivial case). Expects the image to exist at the specified path.
    """
    image_path = os.path.join(
        os.getcwd(), "src", "GeneticAlgorithmVariations", "data", "example_image.png"
    )

    if not os.path.exists(image_path):
        print(f"Please ensure the example image exists at '{image_path}'")
        exit()

    target = load_and_resize_image(image_path, (16, 16))
    # trivial check: fitness of target vs itself should be 1.0
    print(f"Fitness of target vs itself: {image_fitness(target, target):.4f}")
