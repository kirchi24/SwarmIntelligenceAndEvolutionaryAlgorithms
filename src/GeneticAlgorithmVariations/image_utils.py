# genetic_image.py

import numpy as np
from PIL import Image
import os

def load_and_resize_image(path: str, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize(size, Image.LANCZOS)
    return np.array(img, dtype=np.uint8)

def image_fitness(individual: np.ndarray, target: np.ndarray) -> float:
    diff = np.abs(individual.astype(np.int16) - target.astype(np.int16))
    return float(np.mean(diff))

if __name__ == "__main__":
    image_path = f"{os.getcwd()}\\src\\GeneticAlgorithmVariations\\data\\example_image.png"

    if not os.path.exists(image_path):
        print(f"Please ensure the example image exists at '{image_path}'")
        exit()

    target = load_and_resize_image(image_path, (16, 16))