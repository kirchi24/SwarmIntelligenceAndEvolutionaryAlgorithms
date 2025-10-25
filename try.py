from matplotlib import pyplot as plt
from src.GeneticAlgorithmVariations.image_utils import load_and_resize_image, fitness_factory
from src.GeneticAlgorithmVariations.population import Population
import numpy as np
import os


def make_image_fitness(target: np.ndarray):
    """
    Returns a fitness function that compares genes to the target image.
    """

    def fitness_fn(genes: np.ndarray) -> float:
        diff = np.abs(genes - target)
        return 1.0 - float(np.mean(diff))  # 1.0 = perfect match

    return fitness_fn


# -----------------------------
# Main
# -----------------------------
def main():
    image_path = os.path.join(
        os.getcwd(), "src", "GeneticAlgorithmVariations", "data", "example_image.png"
    )
    if not os.path.exists(image_path):
        print(f"Please ensure image exists at '{image_path}'")
        return

    # Load target image
    target = load_and_resize_image(image_path, (16, 16))
    fitness_fn = fitness_factory(target, method="euclidean")

    # Initialize population
    pop = Population(
        size=50,
        shape=(16, 16),
        fitness_fn=fitness_fn,
        initialization_method="random",
        parent_selection="rank",
        survivor_method="fitness",
        mutation_method="uniform_local",
        mutation_rate=0.5,
        mutation_width=0.1,
        crossover_method="arithmetic",
        alpha=0.5,
    )

    best_fitness_history = []

    # -----------------------------
    # Select stopping criterion
    # -----------------------------
    stop_criterion = "generations"  # "generations" or "fitness"

    if stop_criterion == "generations":
        max_generations = 3000
        for gen in range(max_generations):
            pop.evolve()
            best_fit = pop.best().fitness
            best_fitness_history.append(best_fit)
            print(f"Generation {gen+1}: Best fitness = {best_fit:.6f}")

    elif stop_criterion == "fitness":
        epsilon = 1e-6        # minimum improvement to continue
        patience = 10         # number of generations to wait
        gen = 0
        no_improve_count = 0
        last_best = -np.inf

        while True:
            gen += 1
            pop.evolve()
            best_fit = pop.best().fitness
            best_fitness_history.append(best_fit)
            print(f"Generation {gen}: Best fitness = {best_fit:.6f}")

            if best_fit - last_best < epsilon:
                no_improve_count += 1
            else:
                no_improve_count = 0  # reset if improvement
            last_best = best_fit

            if no_improve_count >= patience:
                print(f"Stopping: no significant improvement in last {patience} generations")
                break

    # -----------------------------
    # Plot fitness
    # -----------------------------
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(best_fitness_history) + 1), best_fitness_history, linestyle="-"
    )
    plt.title("Best Fitness per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
