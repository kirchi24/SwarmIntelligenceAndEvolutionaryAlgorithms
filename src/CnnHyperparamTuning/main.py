import streamlit as st
import matplotlib.pyplot as plt
import os
import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.CnnHyperparamTuning.cnn import ConfigurableCNN
from src.CnnHyperparamTuning.fitness_objectives import (
    fitness,
    objective_f1,
    penalty_l2_regularization,
)

# --- Search space ---
SEARCH_SPACE = {
    "num_conv_layers": [1, 2, 3],
    "filters_per_layer": [[8, 8, 8], [16, 16, 16], [32, 32, 32]],
    "kernel_sizes": [[3, 3, 3], [5, 5, 5]],
    "pool_types": [["max", "max", "max"], ["avg", "avg", "avg"]],
    "use_dropout": [
        [False, False, False],
        [False, True, True],
        [False, False, True],
        [True, True, True],
    ],
    "dropout_rates": [
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3],
        [0.4, 0.4, 0.4],
        [0.5, 0.5, 0.5],
    ],
    "fc_neurons": [16, 32, 64, 128],
}


# --- Data loading ---
def get_data_loaders(data_dir, batch_size=128, little_dataset=False):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    if little_dataset:
        # Use only 10% of the data for quick tests
        import torch.utils.data as tud
        train_len = int(0.1 * len(train_dataset))
        test_len = int(0.1 * len(test_dataset))
        train_dataset, _ = tud.random_split(train_dataset, [train_len, len(train_dataset) - train_len])
        test_dataset, _ = tud.random_split(test_dataset, [test_len, len(test_dataset) - test_len])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# --- Helper functions ---
def random_individual(local_search_space=SEARCH_SPACE):
    return {
        k: random.choice(v) if isinstance(v[0], int) else random.choice(v)
        for k, v in local_search_space.items()
    }


def build_model(params, device):
    return ConfigurableCNN(
        num_conv_layers=params["num_conv_layers"],
        filters_per_layer=params["filters_per_layer"],
        kernel_sizes=params["kernel_sizes"],
        pool_types=params["pool_types"],
        use_dropout=params["use_dropout"],
        dropout_rates=params["dropout_rates"],
        fc_neurons=params["fc_neurons"],
        input_channels=1,
        num_classes=10,
    ).to(device)


def evaluate_individual(
    params,
    num_epochs,
    train_loader,
    test_loader,
    device,
    quick_run=True,
    fitness_objectives=None,
    weights=None,
):
    if fitness_objectives is None:
        fitness_objectives = [objective_f1, penalty_l2_regularization]
    if weights is None:
        weights = [1.0, -0.01]

    model = build_model(params, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    for _ in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if quick_run:
                break

    return fitness(model, test_loader, device, fitness_objectives, weights)


# --- Optimization algorithms ---
def genetic_algorithm(
    pop_size,
    generations,
    num_epochs,
    train_loader,
    test_loader,
    device,
    fitness_objectives,
    weights,
    mutation_rate=0.2,
    crossover_rate=0.8,
    elite_size=2,
    quick_run=False,
    local_search_space=SEARCH_SPACE,
    use_streamlit=False,
):
    # --- Initial population ---
    population = [random_individual(local_search_space) for _ in range(pop_size)]

    def fitness_of(ind):
        return evaluate_individual(
            ind,
            num_epochs,
            train_loader,
            test_loader,
            device,
            quick_run,
            fitness_objectives,
            weights,
        )

    # --- Evaluate initial ---
    scores = [fitness_of(ind) for ind in population]

    for gen in range(generations):
        # --- Selection (tournament) ---
        selected = []
        for _ in range(pop_size):
            i, j = random.sample(range(pop_size), 2)
            selected.append(population[i] if scores[i] > scores[j] else population[j])

        # --- Crossover ---
        offspring = []
        for i in range(0, pop_size, 2):
            p1 = selected[i]
            p2 = selected[(i + 1) % pop_size]

            if random.random() < crossover_rate:
                child1, child2 = {}, {}
                for key in local_search_space:
                    if random.random() < 0.5:
                        child1[key] = p1[key]
                        child2[key] = p2[key]
                    else:
                        child1[key] = p2[key]
                        child2[key] = p1[key]
                offspring.extend([child1, child2])
            else:
                offspring.extend([p1.copy(), p2.copy()])

        # --- Mutation ---
        for ind in offspring:
            for key in local_search_space:
                if random.random() < mutation_rate:
                    ind[key] = random.choice(local_search_space[key])

        # --- Elitism ---
        elite_indices = sorted(
            range(pop_size), key=lambda i: scores[i], reverse=True
        )[:elite_size]
        elites = [population[i] for i in elite_indices]

        population = offspring[: pop_size - elite_size] + elites
        scores = [fitness_of(ind) for ind in population]

        best_score = max(scores)
        msg = f"GA Generation {gen+1}: Best Fitness = {best_score:.4f}"
        if use_streamlit:
            st.write(msg)
        else:
            print(msg)

    best_idx = scores.index(max(scores))
    return population[best_idx], scores[best_idx]


def hill_climbing(
    best_params,
    hc_steps,
    num_epochs,
    train_loader,
    test_loader,
    device,
    fitness_objectives,
    weights,
    quick_run=False,
    local_search_space=SEARCH_SPACE,
    use_streamlit=False,
):
    best_fitness = evaluate_individual(
                best_params,
                num_epochs,
                train_loader,
                test_loader,
                device,
                quick_run,
                fitness_objectives,
                weights,)
    
    for step in range(hc_steps):
        improved = False
        neighbors = []
        for key in local_search_space:
            for val in local_search_space[key]:
                if val != best_params[key]:
                    n = best_params.copy()
                    n[key] = val
                    neighbors.append(n)

        for n_params in neighbors:
            score = evaluate_individual(
                n_params,
                num_epochs,
                train_loader,
                test_loader,
                device,
                quick_run,
                fitness_objectives,
                weights,
            )
            if score > best_fitness:
                best_params = n_params
                best_fitness = score
                improved = True
                msg = f"Hill Climbing step {step+1}: Improved fitness to {score:.4f}"
                if use_streamlit:
                    st.write(msg)
                else:
                    print(msg)
                break

        if not improved:
            msg = f"Hill Climbing step {step+1}: No improvement found, stopping."
            if use_streamlit:
                st.write(msg)
            else:
                print(msg)
            break

    return best_params


def visualize_predictions(model, dataloader, device, class_names=None, n_images=8):
    """Displays a batch of images with true and predicted labels."""
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    images = images.cpu().numpy()
    fig, axes = plt.subplots(1, n_images, figsize=(2 * n_images, 2))
    for i in range(n_images):
        ax = axes[i]
        img = images[i][0]  # MNIST is grayscale
        ax.imshow(img, cmap="gray")
        true_label = labels[i].item()
        pred_label = preds[i].item()
        if class_names:
            true_label = class_names[true_label]
            pred_label = class_names[pred_label]
        ax.set_title(f"T:{true_label}\nP:{pred_label}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    return fig


def main():
    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- Data ---
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    train_loader, test_loader = get_data_loaders(data_dir, little_dataset=True)

    # --- Parameters ---
    num_epochs = 3  # pro Individuum
    pop_size = 10  # Population für GA
    ga_gens = 5  # Generationen für GA
    hc_steps = 10  # Schritte für Hill Climbing
    mutation_rate = 0.2
    crossover_rate = 0.8
    elite_size = 2
    quick_run = True
    fitness_objectives = [objective_f1, penalty_l2_regularization]
    weights = [1.0, -0.01]

    # --- Choose method ---
    method = input("Choose optimization method (GA, GA+HC): ").strip().upper()

    if method == "GA":
        best_params, best_score = genetic_algorithm(
            pop_size=pop_size,
            generations=ga_gens,
            num_epochs=num_epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            fitness_objectives=fitness_objectives,
            weights=weights,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elite_size=elite_size,
            quick_run=quick_run,
        )
    elif method == "GA+HC":
        best_params, best_score = genetic_algorithm(
            pop_size=pop_size,
            generations=ga_gens,
            num_epochs=num_epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            fitness_objectives=fitness_objectives,
            weights=weights,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elite_size=elite_size,
            quick_run=quick_run,
        )
        print("\nStarting Hill Climbing on best GA individual...")
        best_params = hill_climbing(
            best_params=best_params,
            hc_steps=hc_steps,
            num_epochs=num_epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            fitness_objectives=fitness_objectives,
            weights=weights,
            quick_run=quick_run,
        )
        best_score = evaluate_individual(
            best_params,
            num_epochs,
            train_loader,
            test_loader,
            device,
            quick_run,
            fitness_objectives,
            weights,
        )
    else:
        print("Invalid method! Choose GA or GA+HC.")
        return

    # --- Results ---
    print("\nBest parameters found:", best_params)
    print(f"Best fitness: {best_score:.4f}")

    # --- Visualize predictions ---
    best_model = build_model(best_params, device)
    fig = visualize_predictions(best_model, test_loader, device)
    plt.show()


if __name__ == "__main__":
    main()
