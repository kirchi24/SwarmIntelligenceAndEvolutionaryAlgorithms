import streamlit as st
import matplotlib.pyplot as plt
import os
import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.CnnHyperparamTuning.cnn import ConfigurableCNN
from src.CnnHyperparamTuning.fitness_objectives import fitness, objective_f1, penalty_l2_regularization

# --- Search space ---
SEARCH_SPACE = {
    "num_conv_layers": [1, 2, 3],
    "filters_per_layer": [[8, 8, 8], [16, 16, 16], [32, 32, 32]],
    "kernel_sizes": [[3, 3, 3], [5, 5, 5]],
    "pool_types": [["max", "max", "max"], ["avg", "avg", "avg"]],
    "use_dropout": [[False, False, False], [False, True, True], [False, False, True], [True, True, True]],
    "dropout_rates": [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4], [0.5, 0.5, 0.5]],
    "fc_neurons": [16, 32, 64, 128],
}


# --- Data loading ---
def get_data_loaders(data_dir, batch_size=128):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# --- Helper functions ---
def random_individual():
    return {
        k: random.choice(v) if isinstance(v[0], int) else random.choice(v)
        for k, v in SEARCH_SPACE.items()
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
def differential_evolution(
    pop_size,
    de_gens,
    num_epochs,
    train_loader,
    test_loader,
    device,
    fitness_objectives,
    weights,
    quick_run=False,
):
    F, CR = 0.8, 0.7
    population = [random_individual() for _ in range(pop_size)]
    best_score = float("-inf")
    best_params = None

    for gen in range(de_gens):
        new_population = []
        for i, target in enumerate(population):
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)
            x_a, x_b, x_c = population[a], population[b], population[c]

            mutant = {}
            for key in SEARCH_SPACE:
                if isinstance(SEARCH_SPACE[key][0], int):
                    val = x_a[key] + F * (x_b[key] - x_c[key])
                    mutant[key] = min(SEARCH_SPACE[key], key=lambda x: abs(x - val))
                else:
                    mutant[key] = random.choice([x_a[key], x_b[key], x_c[key]])

            trial = {
                k: (mutant[k] if random.random() < CR else target[k])
                for k in SEARCH_SPACE
            }
            f_trial = evaluate_individual(
                trial,
                num_epochs,
                train_loader,
                test_loader,
                device,
                quick_run,
                fitness_objectives,
                weights,
            )
            f_target = evaluate_individual(
                target,
                num_epochs,
                train_loader,
                test_loader,
                device,
                quick_run,
                fitness_objectives,
                weights,
            )

            if f_trial > f_target:
                new_population.append(trial)
                if f_trial > best_score:
                    best_score, best_params = f_trial, trial
            else:
                new_population.append(target)
                if f_target > best_score:
                    best_score, best_params = f_target, target

        population = new_population
        st.write(f"Generation {gen+1}: Best fitness so far: {best_score:.4f}")

    return best_params, best_score


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
):
    for step in range(hc_steps):
        improved = False
        neighbors = []
        for key in SEARCH_SPACE:
            for val in SEARCH_SPACE[key]:
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
            if score > fitness(
                best_params, test_loader, device, fitness_objectives, weights
            ):
                best_params = n_params
                improved = True
                st.write(
                    f"Hill Climbing step {step+1}: Improved fitness to {score:.4f}"
                )
                break

        if not improved:
            st.write(f"Hill Climbing step {step+1}: No improvement found, stopping.")
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
    train_loader, test_loader = get_data_loaders(data_dir)

    # --- Parameters ---
    num_epochs = 3  # pro Individuum
    pop_size = 10  # Population für DE
    de_gens = 5  # Generationen für DE
    hc_steps = 10  # Schritte für Hill Climbing
    fitness_objectives = [objective_f1, penalty_l2_regularization]
    weights = [1.0, -0.01]

    # --- Choose method ---
    method = input("Choose optimization method (DE, HC, DE+HC): ").strip().upper()

    if method == "DE":
        best_params, best_score = differential_evolution(
            pop_size,
            de_gens,
            num_epochs,
            train_loader,
            test_loader,
            device,
            fitness_objectives,
            weights,
            True,
        )
    elif method == "HC":
        best_params = random_individual()
        best_params = hill_climbing(
            best_params,
            hc_steps,
            num_epochs,
            train_loader,
            test_loader,
            device,
            fitness_objectives,
            weights,
            True,
        )
        best_score = evaluate_individual(
            best_params,
            num_epochs,
            train_loader,
            test_loader,
            device,
            True,
            fitness_objectives,
            weights,
        )
    elif method == "DE+HC":
        best_params, _ = differential_evolution(
            pop_size,
            de_gens,
            num_epochs,
            train_loader,
            test_loader,
            device,
            fitness_objectives,
            weights,
            True,
        )
        best_params = hill_climbing(
            best_params,
            hc_steps,
            num_epochs,
            train_loader,
            test_loader,
            device,
            fitness_objectives,
            weights,
            True,
        )
        best_score = evaluate_individual(
            best_params,
            num_epochs,
            train_loader,
            test_loader,
            device,
            True,
            fitness_objectives,
            weights,
        )
    else:
        print("Invalid method! Choose DE, HC, or DE+HC.")
        return

    # --- Results ---
    print("Best parameters found:", best_params)
    print(f"Best fitness: {best_score:.4f}")

    # --- Visualize predictions ---
    best_model = build_model(best_params, device)
    best_model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = best_model(images)
        _, preds = torch.max(outputs, 1)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 8, figsize=(16, 2))
    for i in range(8):
        ax = axes[i]
        ax.imshow(images[i].cpu()[0], cmap="gray")
        ax.set_title(f"T:{labels[i].item()}\nP:{preds[i].item()}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
