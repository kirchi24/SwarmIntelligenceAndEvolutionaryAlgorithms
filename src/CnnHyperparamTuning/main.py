import matplotlib.pyplot as plt
import os
import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#from cnn import ConfigurableCNN
#from fitness_objectives import fitness, objective_f1, penalty_l2_regularization

from src.CnnHyperparamTuning.cnn import ConfigurableCNN
from src.CnnHyperparamTuning.fitness_objectives import fitness, objective_f1, penalty_l2_regularization


# --- Search space definition ---
SEARCH_SPACE = {
    "num_conv_layers": [1, 2, 3],
    "filters_per_layer": [[8, 8, 8], [16, 16, 16], [32, 32, 32]],
    "kernel_sizes": [[3, 3, 3], [5, 5, 5]],
    "pool_types": [["max", "max", "max"], ["avg", "avg", "avg"]],
    "use_dropout": [[False, False, False], [True, True, True]],
    "dropout_rates": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    "fc_neurons": [32, 64, 128],
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


# --- Differential Evolution (simplified) ---
def random_individual():
    return {
        "num_conv_layers": random.choice(
            torch.tensor(SEARCH_SPACE["num_conv_layers"])
        ).item(),
        "filters_per_layer": SEARCH_SPACE["filters_per_layer"][
            torch.randint(0, len(SEARCH_SPACE["filters_per_layer"]), (1,)).item()
        ],
        "kernel_sizes": SEARCH_SPACE["kernel_sizes"][
            torch.randint(0, len(SEARCH_SPACE["kernel_sizes"]), (1,)).item()
        ],
        "pool_types": SEARCH_SPACE["pool_types"][
            torch.randint(0, len(SEARCH_SPACE["pool_types"]), (1,)).item()
        ],
        "use_dropout": SEARCH_SPACE["use_dropout"][
            torch.randint(0, len(SEARCH_SPACE["use_dropout"]), (1,)).item()
        ],
        "dropout_rates": SEARCH_SPACE["dropout_rates"][
            torch.randint(0, len(SEARCH_SPACE["dropout_rates"]), (1,)).item()
        ],
        "fc_neurons": random.choice(torch.tensor(SEARCH_SPACE["fc_neurons"])).item(),
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


# --- Fitness evaluation ---
def evaluate_individual(params, num_epochs, train_loader, test_loader, device, quick_run = False, fitness_objectives=None, weights=None):

    # default objectives: maximize F1 score, minimize L2 regularization
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

            # Break early for quick run/testing
            if quick_run:
                break

    return fitness(
        model,
        test_loader,
        device,
        fitness_objectives,
        weights
    )


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


# --- Main optimization loop ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    train_loader, test_loader = get_data_loaders(data_dir)

    num_generations = 5  # Reduced for quick testing
    num_epochs = 3  # Reduced for quick testing
    fitness_objectives = [objective_f1, penalty_l2_regularization]
    weights = [1.0, -0.01]


    # Differential Evolution: population initialization
    population_size = 10
    population = [random_individual() for _ in range(population_size)]
    best_score = float("-inf")
    best_params = None

    for gen in range(num_generations):  # Only a few generations for demo
        print(f"Generation {gen+1}")
        for i, params in enumerate(population):
            score = evaluate_individual(params, num_epochs, train_loader, test_loader, device, True, fitness_objectives, weights)
            print(f"Individual {i+1}: Fitness = {score:.4f}")
            if score > best_score:
                best_score = score
                best_params = params
    print("Best parameters found (DE):", best_params)
    print("Best fitness (DE):", best_score)

    # --- Hill Climbing refinement ---
    def get_neighbors(params):
        neighbors = []
        # Try changing each parameter to a neighboring value in the search space
        for key in SEARCH_SPACE:
            values = SEARCH_SPACE[key]
            current = params[key]
            for v in values:
                if v != current:
                    new_params = params.copy()
                    new_params[key] = v
                    neighbors.append(new_params)
        return neighbors

    hc_steps = 10
    for step in range(hc_steps):
        improved = False
        neighbors = get_neighbors(best_params)
        for n_params in neighbors:
            score = evaluate_individual(n_params, num_epochs, train_loader, test_loader, device, True, fitness_objectives, weights)
            if score > best_score:
                best_score = score
                best_params = n_params
                improved = True
                print(f"Hill Climbing step {step+1}: Improved fitness to {best_score:.4f}")
                break  # Accept first improvement (first-ascent)
        if not improved:
            print(f"Hill Climbing step {step+1}: No improvement found, stopping.")
            break

    print("Best parameters after Hill Climbing:", best_params)
    print("Best fitness after Hill Climbing:", best_score)

    # Visualize predictions of best model
    best_model = build_model(best_params, device)
    optimizer = torch.optim.Adam(best_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    best_model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = best_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        break
    fig = visualize_predictions(best_model, test_loader, device, n_images=8)
    plt.show()


if __name__ == "__main__":
    main()
