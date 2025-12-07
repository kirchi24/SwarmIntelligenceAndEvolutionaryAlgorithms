from typing import Callable, List
import torch
from sklearn.metrics import f1_score

# ---------- Objective Functions ----------

def objective_f1(model, dataloader, device) -> float:
    """Returns macro F1 score of model on dataloader."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    if all_labels:
        return f1_score(all_labels, all_preds, average="macro")
    else:
        return 0.0


def objective_loss(model, dataloader, device) -> float:
    """Returns average cross-entropy loss of model on dataloader."""
    model.eval()
    total_loss = 0.0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total += images.size(0)
    return total_loss / total if total > 0 else float("inf")


def penalty_l2_regularization(model, *args) -> float:
    """Returns L2 regularization penalty (sum of squared weights)."""
    l2_penalty = sum(torch.sum(param ** 2).item() for param in model.parameters())
    return l2_penalty / 10000.0  # scale penalty


def penalty_large_model(model, *args) -> float:
    """Returns penalty proportional to model size (number of parameters)."""
    return sum(p.numel() for p in model.parameters()) / 10000.0


# ---------- Global Min/Max for scaling ----------

GLOBAL_OBJECTIVE_MINS = {
    'objective_f1': 0.0,
    'objective_loss': 0.0,
    'penalty_l2_regularization': 0.0,
    'penalty_large_model': 0.0,
}

GLOBAL_OBJECTIVE_MAXS = {
    'objective_f1': 1.0,     # F1 score naturally between 0-1
    'objective_loss': 5.0,   # adjust if loss can exceed this
    'penalty_l2_regularization': 5.0,
    'penalty_large_model': 5.0,
}


def fitness(
    model,
    dataloader,
    device,
    objectives: List[Callable],
    weights: List[float] = None
) -> float:
    """
    Compute combined fitness of a model given multiple objectives.
    Each objective is scaled globally using predefined min/max.
    """
    if weights is None:
        weights = [1.0] * len(objectives)

    # Compute raw scores
    scores = [obj(model, dataloader, device) for obj in objectives]

    # Scale each score to [0, 1] using global min/max
    scaled_scores = []
    for obj, score in zip(objectives, scores):
        name = obj.__name__
        mn = GLOBAL_OBJECTIVE_MINS.get(name, 0.0)
        mx = GLOBAL_OBJECTIVE_MAXS.get(name, 1.0)
        # Clip score within global min/max and scale
        score_clipped = max(min(score, mx), mn)
        scaled = (score_clipped - mn) / (mx - mn) if mx > mn else 0.0
        scaled_scores.append(scaled)

    # Combine scores with weights
    total_fitness = sum(w * s for w, s in zip(weights, scaled_scores))
    return total_fitness
