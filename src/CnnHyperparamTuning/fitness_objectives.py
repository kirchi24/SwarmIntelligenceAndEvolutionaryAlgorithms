"""
Define multiple objective functions for CNN optimization.
Each function should take (model, dataloader, device) and return a score.
The main fitness function can combine these objectives.
"""

from typing import Callable, List
import torch
from sklearn.metrics import f1_score


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


def penalty_l2_regularization(model, *args) -> float:
    """Returns L2 regularization penalty (sum of squared weights)."""
    l2_penalty = 0.0
    for param in model.parameters():
        l2_penalty += torch.sum(param**2).item()
    return l2_penalty / 10000.0  # scale penalty


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


def penalty_large_model(model, *args) -> float:
    """Returns penalty proportional to model size."""
    return sum(p.numel() for p in model.parameters()) / 10000.0


def fitness(
    model, dataloader, device, objectives: List[Callable], weights: List[float] = None
) -> float:
    """Combines multiple objectives with weights into a single fitness score."""
    if weights is None:
        weights = [1.0] * len(objectives)
    scores = [obj(model, dataloader, device) for obj in objectives]
    return sum(w * s for w, s in zip(weights, scores))


# Example usage:
# fitness(model, test_loader, device, [objective_accuracy, penalty_large_model], [1.0, -0.1])
