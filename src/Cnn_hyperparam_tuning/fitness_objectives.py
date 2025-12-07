# fitness_objectives.py
"""
Define multiple objective functions for CNN optimization.
Each function should take (model, dataloader, device) and return a score.
The main fitness function can combine these objectives.
"""

from typing import Callable, List, Tuple
import torch

# Example objective: maximize accuracy

def objective_accuracy(model, dataloader, device) -> float:
    """Returns accuracy of model on dataloader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0.0

# Example objective: minimize loss

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
    return total_loss / total if total > 0 else float('inf')

# Example penalty: penalize large models

def penalty_large_model(model, *args) -> float:
    """Returns penalty proportional to model size."""
    return sum(p.numel() for p in model.parameters()) / 10000.0

# Generic fitness function

def fitness(model, dataloader, device, objectives: List[Callable], weights: List[float]=None) -> float:
    """Combines multiple objectives with weights into a single fitness score."""
    if weights is None:
        weights = [1.0] * len(objectives)
    scores = [obj(model, dataloader, device) for obj in objectives]
    return sum(w * s for w, s in zip(weights, scores))

# Example usage:
# fitness(model, test_loader, device, [objective_accuracy, penalty_large_model], [1.0, -0.1])
