import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfigurableCNN(nn.Module):
    def __init__(
        self,
        num_conv_layers=2,
        filters_per_layer=[16, 32, 32],
        kernel_sizes=[3, 3, 3],
        pool_types=["max", "max", "max"],
        use_dropout=[False, False, False],
        dropout_rates=[0.0, 0.0, 0.0],
        fc_neurons=64,
        input_channels=1,
        num_classes=10,
    ):
        super(ConfigurableCNN, self).__init__()
        assert 1 <= num_conv_layers <= 3, "num_conv_layers must be 1, 2, or 3"
        assert all(
            f <= 32 for f in filters_per_layer[:num_conv_layers]
        ), "filters per layer max 32"
        assert all(
            k in [3, 5] for k in kernel_sizes[:num_conv_layers]
        ), "kernel size must be 3 or 5"
        assert all(
            p in ["max", "avg"] for p in pool_types[:num_conv_layers]
        ), "pool type must be 'max' or 'avg'"
        assert all(
            0.0 <= d <= 0.5 for d in dropout_rates[:num_conv_layers]
        ), "dropout rate max 0.5"
        assert fc_neurons <= 128, "fc_neurons max 128"

        self.convs = nn.ModuleList()
        self.pools = []
        self.dropouts = nn.ModuleList()
        in_channels = input_channels
        for i in range(num_conv_layers):
            self.convs.append(
                nn.Conv2d(
                    in_channels,
                    filters_per_layer[i],
                    kernel_size=kernel_sizes[i],
                    padding=kernel_sizes[i] // 2,
                )
            )
            if pool_types[i] == "max":
                self.pools.append(nn.MaxPool2d(2))
            else:
                self.pools.append(nn.AvgPool2d(2))
            if use_dropout[i]:
                self.dropouts.append(nn.Dropout(dropout_rates[i]))
            else:
                self.dropouts.append(nn.Identity())
            in_channels = filters_per_layer[i]

        # Calculate the output size after convolutions and pooling
        dummy = torch.zeros(1, input_channels, 28, 28)
        with torch.no_grad():
            x = dummy
            for i in range(num_conv_layers):
                x = self.convs[i](x)
                x = F.relu(x)
                x = self.pools[i](x)
                x = self.dropouts[i](x)
            flat_features = x.view(1, -1).shape[1]

        self.fc = nn.Linear(flat_features, fc_neurons)
        self.out = nn.Linear(fc_neurons, num_classes)

    def forward(self, x):
        for conv, pool, dropout in zip(self.convs, self.pools, self.dropouts):
            x = conv(x)
            x = F.relu(x)
            x = pool(x)
            x = dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = self.out(x)
        return x


def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on the given dataloader.
    Returns accuracy and average loss.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / total if total > 0 else 0
    return accuracy, avg_loss


if __name__ == "__main__":
    import os
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data folder
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    # Transforms
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Load FashionMNIST
    train_dataset = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Use a small subsample for quick test
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Smallest network: 1 conv layer, 8 filters, 3x3 kernel, max pool, no dropout, 32 fc neurons
    model = ConfigurableCNN(
        num_conv_layers=1,
        filters_per_layer=[8, 8, 8],
        kernel_sizes=[3, 3, 3],
        pool_types=["max", "max", "max"],
        use_dropout=[False, False, False],
        dropout_rates=[0.0, 0.0, 0.0],
        fc_neurons=32,
        input_channels=1,
        num_classes=10,
    ).to(device)

    # Simple training loop (1 epoch for quick test)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        break  # Only one batch for quick test

    # Evaluate
    acc, avg_loss = evaluate_model(model, test_loader, device)
    print(f"Test accuracy: {acc:.4f}, Test loss: {avg_loss:.4f}")
