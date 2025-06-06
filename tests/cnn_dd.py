import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .tr import TR

# Use MPS on Mac if available; otherwise fallback to CPU.
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


# Local CNN (patch classifier) with 10 classes for MNIST.
class LocalCNN(nn.Module):
    def __init__(self, in_channels=1, num_filters=8, num_classes=10):
        super(LocalCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        return self.fc(x.view(x.size(0), -1))


# Global CNN (full image classifier).
class GlobalCNN(nn.Module):
    def __init__(self, in_channels=1, num_filters=8, num_classes=10):
        super(GlobalCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        return self.fc(x.view(x.size(0), -1))


# Custom dataset that splits each MNIST image into 4 patches (2x2 grid).
class MNISTPatchesDataset(Dataset):
    def __init__(self, root, train=True, transform=None, patch_size=14):
        self.mnist = datasets.MNIST(
            root=root, train=train, download=True, transform=transform
        )
        self.patch_size = patch_size
        self.image_size = 28  # MNIST images are 28x28.
        self.grid_size = self.image_size // self.patch_size  # 2 patches per dimension.

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]  # image shape: [1,28,28]
        patches = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                patch = image[
                    :,
                    i * self.patch_size : (i + 1) * self.patch_size,
                    j * self.patch_size : (j + 1) * self.patch_size,
                ]
                patches.append(patch)
        # All patches share the same label as the full image.
        patch_labels = [label] * (self.grid_size * self.grid_size)
        return image, patches, patch_labels, label


# Parameters.
batch_size = 64
epochs = 5
num_patches = 4  # 2x2 grid.

# DataLoader using MNIST with a simple ToTensor transform.
transform = transforms.ToTensor()
dataset = MNISTPatchesDataset(
    root="./data", train=True, transform=transform, patch_size=14
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate local models and optimizers (one per patch).
local_models = {i: LocalCNN().to(device) for i in range(num_patches)}
local_optimizers = {
    i: TrustRegionFirstOrder(
        local_models[i],
        lr=1e-3,
        max_iter=10,
        nu_1=0.25,
        nu_2=0.75,
        dec_factor=0.5,
        inc_factor=2.0,
    )
    for i in range(num_patches)
}

# Instantiate global model and its optimizer.
global_model = GlobalCNN().to(device)
global_optimizer = TrustRegionFirstOrder(
    global_model,
    lr=1e-3,
    max_iter=10,
    nu_1=0.25,
    nu_2=0.75,
    dec_factor=0.5,
    inc_factor=2.0,
)
criterion = nn.CrossEntropyLoss()

print("Starting MNIST training with synchronization every epoch on device:", device)
for epoch in range(epochs):
    local_loss_total, local_correct, local_count = 0.0, 0, 0
    global_loss_total, global_correct, global_count = 0.0, 0, 0

    for image, patches_list, patch_labels_list, global_label in dataloader:
        # ----- Local Training (patch-level) -----
        for patch_idx in range(num_patches):
            patch_batch = torch.stack(
                [patches[patch_idx] for patches in patches_list]
            ).to(device)
            label_batch = torch.tensor(
                [labels[patch_idx] for labels in patch_labels_list], device=device
            )
            model = local_models[patch_idx]
            optimizer = local_optimizers[patch_idx]

            def local_closure(compute_grad):
                optimizer.zero_grad()
                logits = model(patch_batch)
                loss = criterion(logits, label_batch)
                if compute_grad:
                    loss.backward()
                return loss

            loss = optimizer.step(local_closure)
            local_loss_total += loss.item()
            # For metrics, evaluate the forward pass.
            logits = model(patch_batch)
            preds = logits.argmax(dim=1)
            local_correct += (preds == label_batch).sum().item()
            local_count += label_batch.size(0)

        # ----- Global Training (full-image) -----
        image = image.to(device)
        global_labels = torch.tensor(global_label, device=device)

        def global_closure(compute_grad=True):
            global_optimizer.zero_grad()
            logits = global_model(image)
            loss = criterion(logits, global_labels)
            if compute_grad:
                loss.backward()
            return loss

        loss = global_optimizer.step(global_closure)
        global_loss_total += loss.item()
        logits = global_model(image)
        preds = logits.argmax(dim=1)
        global_correct += (preds == global_labels).sum().item()
        global_count += global_labels.size(0)

    # Synchronization: average local conv1 parameters and update global, then propagate back.
    with torch.no_grad():
        conv1_weights = torch.stack(
            [local_models[i].conv1.weight.data for i in range(num_patches)], dim=0
        )
        conv1_biases = torch.stack(
            [local_models[i].conv1.bias.data for i in range(num_patches)], dim=0
        )
        avg_weight = conv1_weights.mean(dim=0)
        avg_bias = conv1_biases.mean(dim=0)
        global_model.conv1.weight.data.copy_(avg_weight)
        global_model.conv1.bias.data.copy_(avg_bias)
        for i in range(num_patches):
            local_models[i].conv1.weight.data.copy_(avg_weight)
            local_models[i].conv1.bias.data.copy_(avg_bias)

    local_loss = local_loss_total / local_count
    global_loss = global_loss_total / global_count

    local_acc = 100 * local_correct / local_count
    global_acc = 100 * global_correct / global_count
    print(
        f"Epoch {epoch+1}/{epochs} | Local Loss: {local_loss:.4f}, Accuracy: {local_acc:.2f}% | "
        f"Global Loss: {global_loss:.4f}, Accuracy: {global_acc:.2f}%"
    )
