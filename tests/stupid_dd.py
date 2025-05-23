import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ---------- Model Definition ----------
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------- Two-Level Training Function ----------
def train_two_level(
    num_epochs=100,
    local_steps=1,
    batch_size=64,
    coarse_batch=256,
    num_subdomains=4,
    coarse_dim=10,
    lr_local=0.1,
    lr_coarse=0.1,
):
    # Data transforms and datasets
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(".", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(".", train=False, download=True, transform=transform)

    # Split into subdomains
    subset_size = len(full_train) // num_subdomains
    subsets = [
        Subset(full_train, range(i * subset_size, (i + 1) * subset_size))
        for i in range(num_subdomains)
    ]
    loaders = [DataLoader(s, batch_size=batch_size, shuffle=True) for s in subsets]

    # Full loader for coarse updates
    full_loader = DataLoader(full_train, batch_size=coarse_batch, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    # Initialize local nets and optimizers
    nets = [SimpleNet() for _ in range(num_subdomains)]
    opts = [torch.optim.SGD(net.parameters(), lr=lr_local) for net in nets]

    # Determine coarse dimension
    actual_k = min(coarse_dim, num_subdomains)

    # Initialize coarse coefficients
    alpha = torch.zeros(actual_k, requires_grad=True)
    opt_alpha = torch.optim.SGD([alpha], lr=lr_coarse)

    # Training loop
    for epoch in range(num_epochs):
        # Local updates
        for _ in range(local_steps):
            for net, opt, loader in zip(nets, opts, loaders):
                net.train()
                for x, y in loader:
                    opt.zero_grad()
                    loss = F.cross_entropy(net(x), y)
                    loss.backward()
                    opt.step()

        # Build coarse basis via SVD
        with torch.no_grad():
            Ws = [torch.cat([p.view(-1) for p in net.parameters()]) for net in nets]
            Wmat = torch.stack(Ws, dim=1)
            Wmean = Wmat.mean(dim=1, keepdim=True)
            Wc = Wmat - Wmean
        U, S, V = torch.linalg.svd(Wc, full_matrices=False)
        Uk = U[:, :actual_k]

        # Coarse correction step
        x0, y0 = next(iter(full_loader))
        Wcoarse = Wmean.squeeze() + Uk @ alpha
        temp = SimpleNet()
        with torch.no_grad():
            ptr = 0
            for p in temp.parameters():
                numel = p.numel()
                p.copy_(Wcoarse[ptr : ptr + numel].view_as(p))
                ptr += numel

        opt_alpha.zero_grad()
        loss_c = F.cross_entropy(temp(x0), y0)
        loss_c.backward()
        opt_alpha.step()

        # Inject coarse update back into local nets
        with torch.no_grad():
            delta = Uk @ alpha
            for net in nets:
                ptr = 0
                for p in net.parameters():
                    numel = p.numel()
                    p.add_(delta[ptr : ptr + numel].view_as(p))
                    ptr += numel

        # Evaluate coarse model on test set
        temp.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_test, y_test in test_loader:
                logits = temp(x_test)
                preds = logits.argmax(dim=1)
                correct += (preds == y_test).sum().item()
                total += y_test.size(0)
        acc = correct / total * 100

        print(
            f"Epoch {epoch+1}/{num_epochs} - Coarse loss: {loss_c.item():.4f} - Test Acc: {acc:.2f}%"
        )


# ---------- Main Entry ----------
if __name__ == "__main__":
    train_two_level()
