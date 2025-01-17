import torch
import torch.nn as nn

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current macOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device.")

else:
    mps_device = torch.device("mps")

    # Define a simple network
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(20, 3)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    # Move model to MPS
    model = SimpleNet().to(mps_device)

    # Create a random input on MPS
    x = torch.randn(5, 10, device=mps_device)

    # Forward pass
    pred = model(x)
    print(pred)
