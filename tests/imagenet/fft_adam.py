import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


class FFTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(FFTConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        # Initialize the kernel
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, *self.kernel_size) * (1.0 / (in_channels * kernel_size[0] * kernel_size[1]))**0.5
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        # Input dimensions: (batch_size, in_channels, height, width)
        batch_size, in_channels, height, width = x.shape
        
        # Pad input and kernel to prevent circular convolution effects
        pad_h = height + self.kernel_size[0] - 1
        pad_w = width + self.kernel_size[1] - 1
        
        x_padded = F.pad(x, (0, pad_w - width, 0, pad_h - height))
        weight_padded = F.pad(self.weight, (0, pad_w - self.kernel_size[1], 0, pad_h - self.kernel_size[0]))

        # Compute FFT of input and kernel
        X_fft = torch.fft.rfft2(x_padded)
        W_fft = torch.fft.rfft2(weight_padded, s=x_padded.shape[2:])
        
        # Perform pointwise multiplication in the frequency domain
        Y_fft = torch.einsum("bchw,oihw->bohw", X_fft, W_fft)
        
        # Inverse FFT to return to the spatial domain
        y = torch.fft.irfft2(Y_fft, s=x_padded.shape[2:])
        
        # Crop to output size (account for stride and padding)
        y = y[:, :, self.padding:-self.padding or None:self.stride, self.padding:-self.padding or None:self.stride]
        
        if self.bias is not None:
            y += self.bias.view(1, -1, 1, 1)
        
        return y

# Example CNN using FFT-based convolutions
class FFTCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super(FFTCNN, self).__init__()
        self.conv1 = FFTConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = FFTConv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(128 * 28 * 28, num_classes)  # For input size 224x224
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Training loop for ImageNet
def train_imagenet():
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    # Transformations for ImageNet
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load ImageNet Dataset
    train_dataset = torchvision.datasets.ImageFolder('/path/to/imagenet/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Model, Loss, and Optimizer
    model = FFTCNN(num_classes=1000).to('cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to('cpu'), labels.to('cpu')

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/10], Loss: {running_loss / len(train_loader)}")

# Uncomment to train
train_imagenet()
