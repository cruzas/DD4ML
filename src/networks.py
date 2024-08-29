import torch.nn as nn


def create_feedforward_stage_list(num_stages):
    stage_list = []
    input_size = 784  # Example input size for a flattened 28x28 image
    hidden_size = 256

    for i in range(num_stages):
        layers = [nn.Linear, nn.ReLU]
        layer_params = [{'in_features': input_size if i ==
                         0 else hidden_size, 'out_features': hidden_size}, {}]
        stage_list.append((layers, layer_params))

    # Final output layer for classification
    final_layers = [nn.Linear, nn.LogSoftmax]
    final_params = [{'in_features': hidden_size,
                     'out_features': 10}, {'dim': 1}]
    stage_list.append((final_layers, final_params))

    return stage_list


def create_cnn_stage_list(num_stages):
    stage_list = []
    in_channels = 1  # Assuming grayscale images for simplicity
    out_channels = 16

    for i in range(num_stages):
        layers = [nn.Conv2d, nn.ReLU, nn.MaxPool2d]
        layer_params = [
            {'in_channels': in_channels if i == 0 else out_channels,
                'out_channels': out_channels, 'kernel_size': 3, 'padding': 1},
            {},
            {'kernel_size': 2, 'stride': 2}
        ]
        stage_list.append((layers, layer_params))
        out_channels *= 2  # Increase the number of channels

    final_layers = [nn.Flatten, nn.Linear, nn.LogSoftmax]
    final_params = [
        {'start_dim': 1},
        {'in_features': out_channels // 2 * 7 * 7, 'out_features': 10},
        {'dim': 1}
    ]
    stage_list.append((final_layers, final_params))

    return stage_list


def create_resnet_stage_list(num_stages):
    stage_list = []
    in_channels = 64
    stage_out_channels = [64, 128, 256, 512]

    for i in range(num_stages):
        layers = [nn.Conv2d, nn.BatchNorm2d,
                  nn.ReLU, nn.Conv2d, nn.BatchNorm2d]
        layer_params = [
            {'in_channels': in_channels if i ==
                0 else stage_out_channels[i-1], 'out_channels': stage_out_channels[i], 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'num_features': stage_out_channels[i]},
            {},
            {'in_channels': stage_out_channels[i], 'out_channels': stage_out_channels[i],
                'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'num_features': stage_out_channels[i]}
        ]
        stage_list.append((layers, layer_params))
        in_channels = stage_out_channels[i]

    final_layers = [nn.AdaptiveAvgPool2d, nn.Flatten, nn.Linear, nn.LogSoftmax]
    final_params = [
        {'output_size': (1, 1)},
        {'start_dim': 1},
        {'in_features': 512, 'out_features': 10},
        {'dim': 1}
    ]
    stage_list.append((final_layers, final_params))

    return stage_list


def construct_stage_list(model_type, num_stages):
    if model_type == "feedforward":
        return create_feedforward_stage_list(num_stages)
    elif model_type == "cnn":
        return create_cnn_stage_list(num_stages)
    elif model_type == "resnet":
        return create_resnet_stage_list(num_stages)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


construct_simple_nn = 1
construct_cnn = 0
construct_resnet = 0
if construct_simple_nn:
    # Example usage:
    # Feedforward stage list with 3 stages
    feedforward_stage_list = construct_stage_list('feedforward', 3)
    print(feedforward_stage_list)

if construct_cnn:
    # CNN stage list with 3 stages
    cnn_stage_list = construct_stage_list('cnn', 3)
    print(cnn_stage_list)

if construct_resnet:
    # ResNet stage list with 3 stages
    resnet_stage_list = construct_stage_list('resnet', 3)
    print(resnet_stage_list)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms

# # ResNet-18 implementation
# class BasicBlock(nn.Module):
#     expansion = 1
#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# class ResNet(nn.Module):
#     # Number of layers is 4, number of classes is 10. This is equivalent to the ResNet-18 architecture.
#     def __init__(self, block=BasicBlock, num_layers=4, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_planes = 64
#         self.module = nn.ModuleList()
#         self.layer_list = [nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())]
#         # self.layer_list =  [nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU()]

#         for l in range(num_layers):
#             if l == num_layers - 1:
#                 blocks = self._make_layer(block, (l+1)*64, 1, stride=2)
#             else:
#                 blocks = self._make_layer(block, (l+1)*64, 1, stride=1)
#             for b in range(len(blocks)):
#                 self.layer_list.append(blocks[b])
#         self.linear = nn.Sequential(
#             # Make this depend on the final block output size
#             nn.AvgPool2d(16),  # Adjust the pooling size to match the input image size
#             nn.Flatten(),
#             nn.Linear( (l+1) * 64 * block.expansion, num_classes)
#         )
#         self.layer_list.append(self.linear)
#         for l in range(len(self.layer_list)):
#             self.module.add_module(f'layer{l}', self.layer_list[l])

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return layers

#     def forward(self, x):
#         for module in self.module: # Adaptive forward depending on the number of layers and blocks
#             print(x.shape)
#             x = module(x)
#         return x

# # CNN implementation for MNIST
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
#         self.pool = nn.MaxPool2d(kernel_size=2)
#         self.fc1 = nn.Linear(64*5*5, 128)
#         self.fc2 = nn.Linear(128, 10)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64*5*5)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

# class CNNPart1(nn.Module):
#     def __init__(self):
#         super(CNNPart1, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
#         self.pool = nn.MaxPool2d(kernel_size=2)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         return x

# class CNNPart2(nn.Module):
#     def __init__(self):
#         super(CNNPart2, self).__init__()
#         self.fc1 = nn.Linear(64*5*5, 128)

#     def forward(self, x):
#         x = x.view(-1, 64*5*5)
#         x = F.relu(self.fc1(x))
#         return x

# class CNNPart3(nn.Module):
#     def __init__(self):
#         super(CNNPart3, self).__init__()
#         self.fc2 = nn.Linear(128, 10)
#         # self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         # x = self.dropout(x)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

# class FCNNPart1(nn.Module):
#     def __init__(self):
#         super(FCNNPart1, self).__init__()
#         self.fc1 = nn.Linear(784, 256)
#         self.fc2 = nn.Linear(256, 128)

#     def forward(self, x):
#         x = nn.Flatten()(x)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return x

# class FCNNPart2(nn.Module):
#     def __init__(self):
#         super(FCNNPart2, self).__init__()
#         self.fc3 = nn.Linear(128, 64)

#     def forward(self, x):
#         x = F.relu(self.fc3(x))
#         return x

# class FCNNPart3(nn.Module):
#     def __init__(self):
#         super(FCNNPart3, self).__init__()
#         self.fc4 = nn.Linear(64, 10)

#     def forward(self, x):
#         x = self.fc4(x)
#         return F.log_softmax(x, dim=1)
