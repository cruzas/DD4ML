import torch.nn as nn


def create_feedforward_stage_list(num_stages, input_size=2, out_features=1, hidden_size=20):
    stage_list = []
    if num_stages == 1:
        layers = [nn.Linear, nn.Tanh, nn.Linear, nn.Tanh, nn.Linear]
        layer_params = [{'in_features': input_size, 'out_features': hidden_size}, {}, {'in_features': hidden_size, 'out_features': hidden_size}, {}, {'in_features': hidden_size, 'out_features': out_features}]
        stage_list.append((layers, layer_params))
        return stage_list
    else:
        for _ in range(num_stages - 1):
            layers = [nn.Linear, nn.Tanh]
            layer_params = [{'in_features': input_size, 'out_features': hidden_size}, {}]
            stage_list.append((layers, layer_params))

        layers = [nn.Linear]
        layer_params = [{'in_features': hidden_size, 'out_features': out_features}]
        stage_list.append((layers, layer_params))
                
    return stage_list


def create_cnn_stage_list(num_stages):
    stage_list = []
    in_channels = 1  # Assuming grayscale images for simplicity
    out_channels = 16

    for i in range(num_stages - 1):
        layers = [nn.Conv2d, nn.ReLU, nn.MaxPool2d]
        layer_params = [
            {'in_channels': in_channels if i == 0 else out_channels, 'out_channels': out_channels, 'kernel_size': 3, 'padding': 1},
            {},
            {'kernel_size': 2, 'stride': 2}
        ]
        stage_list.append((layers, layer_params))
        out_channels *= 2  # Increase the number of channels

    # Final output stage
    final_layers = [nn.Flatten, nn.Linear, nn.LogSoftmax]
    final_params = [
        {'start_dim': 1},
        {'in_features': out_channels // 2 * 7 * 7, 'out_features': 10},
        {'dim': 1}
    ]
    stage_list.append((final_layers, final_params))

    return stage_list

class ResNetBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResNetBlock, self).__init__()
        self.layer = nn.Linear(in_features, out_features)
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(self.layer(x))

def create_resnet_stage_list(num_stages):
    stage_list = []
    in_channels = 2  # Starting number of input channels
    stage_out_channels = 20  # Example ResNet stage sizes

    # Add stages
    for i in range(num_stages - 1):
        layers = []
        layer_params = []
        
        # Define a residual block for each stage
        if i == 0:
            # First stage: the input stage
            layers = [nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Conv2d, nn.BatchNorm2d, nn.ReLU]
            layer_params = [
                {'in_channels': in_channels, 'out_channels': stage_out_channels, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'num_features': stage_out_channels},
                {},
                {'in_channels': stage_out_channels, 'out_channels': stage_out_channels, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'num_features': stage_out_channels},
                {}
            ]
        else:
            # Subsequent stages: deeper layers
            layers = [nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Conv2d, nn.BatchNorm2d, nn.ReLU]
            layer_params = [
                {'in_channels': stage_out_channels, 'out_channels': stage_out_channels, 'kernel_size': 3, 'stride': 2, 'padding': 1},
                {'num_features': stage_out_channels},
                {},
                {'in_channels': stage_out_channels, 'out_channels': stage_out_channels, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'num_features': stage_out_channels},
                {}
            ]

        stage_list.append((layers, layer_params))

    # Final output stage
    final_layers = [nn.AdaptiveAvgPool2d, nn.Flatten, nn.Linear, nn.LogSoftmax]
    final_params = [
        {'output_size': (1, 1)},
        {'start_dim': 1},
        {'in_features': stage_out_channels, 'out_features': 1},  # Example for classification with 10 classes
        {'dim': 1}
    ]
    stage_list.append((final_layers, final_params))

    return stage_list


def construct_stage_list(model_type, num_stages):
    return create_feedforward_stage_list(num_stages)
