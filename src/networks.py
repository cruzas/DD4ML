import torch.nn as nn


def create_feedforward_stage_list(num_stages):
    stage_list = []
    input_size = 784  # Example input size for a flattened 28x28 image
    hidden_size = 256

    for i in range(num_stages - 1):
        if i == 0:
            layers = [nn.Flatten, nn.Linear, nn.ReLU, nn.Linear, nn.ReLU]
            layer_params = [{'start_dim': 1}, {'in_features': input_size, 'out_features': hidden_size}, {}, {'in_features': hidden_size, 'out_features': hidden_size}, {}]
        elif i == 1:
            layers = [nn.Linear, nn.ReLU]
            layer_params = [{'in_features': hidden_size, 'out_features': 128}, {}]
        # else:
        #     layers = [nn.Linear, nn.ReLU]
        #     layer_params = [{'in_features': hidden_size, 'out_features': hidden_size}, {}]
        stage_list.append((layers, layer_params))

    # Final output stage
    final_layers = [nn.Linear, nn.Sigmoid, nn.LogSoftmax]
    final_params = [{'in_features': 128, 'out_features': 10}, {}, {'dim': 1}]
    # final_params = [{'in_features': hidden_size, 'out_features': 10}, {}, {'dim': 1}]
    stage_list.append((final_layers, final_params))

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


def create_resnet_stage_list(num_stages):
    stage_list = []
    in_channels = 64
    stage_out_channels = [64, 128, 256, 512]

    for i in range(num_stages - 1):
        layers = [nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Conv2d, nn.BatchNorm2d]
        layer_params = [
            {'in_channels': in_channels if i == 0 else stage_out_channels[i-1], 'out_channels': stage_out_channels[i], 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'num_features': stage_out_channels[i]},
            {},
            {'in_channels': stage_out_channels[i], 'out_channels': stage_out_channels[i], 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'num_features': stage_out_channels[i]}
        ]
        stage_list.append((layers, layer_params))
        in_channels = stage_out_channels[i]

    # Final output stage
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
