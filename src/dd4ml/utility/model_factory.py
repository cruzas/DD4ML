import torch
import torch.nn as nn
from collections import OrderedDict


def create_fc_layers(input_features, fc_layers, dropout_p=0.0, activation=nn.ReLU, use_sequential=True):
    """
    Create fully connected layers with consistent structure.

    Args:
        input_features (int): Number of input features
        fc_layers (list): List of hidden layer sizes
        dropout_p (float): Dropout probability
        activation (nn.Module): Activation function class
        use_sequential (bool): If True, return nn.Sequential; if False, return OrderedDict

    Returns:
        nn.Sequential or OrderedDict: Layers for the network
    """
    layers = OrderedDict()
    in_feats = input_features

    for i, h in enumerate(fc_layers, start=1):
        layers[f"fc{i}"] = nn.Linear(in_feats, h)
        layers[f"relu{i}"] = activation()
        if dropout_p > 0:
            layers[f"dropout{i}"] = nn.Dropout(dropout_p)
        in_feats = h

    return nn.Sequential(layers) if use_sequential else layers


def create_model_dict_layer(layer_name, layer_type, settings, stage, src_layers, dst_layers, num_layer_shards=1):
    """
    Create a standardized model dictionary layer entry.

    Args:
        layer_name (str): Name of the layer
        layer_type (type): PyTorch layer class (e.g., nn.Linear, nn.Conv2d)
        settings (dict): Layer-specific settings
        stage (int): Stage number for pipeline parallelism
        src_layers (list): List of source layer names
        dst_layers (list): List of destination layer names
        num_layer_shards (int): Number of layer shards

    Returns:
        dict: Model dictionary entry for the layer
    """
    return {
        layer_name: {
            "callable": {
                "object": layer_type,
                "settings": settings
            },
            "dst": {"to": dst_layers},
            "rcv": {"src": src_layers, "strategy": None},
            "stage": stage,
            "num_layer_shards": num_layer_shards
        }
    }


def create_sequential_model_dict(layers_config, start_stage=0):
    """
    Create a model dictionary for sequential layers.

    Args:
        layers_config (list): List of tuples (layer_name, layer_type, settings)
        start_stage (int): Starting stage number

    Returns:
        dict: Complete model dictionary
    """
    model_dict = {}

    for i, (layer_name, layer_type, settings) in enumerate(layers_config):
        # Determine source and destination layers
        src_layers = [layers_config[i-1][0]] if i > 0 else []
        dst_layers = [layers_config[i+1][0]] if i < len(layers_config) - 1 else []

        # Create layer entry
        layer_entry = create_model_dict_layer(
            layer_name=layer_name,
            layer_type=layer_type,
            settings=settings,
            stage=start_stage + i,
            src_layers=src_layers,
            dst_layers=dst_layers
        )

        model_dict.update(layer_entry)

    return model_dict


def create_fc_stage_modules(config, start_stage="stage2"):
    """
    Create standardized fully connected stage modules for model dictionaries.

    Args:
        config: Configuration object with fc_layers and dropout_p attributes
        start_stage (str): Name of the starting stage (default: "stage2")

    Returns:
        dict: Dictionary of stage modules
    """
    stages = {}

    for i in range(2, len(config.fc_layers) + 1):
        stage_name = f"stage{i}"

        stages[stage_name] = {
            "callable": {
                "object": nn.Sequential,
                "settings": {
                    "modules": [
                        {
                            "object": nn.Linear,
                            "settings": {
                                "in_features": config.fc_layers[i-2],
                                "out_features": config.fc_layers[i-1]
                            }
                        },
                        {
                            "object": nn.ReLU,
                            "settings": {"inplace": True}
                        },
                        {
                            "object": nn.Dropout,
                            "settings": {"p": config.dropout_p}
                        }
                    ]
                }
            },
            "dst": {"to": [f"stage{i+1}"] if i < len(config.fc_layers) else ["finish"]},
            "rcv": {"src": ["start"] if i == 2 else [f"stage{i-1}"], "strategy": None},
            "stage": i - 1,
            "num_layer_shards": 1
        }

    return stages