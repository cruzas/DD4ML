import torch
import torch.nn as nn
import torch.nn.functional as F

from dd4ml.models.cnn.base_cnn import BaseCNN


# Reusable blocks.
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, pool_size, stride):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv(x))
        return self.pool(x)

class ConvPoolAdaptiveBlock(nn.Module):
    """
    Composite block that performs:
      Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> AdaptiveAvgPool2d

    The adaptive pooling forces the spatial output to the given size.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, pool_size, stride, adaptive_output):
        super(ConvPoolAdaptiveBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Use max pooling as in the original pipeline.
        self.maxpool = nn.MaxPool2d(kernel_size=pool_size, stride=2)
        # Adaptive pooling forces spatial dimensions to adaptive_output.
        self.adaptive_pool = nn.AdaptiveAvgPool2d(adaptive_output)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.adaptive_pool(x)
        return x

class FlattenBlock(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, activation='relu'):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU(inplace=True) if activation == 'relu' else None
        
    def forward(self, x):
        x = self.fc(x)
        if self.activation:
            x = self.activation(x)
        return x

class DropoutFCBlock(nn.Module):
    def __init__(self, in_features, out_features, p):
        super(DropoutFCBlock, self).__init__()
        self.dropout = nn.Dropout(p)
        self.fc = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        x = self.dropout(x)
        return self.fc(x)

# BigCNN uses BaseCNN and its configuration.
class BigCNN(BaseCNN):
    @staticmethod
    def get_default_config():
        C = BaseCNN.get_default_config()
        return C
    
    def __init__(self, config):
        super().__init__(config)
        # Stage 1: Conv only (simulate no pooling by using pool_size=1)
        self.start = nn.Sequential(
            nn.Conv2d(config.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Stage 2: Conv + pooling
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Stage 3: Conv only
        self.stage3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Stage 4: Conv + pooling
        self.stage4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Stage 5: Conv only
        self.stage5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # Stage 6: Composite block that uses adaptive pooling to force output to (3,3)
        self.stage6 = nn.Sequential(
            ConvPoolAdaptiveBlock(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1,
                pool_size=2,
                stride=1,
                adaptive_output=(3, 3)
            )
        )
        # Stage 7: Flatten and fully connected layer.
        self.stage7 = nn.Sequential(
            FlattenBlock(),
            FCBlock(128 * 3 * 3, 256, activation='relu')
        )
        # Stage 8: Dropout and final fully connected layer.
        self.finish = DropoutFCBlock(256, 10, p=0.5)
        
    def forward(self, x):
        x = self.start(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.finish(x)
        return x

    def as_model_dict(self):
        model_dict = {
            'start': {
                'callable': {
                    'object': ConvBlock,
                    'settings': {
                        'in_channels': self.config.input_channels,
                        'out_channels': 32,
                        'kernel_size': 3,
                        'padding': 1,
                        'pool_size': 1,  # no pooling effect
                        'stride': 1,
                    },
                },
                'dst': {'to': ['stage2']},
                'rcv': {'src': [], 'strategy': None},
                'stage': 0,
                'num_layer_shards': 1,
            },
            'stage2': {
                'callable': {
                    'object': ConvBlock,
                    'settings': {
                        'in_channels': 32,
                        'out_channels': 32,
                        'kernel_size': 3,
                        'padding': 1,
                        'pool_size': 2,
                        'stride': 1,
                    },
                },
                'dst': {'to': ['stage3']},
                'rcv': {'src': ['start'], 'strategy': None},
                'stage': 1,
                'num_layer_shards': 1,
            },
            'stage3': {
                'callable': {
                    'object': ConvBlock,
                    'settings': {
                        'in_channels': 32,
                        'out_channels': 64,
                        'kernel_size': 3,
                        'padding': 1,
                        'pool_size': 1,  # no pooling
                        'stride': 1,
                    },
                },
                'dst': {'to': ['stage4']},
                'rcv': {'src': ['stage2'], 'strategy': None},
                'stage': 2,
                'num_layer_shards': 1,
            },
            'stage4': {
                'callable': {
                    'object': ConvBlock,
                    'settings': {
                        'in_channels': 64,
                        'out_channels': 64,
                        'kernel_size': 3,
                        'padding': 1,
                        'pool_size': 2,
                        'stride': 1,
                    },
                },
                'dst': {'to': ['stage5']},
                'rcv': {'src': ['stage3'], 'strategy': None},
                'stage': 3,
                'num_layer_shards': 1,
            },
            'stage5': {
                'callable': {
                    'object': ConvBlock,
                    'settings': {
                        'in_channels': 64,
                        'out_channels': 128,
                        'kernel_size': 3,
                        'padding': 1,
                        'pool_size': 1,  # no pooling
                        'stride': 1,
                    },
                },
                'dst': {'to': ['stage6']},
                'rcv': {'src': ['stage4'], 'strategy': None},
                'stage': 4,
                'num_layer_shards': 1,
            },
            # The stage6 entry uses ConvPoolAdaptiveBlock to force a (3,3) output.
            'stage6': {
                'callable': {
                    'object': ConvPoolAdaptiveBlock,
                    'settings': {
                        'in_channels': 128,
                        'out_channels': 128,
                        'kernel_size': 3,
                        'padding': 1,
                        'pool_size': 2,
                        'stride': 1,
                        'adaptive_output': (3, 3),
                    },
                },
                'dst': {'to': ['stage7']},
                'rcv': {'src': ['stage5'], 'strategy': None},
                'stage': 5,
                'num_layer_shards': 1,
            },
            'stage7': {
                'callable': {
                    'object': FCBlock,
                    'settings': {
                        'in_features': 128 * 3 * 3,
                        'out_features': 256,
                        'activation': 'relu',
                    },
                },
                'dst': {'to': ['finish']},
                'rcv': {'src': ['stage6'], 'strategy': None},
                'stage': 6,
                'num_layer_shards': 1,
            },
            'finish': {
                'callable': {
                    'object': DropoutFCBlock,
                    'settings': {
                        'in_features': 256,
                        'out_features': 10,
                        'p': 0.5,
                    },
                },
                'dst': {'to': []},
                'rcv': {'src': ['stage7'], 'strategy': None},
                'stage': 7,
                'num_layer_shards': 1,
            },
        }
        self.model_dict = model_dict
        self.set_stage()
        return self.model_dict
