from src.models.cnn.base_cnn import *


class MyCNN(BaseCNN):
    
    @staticmethod
    def get_default_config():
        C = BaseCNN.get_default_config()
        # Pipelining options
        C.num_stages = 1
        C.num_subdomains = 1
        C.num_replicas_per_subdomain = 1
        return C
    
    def __init__(self, config):
        super().__init__(config)
        self.model_dict = self.build_cnn_dictionary(config)
        
    def build_cnn_dictionary(self, config):
        model_dict = {}

        # First Convolutional Block (conv1 + pool)
        model_dict['start'] = {
            'callable': {
                'object': ConvBlock,
                'settings': {
                    'in_channels': config.input_channels,    
                    'out_channels': 32,
                    'kernel_size': 3,
                    'pool_size': 2,      # Pooling down by factor of 2
                    'stride': 1,         # Convolution stride
                    'padding': 1,        # Keep spatial size consistent
                },
            },
            'dst': {'to': ['conv2']},
            'rcv': {'src': [], 'strategy': None},
            'stage': 0,
            'num_layer_shards': 1,
        }

        # Second Convolutional Block (conv2 + pool)
        model_dict['conv2'] = {
            'callable': {
                'object': ConvBlock,
                'settings': {
                    'in_channels': 32,
                    'out_channels': 64,
                    'kernel_size': 3,
                    'pool_size': 2,
                    'stride': 1,
                    'padding': 1,
                },
            },
            'dst': {'to': ['fc1']},
            'rcv': {'src': ['start'], 'strategy': None},
            'stage': 0,
            'num_layer_shards': 1,
        }

        # First Fully Connected Block
        model_dict['fc1'] = {
            'callable': {
                'object': FullyConnectedBlock,
                'settings': {
                    'in_features': 64 * 7 * 7,  # Matches output from second pool
                    'out_features': 128,
                },
            },
            'dst': {'to': ['finish']},
            'rcv': {'src': ['conv2'], 'strategy': None},
            'stage': 0,
            'num_layer_shards': 1,
        }

        # Final Fully Connected Layer
        model_dict['finish'] = {
            'callable': {
                'object': nn.Linear,
                'settings': {
                    'in_features': 128,
                    'out_features': config.output_classes,  
                    'bias': True,
                },
            },
            'dst': {'to': []},
            'rcv': {'src': ['fc1'], 'strategy': None},
            'stage': 0,
            'num_layer_shards': 1,
        }

        # Optionally reset all stages (if needed for your pipeline approach)
        for name in model_dict:
            model_dict[name]['stage'] = 0

        return self.set_stage(model_dict, config.num_stages)


    def forward(self, x):
        # Forward logic implemented in pipeline
        pass