import torch.nn as nn


class StandardModel(nn.Module):
    def __init__(self, model_dict):
        super().__init__()
        self.model_dict = model_dict
        
        self.layers = nn.ModuleDict()
        for name, layer_info in self.model_dict.items():
            layer_class = layer_info['callable']['object']
            layer_settings = layer_info['callable']['settings']
            self.layers[name] = layer_class(**layer_settings)
        
    def forward(self, x):
        outputs = {}

        for name in self.model_dict.keys():
            srcs = self.model_dict[name]['rcv']['src']

            if not srcs:  # First layer
                outputs[name] = self.layers[name](x)
            else:
                if len(srcs) == 1:
                    inputs = outputs[srcs[0]]
                else:
                    # Determine if we should add or concatenate
                    input_shapes = [outputs[src].shape for src in srcs]
                    same_shape = all(shape == input_shapes[0] for shape in input_shapes)

                    if same_shape:
                        inputs = sum(outputs[src] for src in srcs)  # Element-wise addition for residual connections
                    else:
                        inputs = torch.cat([outputs[src] for src in srcs], dim=1)  # Concatenation for normal layers

                outputs[name] = self.layers[name](inputs)

        return outputs[list(self.model_dict.keys())[-1]]


def build_standard_model(model_dict):
    return StandardModel(model_dict)
