import torch.nn as nn
import torch

def preprocessing(batch):
    batch = nn.Flatten()(batch)
    return [batch[:,:700], batch[:,700:]]

def average_fun(input1, input2):
    return (input1+input2)/2

def get_model_dict():
    # TODO: Maybe strategy in 'dst' is redundant, as rcv strategy may be enough to handle the output of the sending layer(s)
    model = {
        "start": {
            "callable": {'object': preprocessing, 'settings': {}}, # the output of the callable function HAS to be a pytorch tensor
            "dst": {"to": ["input_layer1_1", "input_layer1_2"]},
            "rcv": {"src": [], "strategy": None},
            "stage": 1,
            "num_layer_shards": 1,
        },
        "input_layer1_1": {
            "callable": {'object': nn.Linear, 'settings': {"in_features": 700, "out_features": 256}},
            "dst": {'to': ["input_layer2_1"]}, # strategy must have as many outputs as the length of the to list (input will be the output of the processed data in current callable)
            "rcv": {'src': ["start"], 'strategy': None},
            "stage": 1, 
            "num_layer_shards": 1,
        },
        "input_layer1_2": {
            "callable": {'object': nn.Linear, "settings": {"in_features": 84, "out_features": 32}},
            "dst": {'to': ["input_layer2_2"]},
            "rcv": {"src": ["start"], "strategy": None},
            "stage": 2, 
            "num_layer_shards": 1,
        },  
        "input_layer2_1": {
            "callable": {'object': nn.Linear, 'settings': {"in_features": 256, "out_features": 128}},
            "dst": {'to': ["finish"]}, # strategy must have as many outputs as the length of the to list (input will be the output of the processed data in current callable)
            "rcv": {'src': ["input_layer1_1"], 'strategy': None},
            "stage": 1, 
            "num_layer_shards": 1,
        },
        "input_layer2_2": {
            "callable": {'object': nn.Linear, "settings": {"in_features": 32, "out_features": 128}},
            "dst": {'to': ["finish"]},
            "rcv": {"src": ["input_layer1_2"], "strategy": None},
            "stage": 2, 
            "num_layer_shards": 1,
        },
        "finish": {
            "callable": {'object': nn.Linear, "settings": {"in_features": 128, "out_features": 10}},
            "dst": {"to": []},
            "rcv": {'src': ["input_layer2_1", "input_layer2_2"], 'strategy': average_fun}, # inputs of "average_fun" are taken in the same order as the src list
            "stage": 2,
            "num_layer_shards": 1,
        },
    }

    return model

class GlobalModel(nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        
        # Define layers based on the provided structure
        self.input_layer1_1 = nn.Linear(700, 256)
        self.input_layer1_2 = nn.Linear(84, 32)
        self.input_layer2_1 = nn.Linear(256, 128)
        self.input_layer2_2 = nn.Linear(32, 128)
        self.finish = nn.Linear(128, 10)
        
    def forward(self, x):
        # Start: Preprocess the input
        x1, x2 = preprocessing(x)  # Assuming the input can be preprocessed as needed
        
        # Stage 1
        out1_1 = self.input_layer1_1(x1)  # Branch 1
        out1_2 = self.input_layer1_2(x2)  # Branch 2
        
        # Stage 2
        out2_1 = self.input_layer2_1(out1_1)  # Processed from input_layer1_1
        out2_2 = self.input_layer2_2(out1_2)  # Processed from input_layer1_2
        
        # Combine outputs using the strategy (average_fun)
        combined_output = average_fun(out2_1, out2_2)
        
        # Finish layer
        output = self.finish(combined_output)
        
        return output