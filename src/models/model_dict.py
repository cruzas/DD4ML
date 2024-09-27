import torch.nn as nn

def preprocessing(batch):
    # flatten the batch
    batch = nn.flatten(batch)
    return batch[:,:700], batch[:,700:]

def average_fun(input1, input2):
    return (input1+input2)/2

def get_model_dict():
    model = {
        "start": {
            "callable": {'object': preprocessing, 'settings': {}},
            "dst": {"to": ["input_layer1_1", "input_layer1_2"], "strategy": None},
            "rcv": {"src": [], "strategy": None},
            "stage": 1,
            "num_layer_shards": 2,
        },
        "input_layer1_1": {
            "callable": {'object': nn.Linear, 'settings': {"in_features": 700, "out_features": 256}},
            "dst": {'to': ["input_layer2_1"], 'strategy': None}, # strategy must have as many outputs as the length of the to list (input will be the output of the processed data in current callable)
            "rcv": {'src': ["start"], 'strategy': None},
            "stage": 1, 
            "num_layer_shards": 2,
        },
        "input_layer1_2": {
            "callable": {'object': nn.Linear, "settings": {"in_features": 84, "out_features": 32}},
            "dst": {'to': ["input_layer2_2"], 'strategy': None},
            "rcv": {"src": ["start"], "strategy": None},
            "stage": 2, 
            "num_layer_shards": 2,
        },  
        "input_layer2_1": {
            "callable": {'object': nn.Linear, 'settings': {"in_features": 256, "out_features": 128}},
            "dst": {'to': ["finish"], 'strategy': None}, # strategy must have as many outputs as the length of the to list (input will be the output of the processed data in current callable)
            "rcv": {'src': ["input_layer1_1"], 'strategy': None},
            "stage": 1, 
            "num_layer_shards": 2,
        },
        "input_layer2_2": {
            "callable": {'object': nn.Linear, "settings": {"in_features": 32, "out_features": 128}},
            "dst": {'to': ["finish"], 'strategy': None},
            "rcv": {"src": ["input_layer1_2"], "strategy": None},
            "stage": 2, 
            "num_layer_shards": 2,
        },
        "finish": {
            "callable": {'object': nn.Linear, "settings": {"in_features": 128, "out_features": 10}},
            "dst": {"to": [], "strategy": None},
            "rcv": {'src': ["input_layer2_1", "input_layer2_2"], 'strategy': average_fun},
            "stage": 2,
            "num_layer_shards": 2,
        },
    }

    return model