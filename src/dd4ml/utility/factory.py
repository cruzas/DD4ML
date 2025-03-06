from .ml_utils import cross_entropy_transformers
from .utils import import_attr


class Factory:
    """Generic Factory for creating components."""

    def __init__(self, mapping: dict):
        self.mapping = mapping

    def create(self, key: str, *args, **kwargs):
        if key not in self.mapping:
            raise ValueError(f"Unknown key: {key}")
        module_path, creator = self.mapping[key]
        if callable(creator):
            return creator(*args, **kwargs)
        cls = import_attr(module_path, creator)
        return cls(*args, **kwargs)


# Mapping definitions.
DATASET_MAP = {
    "mnist": ("dd4ml.datasets.mnist", "MNISTDataset"),
    "cifar10": ("dd4ml.datasets.cifar10", "CIFAR10Dataset"),
    "tinyshakespeare": ("dd4ml.datasets.tinyshakespeare", "TinyShakespeareDataset"),
}

MODEL_MAP = {
    "simple_cnn": ("dd4ml.models.cnn.simple_cnn", "SimpleCNN"),
    "big_cnn": ("dd4ml.models.cnn.big_cnn", "BigCNN"),
    "simple_resnet": ("dd4ml.models.resnet.simple_resnet", "SimpleResNet"),
    "mingpt": ("dd4ml.models.gpt.mingpt.model", "GPT"),
}

CRITERION_MAP = {
    "cross_entropy": ("", lambda ds=None: __import__("torch.nn").nn.CrossEntropyLoss()),
    "weighted_cross_entropy": (
        "",
        lambda ds: __import__("torch.nn").nn.CrossEntropyLoss(
            weight=ds.compute_class_weights()
        ),
    ),
    "mse": ("", lambda ds=None: __import__("torch.nn").nn.MSELoss()),
    "cross_entropy_transformers": (
        "",
        lambda ds=None: cross_entropy_transformers,  # Assumes defined elsewhere.
    ),
}

OPTIMIZER_MAP = {
    "sgd": (
        "",
        lambda model, lr: __import__("torch.optim").optim.SGD(
            model.parameters(), lr=lr, momentum=0.9
        ),
    ),
    "adam": (
        "",
        lambda model, lr: __import__("torch.optim").optim.Adam(
            model.parameters(), lr=lr
        ),
    ),
    "adamw": (
        "",
        lambda model, lr: __import__("torch.optim").optim.AdamW(
            model.parameters(), lr=lr
        ),
    ),
    "adagrad": (
        "",
        lambda model, lr: __import__("torch.optim").optim.Adagrad(
            model.parameters(), lr=lr
        ),
    ),
    "rmsprop": (
        "",
        lambda model, lr: __import__("torch.optim").optim.RMSprop(
            model.parameters(), lr=lr
        ),
    ),
}


# Instantiate factories.
dataset_factory = Factory(DATASET_MAP)
model_factory = Factory(MODEL_MAP)
criterion_factory = Factory(CRITERION_MAP)
optimizer_factory = Factory(OPTIMIZER_MAP)
