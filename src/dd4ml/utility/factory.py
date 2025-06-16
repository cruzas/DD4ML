from .ml_utils import cross_entropy_transformers
from .utils import import_attr


class Factory:
    """Generic Factory for creating components with dynamic registration and optional caching."""

    def __init__(self, mapping: dict = None, cache_enabled: bool = False):
        """
        Args:
            mapping (dict): Initial mapping of key to (module_path, creator).
            cache_enabled (bool): If True, cache instances created by the factory.
        """
        self.mapping = mapping.copy() if mapping is not None else {}
        self.cache_enabled = cache_enabled
        self._cache = {}  # Cache for storing instances.

    def register(self, key: str, module_path: str, creator):
        """Dynamically register a new component to the factory.

        Args:
            key (str): The unique key to identify the component.
            module_path (str): The module path where the component is defined.
            creator: Either a callable for instantiation or a string representing a class name.
        """
        if key in self.mapping:
            raise ValueError(f"Key '{key}' is already registered.")
        self.mapping[key] = (module_path, creator)

    def create(self, key: str, *args, **kwargs):
        """Create a new instance of a component based on its key.

        Args:
            key (str): The key identifying the component.
            *args: Positional arguments passed to the creator.
            **kwargs: Keyword arguments passed to the creator.

        Returns:
            The instantiated component.
        """
        if key not in self.mapping:
            raise ValueError(f"Unknown key: {key}")

        # If caching is enabled, attempt to retrieve the instance from cache.
        cache_key = (key, args, frozenset(kwargs.items()))
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        module_path, creator = self.mapping[key]
        if callable(creator):
            instance = creator(*args, **kwargs)
        else:
            # Use import_attr to dynamically import the class.
            cls = import_attr(module_path, creator)
            instance = cls(*args, **kwargs)

        if self.cache_enabled:
            self._cache[cache_key] = instance

        return instance


# Mapping definitions.
DATASET_MAP = {
    "mnist": ("dd4ml.datasets.mnist", "MNISTDataset"),
    "cifar10": ("dd4ml.datasets.cifar10", "CIFAR10Dataset"),
    "tinyshakespeare": ("dd4ml.datasets.tinyshakespeare", "TinyShakespeareDataset"),
}

MODEL_MAP = {
    "ffnn": ("dd4ml.models.ffnn.simple_ffnn", "SimpleFFNN"),
    "simple_cnn": ("dd4ml.models.cnn.simple_cnn", "SimpleCNN"),
    "big_cnn": ("dd4ml.models.cnn.big_cnn", "BigCNN"),
    "simple_resnet": ("dd4ml.models.resnet.simple_resnet", "SimpleResNet"),
    "nanogpt": ("dd4ml.models.gpt.nanogpt.model", "GPT"),
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

# You can now add new components dynamically at runtime by calling, e.g.:
# dataset_factory.register("new_dataset", "dd4ml.datasets.new_dataset", "NewDatasetClass")

# Instantiate factories.
# You can enable caching by setting cache_enabled=True if desired.
dataset_factory = Factory(DATASET_MAP, cache_enabled=False)
model_factory = Factory(MODEL_MAP, cache_enabled=False)
criterion_factory = Factory(CRITERION_MAP, cache_enabled=False)
optimizer_factory = Factory(OPTIMIZER_MAP, cache_enabled=False)
