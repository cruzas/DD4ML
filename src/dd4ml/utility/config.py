from .utils import CfgNode


def remove_keys(config, keys_to_remove):
    """
    Recursively remove keys from a configuration dict or object.
    """
    if isinstance(config, dict):
        for k in keys_to_remove:
            config.pop(k, None)
        for key, value in config.items():
            config[key] = remove_keys(value, keys_to_remove)
    elif hasattr(config, "__dict__"):
        for k in keys_to_remove:
            if k in config.__dict__:
                del config.__dict__[k]
        for k in list(config.__dict__.keys()):
            value = getattr(config, k)
            updated_value = remove_keys(value, keys_to_remove)
            setattr(config, k, updated_value)
    return config


def make_std_config(config):
    """
    Standardize configuration by removing unnecessary keys.
    """
    keys_to_remove = []

    use_pmw = getattr(config.trainer, "use_pmw", False)
    if not use_pmw:
        keys_to_remove.extend(
            ["num_stages", "num_replicas_per_subdomain", "model_handler"]
        )
        if (
            "apts_d" not in config.optimizer.lower()
            and not "apts_p" in config.optimizer.lower()
        ):
            keys_to_remove.append("num_subdomains")
        config = remove_keys(config, keys_to_remove)
    if "apts" not in config.optimizer.lower():
        keys_to_remove = [
            "loc_opt",
            "loc_opt_hparams",
            "glob_opt",
            "glob_opt_hparams",
            "max_loc_iters",
            "max_glob_iters",
            "norm_type",
            "delta",
            "min_delta",
            "max_delta",
            "glob_pass",
            "foc",
            "dogleg" "glob_second_order",
            "loc_second_order",
            "max_wolfe_iters",
            "mem_length",
        ]
        config = remove_keys(config, keys_to_remove)
    if config.optimizer.lower() == "apts_d" or config.optimizer.lower() == "apts_p":
        keys_to_remove = [
            "glob_opt",
            "loc_opt",
        ]
        config = remove_keys(config, keys_to_remove)
    return config


def get_config(dataset_name: str, model_name: str, optimizer: str):
    """
    Create the base configuration by combining system, data, model, and trainer settings.
    """
    from importlib import import_module

    from dd4ml.trainer import (
        Trainer,
    )  # Assumes Trainer and CfgNode are defined in dd4ml.trainer.

    C = CfgNode()  # CfgNode should be defined elsewhere.

    # System configuration.
    C.system = CfgNode()
    C.system.seed = 3407
    C.system.trial = 0
    C.system.work_dir = f"../../saved_networks/{dataset_name}/{model_name}/{optimizer}/"

    # Data configuration via DATASET_MAP.
    from .factory import DATASET_MAP

    if dataset_name not in DATASET_MAP:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    ds_module, ds_class_name = DATASET_MAP[dataset_name]
    dataset_module = import_module(ds_module)
    dataset_cls = getattr(dataset_module, ds_class_name)
    C.data = dataset_cls.get_default_config()  # Assumes method exists.

    # Model configuration via MODEL_MAP.
    from .factory import MODEL_MAP

    key = next((k for k in MODEL_MAP if k in model_name.lower()), None)
    if key is None:
        raise ValueError(f"Unknown model name: {model_name}.")
    model_module, model_class_name = MODEL_MAP[key]
    model_module_obj = import_module(model_module)
    model_cls = getattr(model_module_obj, model_class_name)
    C.model = model_cls.get_default_config()
    C.model.model_class = model_cls

    # Propagate image-specific properties if available.
    for attr in ["input_channels", "input_height", "input_width", "output_classes"]:
        if hasattr(C.data, attr):
            setattr(C.model, attr, getattr(C.data, attr))

    # Trainer configuration.
    C.trainer = Trainer.get_default_config()
    return C
