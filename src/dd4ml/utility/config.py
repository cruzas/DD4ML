from .utils import CfgNode

GPT_MODEL_ALIASES = {
    "nanogpt": "gpt-nano",
    "gptnano": "gpt-nano",
    "microgpt": "gpt-micro",
    "gptmicro": "gpt-micro",
    "minigpt": "gpt-mini",
    "gptmini": "gpt-mini",
    "gpt2": "gpt2",
}


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
    keys_to_remove = set()

    use_pmw = getattr(config.trainer, "use_pmw", False)
    optimizer_lower = config.optimizer.lower()

    # Cache optimizer type checks
    has_apts = "apts" in optimizer_lower
    has_apts_d = "apts_d" in optimizer_lower
    has_apts_p = "apts_p" in optimizer_lower
    has_tr = "tr" in optimizer_lower

    if not use_pmw:
        keys_to_remove.update(["num_stages", "num_replicas_per_subdomain", "model_handler"])
        if not has_apts_d and not has_apts_p:
            keys_to_remove.add("num_subdomains")

    if not has_apts and not has_tr:
        keys_to_remove.update([
            "loc_opt", "loc_opt_hparams", "glob_opt", "glob_opt_hparams",
            "max_loc_iters", "max_glob_iters", "norm_type", "delta",
            "min_delta", "max_delta", "glob_pass", "foc", "dogleg",
            "glob_second_order", "loc_second_order", "max_wolfe_iters", "mem_length"
        ])

    config = remove_keys(config, list(keys_to_remove))
    return config


def get_config(dataset_name: str, model_name: str, optimizer: str):
    """
    Create the base configuration by combining system, data, model, and trainer settings.
    """
    from importlib import import_module
    from dd4ml.trainer import Trainer
    from .factory import DATASET_MAP, MODEL_MAP

    # Early validation
    if dataset_name not in DATASET_MAP:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    model_key = model_name.lower()
    if model_key not in MODEL_MAP:
        raise ValueError(f"Unknown model name: {model_name}")

    C = CfgNode()

    # System configuration
    C.system = CfgNode()
    C.system.seed = 3407
    C.system.trial = 0
    C.system.work_dir = f"../../saved_networks/{dataset_name}/{model_name}/{optimizer}/"
    C.dataset_name = dataset_name
    C.model_name = model_name
    C.optimizer = optimizer

    # Data configuration via DATASET_MAP
    ds_module, ds_class_name = DATASET_MAP[dataset_name]
    dataset_module = import_module(ds_module)
    dataset_cls = getattr(dataset_module, ds_class_name)
    C.data = dataset_cls.get_default_config()

    # Model configuration via MODEL_MAP
    model_module, model_class_name = MODEL_MAP[model_key]
    model_module_obj = import_module(model_module)
    model_cls = getattr(model_module_obj, model_class_name)
    C.model = model_cls.get_default_config()
    C.model.model_class = model_cls
    if model_key in GPT_MODEL_ALIASES:
        C.model.model_type = GPT_MODEL_ALIASES[model_key]

    # Efficiently propagate image-specific properties
    image_attrs = ["input_channels", "input_height", "input_width", "output_classes"]
    for attr in image_attrs:
        if hasattr(C.data, attr):
            setattr(C.model, attr, getattr(C.data, attr))

    # Trainer configuration
    C.trainer = Trainer.get_default_config()
    return C
