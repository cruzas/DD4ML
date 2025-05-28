import math

import torch.distributed as dist


class Timer:
    def __init__(self, timings, key):
        self.timings = timings
        self.key = key

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        self.timings[self.key] += time.time() - self.start


def get_state_dict(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def get_trust_region_params(config):
    lr = config.learning_rate
    return {
        "lr": lr,
        "max_lr": 2.0,
        "min_lr": 1e-6,
        "nu": 0.5,
        "inc_factor": 1.2,
        "dec_factor": 0.9,
        "nu_dec": 0.25,
        "nu_inc": 0.75,
        "max_iter": config.max_global_iters,
        "norm_type": config.norm_type,
        "second_order": config.global_second_order,
    }


def get_local_trust_region_params(config):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    norm_type = config.norm_type
    lr_scale = 1.0 / world_size if config.norm_type != math.inf else 1.0
    lr = config.learning_rate * lr_scale
    return {
        "lr": lr,
        "max_lr": 2.0,  # Lower maximum for local updates
        "min_lr": 1e-6,
        "nu": (
            0.45 if norm_type != math.inf else 0.5
        ),  # Adjusted to be more conservative locally
        "inc_factor": 1.5 if norm_type != math.inf else 1.2,  # Reduced increase factor
        "dec_factor": (
            0.6 if norm_type != math.inf else 0.9
        ),  # Slightly more aggressive reduction
        "nu_dec": 0.3 if norm_type != math.inf else 0.25,
        "nu_inc": 0.7 if norm_type != math.inf else 0.75,
        "max_iter": config.max_subdomain_iters,
        "norm_type": config.norm_type,
        "second_order": config.local_second_order,
    }
