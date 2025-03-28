def get_state_dict(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()

def get_trust_region_params(config, lr_scale=1.0, max_iter=3):
    return {
        "lr": config.learning_rate * lr_scale,
        "max_lr": 2.0,
        "min_lr": 1e-4,
        "nu": 0.5,
        "inc_factor": 2.0,
        "dec_factor": 0.5,
        "nu_1": 0.25,
        "nu_2": 0.75,
        "max_iter": max_iter,
        "norm_type": 2,
    }
    
def get_local_trust_region_params(config, lr_scale=1.0, max_iter=3):
    return {
        "lr": config.learning_rate * lr_scale,
        "max_lr": 1.5,        # Lower maximum for local updates
        "min_lr": 1e-4,
        "nu": 0.45,           # Adjusted to be more conservative locally
        "inc_factor": 1.5,    # Reduced increase factor
        "dec_factor": 0.6,    # Slightly more aggressive reduction
        "nu_1": 0.3,
        "nu_2": 0.7,
        "max_iter": max_iter,
        "norm_type": 2,
    }

