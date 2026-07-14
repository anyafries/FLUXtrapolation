import inspect

def test_model(model):
    """
    Checks model capabilities.
    Returns: (use_eval_set, use_envs)
    """
    # 1. Basic sanity check
    if not (hasattr(model, "fit") and hasattr(model, "predict")):
        raise AttributeError(f"Model {type(model)} missing fit/predict methods.")

    # 2. Introspect the fit method signature
    sig = inspect.signature(model.fit)
    params = sig.parameters

    if not 'X' in params or not 'y' in params:
        raise ValueError(f"Model {type(model)} fit method must have 'X' and 'y' parameters.")
    
    use_eval_set = 'eval_set' in params
    # OOD models (CORAL, GDRO, MMD) usually require 'envs'
    use_envs = 'envs' in params
    use_feature_names = 'feature_names' in params

    return use_eval_set, use_envs, use_feature_names