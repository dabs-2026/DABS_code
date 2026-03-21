import warnings

from pyro.poutine.runtime import am_i_wrapped, apply_stack


def sample_with_type(type_string, name, fn, *args, **kwargs):
    obs = kwargs.pop("obs", None)
    infer = kwargs.pop("infer", {}).copy()
    if not am_i_wrapped():
        if obs is not None and not infer.get("_deterministic"):
            warnings.warn(
                "trying to observe a value outside of inference at " + name,
                RuntimeWarning,
            )
            return obs
        return fn(*args, **kwargs)
    else:
        msg = {
            "type": "sample",
            "subtype": type_string,
            "name": name,
            "fn": fn,
            "is_observed": False,
            "args": args,
            "kwargs": kwargs,
            "value": None,
            "infer": infer,
            "scale": 1.0,
            "mask": None,
            "cond_indep_stack": (),
            "done": False,
            "stop": False,
            "continuation": None,
        }
        if obs is not None:
            msg["value"] = obs
            msg["is_observed"] = True
        apply_stack(msg)
        return msg["value"]


def observation_sample(name, fn, *args, **kwargs):
    return sample_with_type("observation_sample", name, fn, *args, **kwargs)


def compute_design(name, fn, *args, **kwargs):
    return sample_with_type("design_sample", name, fn, *args, **kwargs)


def latent_sample(name, fn, *args, **kwargs):
    return sample_with_type("latent_sample", name, fn, *args, **kwargs)