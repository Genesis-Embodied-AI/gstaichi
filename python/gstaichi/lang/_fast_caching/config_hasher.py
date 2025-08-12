from gstaichi.lang import impl

from .hash_utils import hash_string


def hash_compile_config() -> str:
    config = impl.get_runtime().prog.config()
    config_l = []
    for k in dir(config):
        if k.startswith("_"):
            continue
        v = getattr(config, k)
        config_l.append(f"{k}={v}")
    config_str = "\n".join(config_l)
    config_hash = hash_string(config_str)
    return config_hash
