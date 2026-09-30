import numpy as np


def stack_transition_info(infos, specs):
    """Stack explicitly allowlisted info fields for replay transport."""
    result = {}
    for key, (shape, dtype) in specs.items():
        zero = np.zeros(shape, dtype=dtype)
        values = [np.asarray(info.get(key, zero), dtype=dtype) for info in infos]
        result[key] = np.stack(values)
    return result
