import numpy as np

def split_forgetting_indices(n: int, forget_fraction: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < forget_fraction < 1: raise ValueError('forget_fraction must be in (0, 1)')
    order = np.random.default_rng(seed).permutation(n)
    count = round(n * forget_fraction)
    return np.sort(order[:count]), np.sort(order[count:])
