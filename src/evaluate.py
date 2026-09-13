import numpy as np

def forgetting_score(before: np.ndarray, after: np.ndarray) -> float:
    """Positive values mean lower post-unlearning confidence on forgotten labels."""
    return float(np.mean(before - after))
