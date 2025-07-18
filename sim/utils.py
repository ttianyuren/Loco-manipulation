import mink
import numpy as np


def configuration_reached(
    configuration: mink.Configuration,
    task: mink.Task,
    pos_threshold: float = 5e-3,
    ori_threshold: float = 5e-3,
) -> float:
    """Check if a desired configuration has been reached."""
    err = task.compute_error(configuration)
    return (
        np.linalg.norm(err[:3]) <= pos_threshold
        and np.linalg.norm(err[3:]) <= ori_threshold
    )
