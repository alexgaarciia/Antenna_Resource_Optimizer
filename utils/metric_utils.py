import numpy as np

def compute_f_alpha(throughputs, alpha):
    """
    Compute the fairness-utility metric f_alpha for a given list of throughputs.

    Args:
        throughputs (list or np.ndarray): List of per-user throughputs (must be positive).
        alpha (float): Fairness parameter (e.g., 0 for linear, 1 for log, large for max-min).

    Returns:
        float: The value of the f_alpha utility.
    """
    throughputs = np.array(throughputs)

    if alpha == 1:
        utilities = np.log(throughputs)
    else:
        utilities = (throughputs ** (1 - alpha)) / (1 - alpha)

    return np.sum(utilities)
