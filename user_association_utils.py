import numpy as np
from map_utils import get_euclidean_distance

def search_closest_macro(device_pos, base_stations):
    """
    Finds the closest base station (macro or otherwise) to the given device.

    Parameters
    ----------
    device_pos : array-like of shape (2,)
        The (x, y) coordinates of the device.
    base_stations : np.ndarray
        Array of shape (N, 2) or (N, >=2), where each row contains [x, y, ...].

    Returns
    -------
    int
        Index of the closest base station.
    """
    min_dist = float('inf')
    closest_idx = 0

    for i, bs in enumerate(base_stations):
        dist = get_euclidean_distance(device_pos, bs[:2])
        if dist < min_dist:
            min_dist = dist
            closest_idx = i

    return closest_idx
