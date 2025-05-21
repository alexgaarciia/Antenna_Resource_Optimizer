import numpy as np
from utils.map_utils import get_euclidean_distance

def compute_sinr_dl(p, base_stations, closest_idx, alpha, p_macro, p_femto, n_macro, noise, b):
    """
    Computes the downlink SINR at a user location given a list of base stations.

    Parameters
    ----------
    p : tuple of float
        User position (x, y).
    base_stations : np.ndarray
        Array of shape (N, 3) where each row is [x, y, power_weight].
    closest_idx : int
        Index of the serving base station.
    alpha : float
        Path loss exponent.
    p_macro : float
        Transmit power of macro cells.
    p_femto : float
        Transmit power of femto cells.
    n_macro : int
        Number of macro base stations (indices 0 to n_macro-1).
    noise : float
        Noise power (linear scale).
    b : float
        Rayleigh distribution parameter (used in fading, currently fixed).

    Returns
    -------
    sinr : float
        The downlink SINR in dB.
    """
    d_serving = get_euclidean_distance(p, base_stations[closest_idx][:2])
    power_serving = p_macro if closest_idx < n_macro else p_femto
    hx = 1  # deterministic fading (could use np.random.rayleigh(b))

    signal_db = 10 * np.log10(power_serving * hx * d_serving**(-alpha))

    final_interference = 0.0
    n_stations = base_stations.shape[0]

    for k in range(n_stations):
        if k == closest_idx:
            continue
        int_power = p_macro if k < n_macro else p_femto
        h = 1  # deterministic
        dist = get_euclidean_distance(p, base_stations[k][:2])
        final_interference += int_power * h * dist**(-alpha)

    sinr = signal_db - 10 * np.log10(final_interference + noise)
    return sinr
