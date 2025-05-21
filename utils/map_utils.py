import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point

def apollonius_circle_path_loss(p1, p2, w1, w2, alpha):
    """
    Compute the Apollonius circle for path loss-based dominance.
    Given two points (base stations) and their power weights,
    returns the center and radius of the Apollonius circle.

    Parameters
    ----------
    p1 : array-like
        Coordinates of the first base station [x, y].
    p2 : array-like
        Coordinates of the second base station [x, y].
    w1 : float
        Power weight of the first base station.
    w2 : float
        Power weight of the second base station.
    alpha : float
        Path loss exponent.

    Returns
    -------
    cx : float
        X coordinate of the circle center.
    cy : float
        Y coordinate of the circle center.
    r : float
        Radius of the circle.
    """
    lam = (w1 / w2) ** (1 / alpha)
    cx = (p1[0] - p2[0] * lam**2) / (1 - lam**2)
    cy = (p1[1] - p2[1] * lam**2) / (1 - lam**2)
    r = lam * np.linalg.norm(np.array(p1) - np.array(p2)) / abs(1 - lam**2)
    return cx, cy, r

def get_circle(cx, cy, r, resolution=100):
    """
    Generate (x, y) coordinates of a circle for given center and radius.

    Parameters
    ----------
    cx, cy : float
        Coordinates of the center.
    r : float
        Radius of the circle.
    resolution : int
        Number of points for the circle perimeter.

    Returns
    -------
    x : np.ndarray
        X coordinates of the circle.
    y : np.ndarray
        Y coordinates of the circle.
    """
    th = np.linspace(0, 2 * np.pi, resolution)
    x = r * np.cos(th) + cx
    y = -r * np.sin(th) + cy  # Negative sign for orientation compatibility
    return x, y

def get_euclidean_distance(x, y):
    """
    Compute the Euclidean distance between two points.

    Parameters
    ----------
    x, y : array-like
        Coordinates of the two points.

    Returns
    -------
    float
        Euclidean distance.
    """
    return np.linalg.norm(np.array(x) - np.array(y))

def perpendicular_bisector(p1, p2):
    """
    Calculate the slope and intercepts of the perpendicular bisector
    between two points in 2D.

    Parameters
    ----------
    p1, p2 : array-like
        Points [x, y].

    Returns
    -------
    intercept_0 : float
        Y intercept at x=0.
    intercept_1 : float
        Y intercept at x=1.
    """
    xmed = (p1[0] + p2[0]) / 2
    ymed = (p1[1] + p2[1]) / 2
    # Slope of the perpendicular bisector
    slope = -(p2[0] - p1[0]) / (p2[1] - p1[1])
    intercept_0 = ymed - slope * xmed
    intercept_1 = slope * 1 + intercept_0
    return intercept_0, intercept_1

def get_dominance_area(p1, p2, limit):
    """
    Compute the dominance area polygon for one base station over another
    using the perpendicular bisector and map boundaries.

    Parameters
    ----------
    p1, p2 : array-like
        Coordinates of the two points (base stations).
    limit : float
        The map limit (boundary).

    Returns
    -------
    rx, ry : list of floats
        X and Y coordinates of the dominance polygon.
    """
    b0, b1 = perpendicular_bisector(p1, p2)

    poly_full = Polygon([(0, 0), (0, limit), (limit, limit), (limit, 0)])

    # Create a stripe polygon along the bisector line.
    x0, y0 = 0, b0
    x1, y1 = 1, b1
    dx, dy = x1 - x0, y1 - y0
    nx, ny = -dy, dx  # Perpendicular vector
    norm = (nx**2 + ny**2)**0.5
    nx /= norm
    ny /= norm
    offset = 1e5  # Large enough to cover the map

    # Large perpendicular strip polygon
    px = [x0 + nx*offset, x1 + nx*offset, x1 - nx*offset, x0 - nx*offset]
    py = [y0 + ny*offset, y1 + ny*offset, y1 - ny*offset, y0 - ny*offset]
    half_plane = Polygon(zip(px, py))
    clipped = poly_full.intersection(half_plane)

    # Determine which region contains the base station
    if not clipped.contains(Point(p1)):
        diff = poly_full.difference(clipped)
        # If diff is a MultiPolygon, select the largest area
        if isinstance(diff, MultiPolygon):
            largest = max(diff.geoms, key=lambda p: p.area)
            return list(largest.exterior.xy)
        elif isinstance(diff, Polygon):
            return list(diff.exterior.xy)
        else:
            return [[], []]  # Invalid or empty region
    else:
        return list(clipped.exterior.xy)

def recompute_regions(n_points, map_limit, base_stations, alpha_loss, to_remove=None):
    """
    Compute the region (coverage area) of each base station using dominance criteria.

    Parameters
    ----------
    n_points : int
        Number of base stations to consider.
    map_limit : float
        The map boundary (assumed square).
    base_stations : np.ndarray
        Array of shape (N, 3) with [x, y, power_weight] per row.
    alpha_loss : float
        Path loss exponent.
    to_remove : list of int, optional
        Indices of base stations to exclude (e.g. already chosen).

    Returns
    -------
    regions : list of shapely.Polygon
        List of region polygons (one per base station).
    """
    full_region = Polygon([(0,0), (0,map_limit), (map_limit,map_limit), (map_limit,0)])
    regions = []
    
    if to_remove is None:
        to_remove = []
    to_remove_set = set(to_remove)
    active_indices = [i for i in range(len(base_stations)) if i not in to_remove_set]
    base_stations = base_stations[active_indices]

    # Iterate in reverse for region partitioning as in original algorithm
    for k in reversed(range(len(base_stations))):
        region = full_region
        for j in range(len(base_stations)):
            if j >= k:
                continue
            # Use Apollonius circle if power differs, else use perpendicular bisector
            if base_stations[k][2] != base_stations[j][2]:
                cx, cy, r = apollonius_circle_path_loss(
                    base_stations[k][:2], base_stations[j][:2],
                    base_stations[k][2], base_stations[j][2], alpha_loss)
                circ_x, circ_y = get_circle(cx, cy, r)
                circle_poly = Polygon(zip(circ_x, circ_y))
                region = region.intersection(circle_poly)
            else:
                rx, ry = get_dominance_area(base_stations[k][:2], base_stations[j][:2], map_limit)
                bisector_poly = Polygon(zip(rx, ry))
                region = region.intersection(bisector_poly)
        regions.append(region)
        # Remove the claimed region from the remaining area
        full_region = full_region.difference(region)

    return list(reversed(regions))  # Restore the original order

def search_closest_bs(position, regions):
    """
    Find the index of the base station whose region contains the given position.

    Parameters
    ----------
    position : tuple of float
        The (x, y) coordinates of the query point (user position).
    regions : list of shapely.Polygon
        List of region polygons for all base stations.

    Returns
    -------
    int
        Index of the closest base station (first found).
    """
    p = Point(position)
    for i, region in enumerate(regions):
        if region.contains(p):
            return i
    return 0  # fallback to first base station if none matched
