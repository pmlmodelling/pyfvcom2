"""Coordinate transformations for FVCOM model output"""

import numpy as np


def sigma_to_z_coords(
    sigma_coords: np.ndarray, zeta: np.ndarray, bathymetry: np.ndarray
) -> np.ndarray:
    """Convert sigma to z coordinates.

    Args:
        sigma_coords (np.ndarray): 2D array of sigma coords (n_sigma, n_points).
        zeta (np.ndarray): 1D array of zeta values at each horizontal point.
        bathymetry (np.ndarray): 1D array of bathymetry values at each horizontal point.

    Returns:
        np.ndarray: 2D array of z coordinates (depth levels, horizontal points).
    """
    n_levels = sigma_coords.shape[0]
    n_points = len(zeta)

    z_coords = np.empty((n_levels, n_points), dtype=np.float32)

    for i in range(n_points):
        h = bathymetry[i]
        zet = zeta[i]
        for k in range(n_levels):
            sigma = sigma_coords[k, i]
            z = zet + (h + zet) * sigma
            z_coords[k, i] = z

    return z_coords


def z_to_sigma_coords(
    z_coords: np.ndarray, zeta: np.ndarray, bathymetry: np.ndarray
) -> np.ndarray:
    """Convert z to sigma coordinates.

    Args:
        z_coords (np.ndarray): 2D array of z coords (n_levels, n_points).
        zeta (np.ndarray): 1D array of zeta values at each horizontal point.
        bathymetry (np.ndarray): 1D array of bathymetry values at each horizontal point.

    Returns:
        np.ndarray: 2D array of sigma coordinates (n_levels, n_points).
    """
    n_levels = z_coords.shape[0]
    n_points = len(zeta)

    sigma_coords = np.empty((n_levels, n_points), dtype=np.float32)

    for i in range(n_points):
        h = bathymetry[i]
        zet = zeta[i]
        for k in range(n_levels):
            z = z_coords[k, i]
            sigma = (z - zet) / (h + zet)
            sigma_coords[k, i] = sigma

    return sigma_coords
