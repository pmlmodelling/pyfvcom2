"""Coordinate transformations for FVCOM model output"""

import numpy as np

__all__ = [
    "sigma_to_z_coords",
    "z_to_sigma_coords"
]


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
    # Vectorized computation using NumPy broadcasting
    # zeta and bathymetry are 1D arrays that will be broadcast across all levels
    h_plus_zeta = bathymetry + zeta

    # Broadcasting: zeta (1D) + (h_plus_zeta (1D) * sigma_coords (2D))
    # This computes all z coordinates in a single vectorized operation
    z_coords = zeta + h_plus_zeta * sigma_coords

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
    # Vectorized computation using NumPy broadcasting
    h_plus_zeta = bathymetry + zeta

    # Broadcasting: (z_coords (2D) - zeta (1D)) / h_plus_zeta (1D)
    # This computes all sigma coordinates in a single vectorized operation
    sigma_coords = (z_coords - zeta) / h_plus_zeta

    return sigma_coords
