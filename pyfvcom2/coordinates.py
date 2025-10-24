"""Coordinate transformations for FVCOM model output"""

import numpy as np

from pyfvcom2.exceptions import PyFVCOM2ValueError


__all__ = ["sigma_to_z_coords", "z_to_sigma_coords"]


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
    # Check array shapes
    if sigma_coords.ndim != 2:
        raise PyFVCOM2ValueError("sigma_coords must be a 2D array")
    if zeta.ndim != 1:
        raise PyFVCOM2ValueError("zeta must be a 1D array")
    if bathymetry.ndim != 1:
        raise PyFVCOM2ValueError("bathymetry must be a 1D array")
    if sigma_coords.shape[1] != zeta.shape[0]:
        raise PyFVCOM2ValueError(
            "Number of horizontal points in sigma_coords must match length of zeta"
        )
    if sigma_coords.shape[1] != bathymetry.shape[0]:
        raise PyFVCOM2ValueError(
            "Number of horizontal points in sigma_coords must match length of bathymetry"
        )

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
