"""Coordinate transformations for FVCOM model output"""

import numpy as np
from typing import Optional
import pyproj
from pyproj import CRS
from pyproj import Transformer
from pyproj.database import query_utm_crs_info
from pyproj.aoi import AreaOfInterest


from pyfvcom2.exceptions import PyFVCOM2ValueError, PyFVCOM2RuntimeError


__all__ = ["sigma_to_z_coords", "z_to_sigma_coords", "get_epsg_code", "utm_from_lonlat", "lonlat_from_utm"]


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


def get_epsg_code(lon: float, lat: float, datum: Optional[str] = "WGS 84") -> str:
    """Calculate EPSG code from longitude and latitude values.

    Args:
        lon (float): Longitude value.
        lat (float): Latitude value.
        datum (str, optional): The datum to use. Defaults to "WGS 84".

    Returns:
        str: The EPSG code.
    """
    a = AreaOfInterest(west_lon_degree=lon,
                       south_lat_degree=lat,
                       east_lon_degree=lon,
                       north_lat_degree=lat)

    utm_crs_list = query_utm_crs_info(datum_name=f"{datum}", area_of_interest=a)

    return utm_crs_list[0].code


def utm_from_lonlat(longitude, latitude, epsg_code: Optional = None):
    """Convert lats and lons to UTM coordinates using pyproj.

    East Longitudes are positive, west longitudes are negative. North latitudes
    are positive, south latitudes are negative. Lats and lons are in decimal
    degrees.

    The desired EPSG code can be found by searching websites such as
    epsg.io or spatialreference.org. If it is not provided, it will be calculated
    automatically from the first lon and lat values for the WGS 84 datum using
    the approach described here: https://tinyurl.com/aa83npwy.

    Be careful when using this function for coordinates that span multiple zones
    as the transformation is specific to the specified EPSG reference system.

    Args:
        longitude (int, float, tuple, list, np.ndarray): Longitudes. Can be a single value or array like.
        latitude (int, float, tuple, list, np.ndarray): Latitudes. Can be a single value or array like.
        epsg_code (int, str, optional): The EPSG code for the utm transformation.. E.g. 32630.

    Returns:
        tuple: A tuple containing:
            - eastings (numpy.ndarray): Eastings in the supplied reference system.
            - northings (numpy.ndarray): Northings in the supplied reference system.
            - epsg_code (int): The EPSG code for the utm transformation. E.g., 32630.
    """
    lon = np.asarray(longitude)
    lat = np.asarray(latitude)

    # Check the array sizes match
    if np.shape(lon) != np.shape(lat):
        raise PyFVCOM2RuntimeError('Lat and lon array sizes do not match')

    # Use the first longitude and latitude values to determine the EPSG code
    # if it has not been given.
    if epsg_code is None:
        # Handle scalar case by using .item() to extract scalar value
        first_lon = lon.item() if lon.ndim == 0 else lon.flat[0]
        first_lat = lat.item() if lat.ndim == 0 else lat.flat[0]
        epsg_code = get_epsg_code(first_lon, first_lat)
    else:
        epsg_code = int(epsg_code)

    utm_crs = CRS.from_epsg(epsg_code)

    proj = Transformer.from_crs(utm_crs.geodetic_crs, utm_crs, always_xy=True)

    eastings, northings = proj.transform(lon, lat)

    return np.asarray(eastings), np.asarray(northings), epsg_code


def lonlat_from_utm(eastings, northings, epsg_code: str):
    """Convert UTM coordinates to lat/lon.

    East Longitudes are positive, west longitudes are negative. North latitudes
    are positive, south latitudes are negative. Lat and Long are in decimal
    degrees.

    Args:
        eastings (int, float, tuple, list, np.ndarray): Eastings. Can be single values or array like.
        northings (int, float, tuple, list, np.ndarray): Northings. Can be single values or array like.
        epsg_code (int, str): The EPSG code for the utm transformation. E.g. 32630.

    Returns:
        tuple: A tuple containing:
            - lons (numpy.ndarray): Longitudes for the given eastings and northings.
            - lats (numpy.ndarray): Latitudes for the given eastings and northings.
    """

    # Handle scalar inputs by checking shapes instead of using len()
    if np.shape(eastings) != np.shape(northings):
        raise PyFVCOM2RuntimeError('Easting and northing array sizes do not match')

    epsg_code = int(epsg_code)
    utm_crs = CRS.from_epsg(epsg_code)

    proj = Transformer.from_crs(utm_crs.geodetic_crs, utm_crs, always_xy=True)

    eastings = np.asarray(eastings)
    northings = np.asarray(northings)

    lons, lats = proj.transform(eastings, northings, direction=pyproj.enums.TransformDirection.INVERSE)

    return np.asarray(lons), np.asarray(lats)
