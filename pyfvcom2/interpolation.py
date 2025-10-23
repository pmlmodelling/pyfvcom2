"""Interpolation functions"""

import numpy as np
from scipy import interpolate

from pyfvcom2.cmems_reader import CMEMSReader, CMEMSVariableMap
from pyfvcom2.fvcom_reader import FVCOMReader
from pyfvcom2.coordinates import sigma_to_z_coords
from pyfvcom2.exceptions import PyFVCOM2ValueError


def cmems_to_fvcom(
    cmems_reader: CMEMSReader,
    fvcom_reader: FVCOMReader,
    var_name_maps: list[CMEMSVariableMap],
    time_index: int,
) -> dict:
    """Interpolate CMEMS data onto FVCOM grid.

    Args:
        cmems_reader (CMEMSReader): An instance of CMEMSReader with loaded data.
        fvcom_reader (FVCOMReader): An instance of FVCOMReader with loaded data.
        var_names (CMEMSVariableMap): A mapping of variable names between CMEMS and FVCOM.
        time_index (int): The time index in the CMEMS data to use for the interpolation.

    Returns:
        dict: A dictionary containing interpolated variables on the FVCOM grid.
    """
    # Dict of interpolated data
    interpolated_data = {}

    # Establish whether the variables are 2D or 3D based on CMEMS data
    for var_name_map in var_name_maps:
        print(f"Interpolating CMEMS {var_name_map.cmems_name} to FVCOM grid...")
        cmems_var_name = var_name_map.cmems_name

        var_ndims = cmems_reader.get_var_ndims(cmems_var_name)

        # If a 2D spatial variable (time, lat, lon)
        if var_ndims == 3:
            interpolated_data[var_name_map.fvcom_name] = cmems_to_fvcom_2d(
                cmems_reader, fvcom_reader, var_name_map, time_index
            )
        # If a 3D spatial variable (time, depth, lat, lon)
        elif var_ndims == 4:
            interpolated_data[var_name_map.fvcom_name] = cmems_to_fvcom_3d(
                cmems_reader, fvcom_reader, var_name_map, time_index
            )

    return interpolated_data


def cmems_to_fvcom_2d(
    cmems_reader: CMEMSReader,
    fvcom_reader: FVCOMReader,
    var_name_map: CMEMSVariableMap,
    time_index: int,
) -> np.ndarray:
    """Interpolate a 2D CMEMS variable onto the FVCOM grid.

    Args:
        cmems_reader (CMEMSReader): An instance of CMEMSReader with loaded data.
        fvcom_reader (FVCOMReader): An instance of FVCOMReader with loaded data.
        var_name_map (CMEMSVariableMap): A mapping of variable names between CMEMS and FVCOM.
        time_index (int): The time index in the CMEMS data to use for the interpolation.

    Returns:
        np.ndarray: Interpolated variable on the FVCOM grid.
    """

    # Set FVCOM lats/lons depending on whether interpolating to nodes or elements
    if var_name_map.grid_position == "nodes":
        fvcom_lons = fvcom_reader.lon_nodes
        fvcom_lats = fvcom_reader.lat_nodes
    elif var_name_map.grid_position == "elements":
        fvcom_lons = fvcom_reader.lon_elements
        fvcom_lats = fvcom_reader.lat_elements
    else:
        raise PyFVCOM2ValueError(
            f"grid_position must be either 'nodes' or 'elements', received: {var_name_map.grid_position}"
        )

    # Get CMEMS unmasked lons/lats
    cmems_unmasked_lons = cmems_reader.unmasked_lons
    cmems_unmasked_lats = cmems_reader.unmasked_lats
    cmems_unmasked_data = cmems_reader.get_unmasked_variable(
        var_name_map.cmems_name, time_index
    )

    return interpolate.griddata(
        (cmems_unmasked_lons, cmems_unmasked_lats),
        cmems_unmasked_data,
        (fvcom_lons, fvcom_lats),
        method="linear",
    )


def cmems_to_fvcom_3d(
    cmems_reader: CMEMSReader,
    fvcom_reader: FVCOMReader,
    var_name_map: CMEMSVariableMap,
    time_index: int,
) -> np.ndarray:
    """Interpolate a 3D CMEMS variable onto the FVCOM grid.

    Args:
        cmems_reader (CMEMSReader): An instance of CMEMSReader with loaded data.
        fvcom_reader (FVCOMReader): An instance of FVCOMReader with loaded data.
        var_name_map (CMEMSVariableMap): A mapping of variable names between CMEMS and FVCOM.
        time_index (int): The time index in the CMEMS data to use for the interpolation.

    Returns:
        np.ndarray: Interpolated variable on the FVCOM grid.
    """

    # Set FVCOM lats/lons depending on whether interpolating to nodes or elements
    if var_name_map.grid_position == "nodes":
        n_points = fvcom_reader.n_nodes
        fvcom_lons = fvcom_reader.lon_nodes
        fvcom_lats = fvcom_reader.lat_nodes
        sigma_layers = fvcom_reader.sigma_layers_nodes
        bathy = fvcom_reader.bathy_nodes
    elif var_name_map.grid_position == "elements":
        n_points = fvcom_reader.n_elements
        fvcom_lons = fvcom_reader.lon_elements
        fvcom_lats = fvcom_reader.lat_elements
        sigma_layers = fvcom_reader.sigma_layers_elements
        bathy = fvcom_reader.bathy_elements
    else:
        raise PyFVCOM2ValueError(
            f"grid_position must be either 'nodes' or 'elements', received: {var_name_map.grid_position}"
        )

    # Save number of sigma layers
    n_sigma_layers = fvcom_reader.n_sigma_layers

    # Ignore temporal variations in zeta and set it to zero
    zeta = np.zeros_like(bathy)

    # Calculate depths in z coordinates
    z_layers = sigma_to_z_coords(sigma_layers, zeta, bathy)

    # Get the filled 3D CMEMS variable data
    var_filled = cmems_reader.get_filled_3D_var(var_name_map.cmems_name, time_index)

    # First, interpolate onto the horizontal grid for each depth level
    var_on_fvcom_horizontal_grid = np.empty(
        (cmems_reader.n_depths, n_points), dtype=var_filled.dtype
    )

    for depth_index in range(cmems_reader.n_depths):
        layer_data = var_filled[depth_index, :, :]

        interp = interpolate.RegularGridInterpolator(
            (cmems_reader.lons, cmems_reader.lats), layer_data.T
        )
        var_on_fvcom_horizontal_grid[depth_index, :] = interp((fvcom_lons, fvcom_lats))

    # Next, interpolate onto the FVCOM vertical sigma layers for each horizontal point
    var_on_fvcom_grid = np.empty((n_sigma_layers, n_points), dtype=var_filled.dtype)

    for i in range(n_points):
        var_profile = var_on_fvcom_horizontal_grid[:, i]
        target_depths = z_layers[:, i]

        interp = interpolate.interp1d(
            cmems_reader.depth_levels,
            var_profile,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        var_on_fvcom_grid[:, i] = interp(target_depths)

    return var_on_fvcom_grid
