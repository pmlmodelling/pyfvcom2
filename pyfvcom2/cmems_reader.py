"""Read CMEMS data"""

import numpy as np
import xarray as xr
from scipy import interpolate
from typing import Optional
from collections import namedtuple
from pyfvcom2.exceptions import PyFVCOM2ValueError


# Named tuple for mapping cmems variable names to fvcom variable names and the fvcom grid position
CMEMSVariableMap = namedtuple("CMEMSVariableMap", "cmems_name fvcom_name grid_position")


class CMEMSReader:
    """Class to read CMEMS Data files"""

    def __init__(
        self,
        file_path: str,
        reference_var_name: str,
        dimension_var_names: Optional[dict] = None,
    ):
        self.file_path = file_path

        print("Opening CMEMS file:", self.file_path)
        self.dataset = xr.open_dataset(self.file_path)

        # Set dimension variable names
        self.time_dim_name = (
            dimension_var_names.get("time", "time") if dimension_var_names else "time"
        )
        self.depth_dim_name = (
            dimension_var_names.get("depth", "depth")
            if dimension_var_names
            else "depth"
        )
        self.lon_dim_name = (
            dimension_var_names.get("longitude", "longitude")
            if dimension_var_names
            else "longitude"
        )
        self.lat_dim_name = (
            dimension_var_names.get("latitude", "latitude")
            if dimension_var_names
            else "latitude"
        )

        # Confirm dimension variable names exist in dataset. Assumes dimension and variable have same name.
        for dim_name in [self.time_dim_name, self.lon_dim_name, self.lat_dim_name]:
            if (
                dim_name not in self.dataset.dims
                and dim_name not in self.dataset.variables
            ):
                raise PyFVCOM2ValueError(
                    f"Dimension variable name {dim_name} not found in CMEMS file {self.file_path}"
                )

        # If reading 3D variables, check depth dimension exists
        self.has_depth_dimension = True
        if (
            self.depth_dim_name not in self.dataset.dims
            and self.depth_dim_name not in self.dataset.variables
        ):
            print(
                f"Depth dimension variable name {self.depth_dim_name} not found in CMEMS file {self.file_path}."
            )
            print(f"Assuming the dataset includes 2D variables only.")
            self.has_depth_dimension = False

        print("Using dimension variable names:")
        print(f"  Time: {self.time_dim_name}")
        if self.has_depth_dimension:
            print(f"  Depth: {self.depth_dim_name}")
        print(f"  Longitude: {self.lon_dim_name}")
        print(f"  Latitude: {self.lat_dim_name}")

        # Check reference var exists
        self.reference_var_name = reference_var_name
        print(f"Using reference variable {self.reference_var_name}.")

        if self.reference_var_name not in self.dataset.variables:
            raise PyFVCOM2ValueError(
                f"Reference variabe {self.reference_var_name} not found in dataset {file_path}."
            )

        # Check reference var dimensions
        if self.has_depth_dimension:
            if (
                self.depth_dim_name
                not in self.dataset.variables[self.reference_var_name].dims
            ):
                raise PyFVCOM2ValueError(
                    f"Please provide a 3D reference variable so the depth mask can be inferred. "
                    f"The supplied reference variable {self.reference_var_name} does not have a depth axis."
                )

        # Set masks
        self._set_masks()

        # Determine unmasked lon/lat points
        self._set_unmasked_lons_lats()

        # Store variable for bottom indices. Only compute this if it is needed.
        self._bottom_indices = None

    def _set_masks(self):
        """Use reference variable to infer the mask"""

        var = self.dataset[self.reference_var_name].isel({self.time_dim_name: 0})
        var_mask = self.get_mask(var)

        if not self.has_depth_dimension:
            reference_mask_3D = None
            reference_mask_2D = var_mask
        else:
            reference_mask_3D = var_mask

            # Set the 2D mask from the 3D mask (surface layer)
            reference_mask_2D = reference_mask_3D[0, :, :]

        # Save the mask
        self.mask_2D = reference_mask_2D
        self.mask_3D = reference_mask_3D

    def _set_unmasked_lons_lats(self):
        """Determine the unmasked longitude and latitude points.

        A 2D meshgrid is first formed from the 1D lon-lat variables. Unmasked
        lons and lats are then identified from this.
        """
        lons = self.dataset.variables[f"{self.lon_dim_name}"][:]
        lats = self.dataset.variables[f"{self.lat_dim_name}"][:]
        self._lon_grid, self._lat_grid = np.meshgrid(lons, lats)

        self._unmasked_lons = self._lon_grid[~self.mask_2D]
        self._unmasked_lats = self._lat_grid[~self.mask_2D]

    @property
    def n_depths(self):
        if not self.has_depth_dimension:
            raise PyFVCOM2ValueError("The dataset does not have a depth dimension.")

        return self.dataset.sizes[self.depth_dim_name]

    @property
    def lons(self):
        return self.dataset.variables[f"{self.lon_dim_name}"][:]

    @property
    def lats(self):
        return self.dataset.variables[f"{self.lat_dim_name}"][:]

    @property
    def lons_2D(self):
        return self._lon_grid

    @property
    def lats_2D(self):
        return self._lat_grid

    @property
    def unmasked_lons(self):
        return self._unmasked_lons

    @property
    def unmasked_lats(self):
        return self._unmasked_lats

    @property
    def depth_levels(self):
        if not self.has_depth_dimension:
            raise PyFVCOM2ValueError("The dataset does not have a depth dimension.")

        return -self.dataset.variables[f"{self.depth_dim_name}"][:].values

    def get_var_ndims(self, var_name: str) -> int:
        """Get the number of dimensions of a variable.

        Args:
            var_name (str): Variable name.

        Returns:
            int: Number of dimensions.
        """
        if var_name not in self.dataset.variables:
            raise PyFVCOM2ValueError(
                f"The supplied variable {var_name} is not in the dataset {self.file_path}"
            )

        var = self.dataset[var_name]
        return len(var.dims)

    def get_mask(self, var) -> np.ndarray:
        """Get the mask for a variable.
        Args:
            var (xarray.DataArray): Variable to get the mask for.
        Returns:
            np.ndarray: Boolean mask array, where True indicates a masked value.
        """
        arr = (
            var.values
        )  # materialise the array (may be numpy masked array, ndarray or dask array)

        # 1) If it's already a masked array
        if np.ma.is_masked(arr):
            var_mask = np.ma.getmaskarray(arr)
        else:
            # 2) If xarray decoded fill values to NaN (common default)
            var_mask = np.isnan(arr)

            # 3) If still no mask, try the _FillValue / missing_value fallback
            if not var_mask.any():
                fill = var.encoding.get("_FillValue", var.attrs.get("_FillValue", None))
                if fill is not None:
                    # use isclose for floats to avoid precision issues
                    var_mask = np.isclose(arr, fill)

        return var_mask

    def get_bottom_indices(self) -> np.ndarray:
        """Get indices of the deepest unmasked level for each horizontal point.

        Returns:
            np.ndarray: 2D array of bottom indices (lat, lon)
        """
        if self._bottom_indices is not None:
            return self._bottom_indices

        if self.mask_3D is None:
            raise PyFVCOM2ValueError(
                "3D variable mask not set. Does the output file contain 3D variables?"
            )

        # By setting it to zero, the surface level is always considered unmasked, even if that's a land point
        # TODO - use xarray to get lon/lat coords?
        bottom_indices = np.zeros(
            (self.mask_3D.shape[1], self.mask_3D.shape[2]), dtype=int
        )

        for j in range(self.mask_3D.shape[1]):
            for i in range(self.mask_3D.shape[2]):
                indices = np.where(self.mask_3D[:, j, i] == False)[0]

                # Applied to non-land points only
                if len(indices) != 0:
                    k = indices[-1]
                    bottom_indices[j, i] = k

        self._bottom_indices = bottom_indices

        return self._bottom_indices

    def get_var(
        self, var_name: str, time_index: int = 0, depth_index: int = None
    ) -> np.ndarray:
        """Get the values of a variable at a given time and depth index.

        Args:
            var_name (str): Variable name.
            time_index (int): Time index.
            depth_index (int, optional): Depth index for 3D variables. Defaults to None.
        Returns:
            np.ndarray: Variable values.
        """
        if var_name not in self.dataset.variables:
            raise PyFVCOM2ValueError(f"The supplied variable {var_name} is not in the dataset")

        if not self.has_depth_dimension:
            var = self.dataset[var_name].isel({self.time_dim_name: time_index})
            var_data = var.values
            return var_data

        else:
            if depth_index is None:
                raise PyFVCOM2ValueError("depth_index must be provided for 3D variables")
            var = self.dataset[var_name].isel(
                {self.time_dim_name: time_index, self.depth_dim_name: depth_index}
            )
            var_data = var.values
            return var_data

    def get_unmasked_variable(
        self, var_name: str, time_index: int = 0, depth_index: int = None
    ) -> np.ndarray:
        """Get the unmasked values of a variable at a given time and depth index.

        Args:
            var_name (str): Variable name.
            time_index (int): Time index.
            depth_index (int, optional): Depth index for 3D variables. Defaults to None.
        Returns:
            np.ndarray: Unmasked variable values.
        """
        if var_name not in self.dataset.variables:
            raise PyFVCOM2ValueError(f"The supplied variable {var_name} is not in the dataset")

        if not self.has_depth_dimension:
            var = self.dataset[var_name].isel({self.time_dim_name: time_index})
            var_data = var.values
            return var_data[~self.mask_2D]

        else:
            if depth_index is None:
                raise PyFVCOM2ValueError("depth_index must be provided for 3D variables")
            var = self.dataset[var_name].isel(
                {self.time_dim_name: time_index, self.depth_dim_name: depth_index}
            )
            var_data = var.values
            return var_data[~self.mask_2D]

    def get_filled_3D_var(self, var_name: str, time_index: int = 0) -> np.ndarray:
        """Fill masked values in a 3D variable by interpolation

        First, use griddata to interpolate over all masked surface values.
        Then, for each horizontal point, fill masked depth levels by
        downward extrapolation from the nearest unmasked depth level.

        Args:
            var_name (str): Variable name.
            time_index (int): Time index.

        Returns:
            np.ndarray: Filled variable values.
        """
        if var_name not in self.dataset.variables:
            raise PyFVCOM2ValueError(f"Variable {var_name} was not specified as a 3D variable")

        var = self.dataset[var_name].isel({self.time_dim_name: time_index})
        var_data = var.values  # shape (depth, lat, lon)

        # Create an array to hold filled data
        var_data_filled = np.empty_like(var_data)

        # First, fill the surface layer
        surface = var_data[0, :, :]
        surface_valid = surface[~self.mask_2D]

        # Assume data is unstructured and interpolate using griddata
        var_data_filled[0, :, :] = interpolate.griddata(
            (self._unmasked_lons, self._unmasked_lats),
            surface_valid,
            (self._lon_grid, self._lat_grid),
            method="linear",
        )

        # Copy in unmasked values for other depth levels
        var_data_filled[1:, :, :][~self.mask_3D[1:, :, :]] = var_data[1:, :, :][
            ~self.mask_3D[1:, :, :]
        ]

        # Now extraplolate downwards
        bottom_indices = self.get_bottom_indices()
        for j in range(var_data.shape[1]):
            for i in range(var_data.shape[2]):
                k = bottom_indices[j, i]
                var_data_filled[k:, j, i] = var_data_filled[k, j, i]

        return var_data_filled

    def close(self):
        """Close the dataset to free up resources."""
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None
