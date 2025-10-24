"""FVCOM data reader for PyFVCOM2"""

__all__ = ["FVCOMReader"]

import numpy as np
from netCDF4 import Dataset


class FVCOMReader:
    """A class to read FVCOM model output and restart files.

    This class provides methods to read FVCOM netCDF files and extract relevant data.

    Attributes:
        filepath (str): Path to the FVCOM netCDF file.
        dataset (xarray.Dataset): The loaded FVCOM dataset.
    """

    def __init__(self, filepath):
        """Initialize the FVCOMReader with the path to the netCDF file.

        Args:
            filepath (str): Path to the FVCOM netCDF file.
        """
        self.filepath = filepath
        self.dataset = None

        # Load the dataset upon initialization
        self._load_data()

    def _load_data(self):
        """Load the FVCOM netCDF file into an xarray Dataset."""
        self.dataset = Dataset(self.filepath)

    @property
    def n_nodes(self):
        """Get the number of nodes in the FVCOM grid."""
        return self.dataset.dimensions["node"].size

    @property
    def n_elements(self):
        """Get the number of elements in the FVCOM grid."""
        return self.dataset.dimensions["nele"].size

    @property
    def n_sigma_layers(self):
        """Get the number of sigma layers in the FVCOM grid."""
        return self.dataset.dimensions["siglay"].size

    @property
    def n_sigma_levels(self):
        """Get the number of sigma levels in the FVCOM grid."""
        return self.dataset.dimensions["siglev"].size

    @property
    def lon_nodes(self):
        """Get the longitude values from the dataset."""
        return self.dataset.variables["lon"][:]

    @property
    def lat_nodes(self):
        """Get the latitude values from the dataset."""
        return self.dataset["lat"][:]

    @property
    def lon_elements(self):
        """Get the longitude values of element centroids."""
        return self.dataset["lonc"][:]

    @property
    def lat_elements(self):
        """Get the latitude values of element centroids."""
        return self.dataset["latc"][:]

    @property
    def bathy_nodes(self):
        """Get the bathymetry values at nodes (transformed so to be positive up)"""
        return self._return_variable_data("h") * -1.0

    @property
    def bathy_elements(self):
        """Get the bathymetry values at element centroids (transformed so to be positive up)"""
        return self._return_variable_data("h_center") * -1.0

    @property
    def sigma_layers_nodes(self):
        """Get the sigma layer values at nodes."""
        return self._return_variable_data("siglay")

    @property
    def sigma_layers_elements(self):
        """Get the sigma layer values at element centroids."""
        return self._return_variable_data("siglay_center")

    def _return_variable_data(self, var_name):
        """Return the data for a given variable name.

        Warn if the variable contains masked data for any reason.

        Args:
            var_name (str): The name of the variable to retrieve.
        Returns:
            np.ndarray: The data array for the specified variable.
        """
        if np.ma.is_masked(self.dataset["siglay_center"][:]):
            print(
                f"Warning: {var_name} contains masked data. "
                "Masked values will be filled with default fill value."
            )
        return np.ma.getdata(self.dataset.variables[var_name][:])

    def close(self):
        """Close the dataset to free up resources."""
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None
