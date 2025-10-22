"""FVCOM Reader"""

import numpy as np
import xarray as xr


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
        self.dataset = xr.open_dataset(self.filepath)

    @property
    def n_nodes(self):
        """Get the number of nodes in the FVCOM grid."""
        return self.dataset.dims["node"].size

    @property
    def n_elements(self):
        """Get the number of elements in the FVCOM grid."""
        return self.dataset.dims["nele"].size

    @property
    def n_sigma_layers(self):
        """Get the number of sigma layers in the FVCOM grid."""
        return self.dataset.dims["siglay"].size

    @property
    def n_sigma_levels(self):
        """Get the number of sigma levels in the FVCOM grid."""
        return self.dataset.dims["siglev"].size

    @property
    def lon_nodes(self):
        """Get the longitude values from the dataset."""
        return self.dataset["lon"].values

    @property
    def lat_nodes(self):
        """Get the latitude values from the dataset."""
        return self.dataset["lat"].values

    @property
    def lon_elements(self):
        """Get the longitude values of element centroids."""
        return self.dataset["lonc"].values

    @property
    def lat_elements(self):
        """Get the latitude values of element centroids."""
        return self.dataset["latc"].values

    @property
    def bathy_nodes(self):
        """Get the bathymetry values at nodes (transformed so to be positive up)"""
        return -self.dataset["h"].values

    @property
    def bathy_elements(self):
        """Get the bathymetry values at element centroids (transformed so to be positive up)"""
        return -self.dataset["h_center"].values

    @property
    def sigma_layers_nodes(self):
        """Get the sigma layer values at nodes."""
        return self.dataset["siglay"].values

    @property
    def sigma_layers_elements(self):
        """Get the sigma layer values at element centroids."""
        return self.dataset["siglay_center"].values
