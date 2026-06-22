from __future__ import annotations

import os
import numpy as np
from datetime import datetime
from typing import Optional

from pyfvcom2.fvcom_writer import FVCOMWriter

from .version import full_version
from .interpolation_coordinates import InterpolationCoordinates
from .interpolation import Interpolator
from .exceptions import PyFVCOM2ValueError

from metpy.calc import relative_humidity_from_dewpoint
from metpy.units import units

# Default mapping of fvcom variable names to ERA5 variables names
default_fvcom_to_era5_var_names = {
    'vwind_speed': 'v10',
    'uwind_speed': 'u10',
    'short_wave': 'avg_snswrf',
    'long_wave': 'avg_sdlwrf',
    'evap': 'e',
    'precip': 'tp',
    'cloud_cover': 'tcc',
    'air_pressure': 'sp',
    'air_temperature': 't2m',
    'relative_humidity': 'relative_humidity'
}

# Default node/element variables for surface forcing
default_fvcom_surface_vars = {
    'vwind_speed': 'element',
    'uwind_speed': 'element',
    'short_wave': 'node',
    'long_wave': 'node',
    'evap': 'node',
    'precip': 'node',
    'cloud_cover': 'node',
    'air_pressure': 'node',
    'air_temperature': 'node',
    'relative_humidity': 'node'
}


from .forcing_reader import RegularReader
class ERA5Reader(RegularReader):
    def __init__(
        self,
        file_path: Union[str, List[str]],
        reference_var_name: str,
        dimension_var_names: Optional[dict] = None,
    ):
        self._default_dims = {
            "time": "valid_time",
            "longitude": "longitude",
            "latitude": "latitude",
            "depth":"depth"
            }

        super().__init__(file_path, reference_var_name, dimension_var_names)
        # Need timestep length in seconds to convert accumulated variables (e.g. rain) to rates
        # We assume the timestp at the start of the metadata_dataset is valid for all data loaded
        try:
            self._input_timestep = (self._metadata_dataset['valid_time'][1] - self._metadata_dataset['valid_time'][0]).item() / 1e9
        except IndexError:
            raise PyFVCOM2ValueError("More than one timestep in first file required to be able to calculate timestep length")

    def get_var_ndims(self, var_name: str) -> int:
        """Get the number of dimensions of a variable.

        Args:
            var_name (str): Variable name.

        Returns:
            int: Number of dimensions.
        """
        if var_name == 'relative_humidity':
            return super().get_var_ndims('t2m')
        else:
            return super().get_var_ndims(var_name)


    def get_var(
        self, var_name: str, target_datetime: datetime, depth_index: int = None, tolerance=None
    ) -> np.ndarray:
        """Get the values of a variable at a given datetime and depth index.

        Args:
            var_name (str): Variable name.
            target_datetime (datetime): Target datetime to retrieve data for.
            depth_index (int, optional): Depth index for 3D variables. Defaults to None.
            tolerance (timedelta, optional): Maximum allowed time difference. Defaults to None.
        Returns:
            np.ndarray: Variable values.
        """
        if var_name == 'relative_humidity':
            temp   = self.get_var('t2m', target_datetime, depth_index, tolerance)*units.degC
            dewpoint_temp = self.get_var('d2m', target_datetime, depth_index, tolerance)*units.degC
            var_data = relative_humidity_from_dewpoint(temp, dewpoint_temp).to('percent')
        else:
            var_data = super().get_var(var_name, target_datetime, depth_index, tolerance)

        return self._era5_proc(var_data, var_name)

    # In the case of atmospheric data, there is no masking so these functions are all equivalent
    get_unmasked_variable = get_var
    get_filled_2D_var = get_var

    def _era5_proc(self, data: np.array, var_name: str):
        if var_name in ['t2m', 'd2m']:
            data = np.asarray((data*units.K).to('degC'))
        elif var_name in ['tp', 'e']:
            data = self._de_accumulate(data)
        return data

    def _de_accumulate(self, data):
        '''
        For hourly accumulated data
        '''
        return data/self._input_timestep


from .interpolation import RegularInterpolator
from scipy import interpolate
class ERA5Interpolator(RegularInterpolator):
    """ CMEMS interpolator class
    
    Args:
        cmems_reader (CMEMSReader): An instance of CMEMSReader with loaded data.
        fvcom_name_map (dict): A mapping of variable names between FVCOM and CMEMS.
        The keys are FVCOM variable names and the values are CMEMS variable names.
    
    """

    def __init__(self, era5_reader: ERA5Reader, fvcom_to_era5_var_names: Optional[dict] = None):
        super().__init__()

        self._model = "ERA5"
        self.source_reader = era5_reader

        if fvcom_to_era5_var_names is None:
            self.variable_mapping = default_fvcom_to_era5_var_names
        else:
            self.variable_mapping = fvcom_to_era5_var_names

    def _onelayer_interpolator(self, data):
        return interpolate.RegularGridInterpolator(
                (self.source_reader.lons, self.source_reader.lats), data)



class SurfaceManager:
    def __init__(self, grid: Grid):
        self._grid_ref = grid

        # Initialise empty list of dates on which to generate forcing data
        self._dates = []

        # Initialise empty dict to hold forcing data        
        self._forcing_data = {}


    def set_dates(self, dates: list[datetime]) -> None:
        """ Set the dates for the forcing data

        Args:
            dates: List of datetime objects.
        """
        if not self._forcing_data:
            print(f'Updating SurfaceManager dates and purging old forcing data for the previous dates.')

        self._forcing_data = {}
        self._dates = dates

    def get_forcing_data(self, variable: Optional[str] = None) -> dict | np.ndarray:
        """Get forcing data

        Args:
            variable: If given, return the array for that variable only.
                If None, return a dict of all forcing variables.

        Returns:
            Dictionary of variable name to array, or a single array when
            *variable* is specified.

        Raises:
            PyFVCOM2ValueError: If no forcing data has been added, or the
                requested variable is not available.
        """
        if not self._forcing_data:
            raise PyFVCOM2ValueError(
                "No forcing data available. Call add_forcing_data first."
            )

        if variable is not None:
            if variable not in self._forcing_data:
                raise PyFVCOM2ValueError(
                    f"Forcing data for '{variable}' not available. "
                    f"Available variables: {list(self._forcing_data.keys())}"
                )
            return self._forcing_data[variable]

        return self._forcing_data


    def add_forcing_data(self, interpolator: Interpolator, fvcom_var_name: str, horizontal_position: str) -> None:
        """ Add surface forcing data

        Args:
            interpolator: Interpolator instance to use for interpolation.
            fvcom_var_name: FVCOM name for the forcing variable.
            horizontal_position: Whether coordinates are at mesh nodes or element centres ('node' or 'element').
        """
        if not self._dates:
            raise PyFVCOM2ValueError(
                    f"No output dates set, call set_dates before calling add_forcing_data."
                )

        interpolation_coords = self._grid_ref.get_interpolation_coordinates(horizontal_position, 'layer_centre')
        interpolation_coords.dates = self._dates
        forcing_data = interpolator.interpolate(interpolation_coords, fvcom_var_name)
        self._forcing_data[fvcom_var_name] = forcing_data

    def add_standard_forcing(self, interpolator: Interpolator) -> None:
        """ Adds all the standard surface forcing variables from a single interpolator object 

        Args:
            interpolator: Interpolator instance to use for interpolation.
        """
        for var, location in default_fvcom_surface_vars.items():
            self.add_forcing_data(interpolator, var, location)

    def create_forcing_file(self, output_path: str,
                            format='NETCDF4',
                            ice_forcing: Optional[bool]=False,
                            **kwargs) -> None:

        """Write data to a FVCOM surface forcing file in NetCDF4 format

        Args:
            output_path: Path to the output NetCDF file.
            format: NetCDF format to use. Defaults to 'NETCDF4'.
            **kwargs: Additional keyword arguments for writing the forcing file.
        """

        ncfile = os.path.basename(output_path)

        globals = {'title': f'FVCOM surface forcing fyle',
                   'history': f'File created using PyFVCOM2 version {full_version}',
                   'filename': str(ncfile),
                   'Conventions': 'CF-1.0',
                   'source': 'FVCOM grid (unstructured) surface forcing'}

        # Dimensions
        dims = {'nele': self._grid_ref.n_elements, 'node': self._grid_ref.n_nodes, 'time': 0,
                'DateStrLen': 26, 'three': 3}

        # Options for compression etc.
        if 'ncopts' in kwargs:
            ncopts = kwargs.pop('ncopts')
        else:
            ncopts = {}

        with FVCOMWriter(str(ncfile), dims, global_attributes=globals,
                clobber=True, format=format, **kwargs) as write_ncfile:

            # Add standard times
            # ------------------
            write_ncfile.write_fvcom_time(self._dates, ncopts=ncopts)

            # Add space variables
            # -------------------
            print('Adding grid variables to netCDF')
            write_ncfile.write_fvcom_grid(self._grid_ref, depth=False)

            element_variables = {
                'uwind_speed': {'long_name': 'Eastward Wind Speed', 'units': 'm/s'},
                'vwind_speed': {'long_name': 'Eastward Wind Speed', 'units': 'm/s'}}


            node_variables = {
                'precip': {
                    'long_name': 'Precipitation',
                    'description': 'Precipitation, ocean lose water if negative',
                    'units': 'm s-1',
                    'positive': 'up'
                    },
                'evap': {
                    'long_name': 'Evaporation',
                    'description': 'Evaporation, ocean lose water is negative',
                    'units': 'm s-1',
                    'positive': 'up'
                    },
                'relative_humidity': {'long_name': 'Relative Humidity', 'units': '%'},
                'long_wave': {'long_name': 'Long Wave Radiation', 'units': 'W m-2'},
                'short_wave': {'long_name': 'Short Wave Radiation', 'units': 'W m-2'},
                'cloud_cover': {'long_name': 'Cloud Area Fraction', 'units': 'cloud covered fraction of sky [0,1]'},
                'air_pressure': {'long_name': 'Surface Air Pressure', 'units': 'Pa'},
                'air_temperature': {'long_name': 'Sea Surface Air Temperature', 'units': 'Degree (C)'}}

            if ice_forcing:
                node_variables |= {
                        'ice_cover':{'long_name':'Ice Cover', 'units':'[0-1]'},
                        'ice_thick':{'long_name':'Ice Thickness', 'units':'[m]'}}
       
            # Check all data is present
            for var in list(element_variables) + list(node_variables):
                if var not in self._forcing_data.keys():
                    raise PyFVCOM2ValueError(
                        f"Forcing data missing for {var}. Add forcing first.")

            for varname, atts in element_variables.items():
                write_ncfile.add_variable(varname, self._forcing_data[varname],
                    ['time', 'nele'], attributes=atts, ncopts=ncopts)

            for varname, atts in node_variables.items():
                write_ncfile.add_variable(varname, self._forcing_data[varname],
                    ['time', 'node'], attributes=atts, ncopts=ncopts)

