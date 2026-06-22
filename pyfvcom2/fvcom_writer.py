from netCDF4 import Dataset, date2num
import numpy as np


class FVCOMWriter(object):
    """ Create an FVCOM netCDF input file. """

    def __init__(self, filename, dimensions, global_attributes=None, **kwargs):
        """ Create a netCDF file.

        Parameters
        ----------
        filename : str, pathlib.Path
            Output netCDF path.
        dimensions : dict
            Dictionary of dimension names and sizes.
        global_attributes : dict, optional
            Global attributes to add to the netCDF file.

        Remaining arguments are passed to netCDF4.Dataset.

        """

        self.nc = Dataset(str(filename), 'w', **kwargs)

        for dimension in dimensions:
            self.nc.createDimension(dimension, dimensions[dimension])

        if global_attributes:
            for attribute in global_attributes:
                setattr(self.nc, attribute, global_attributes[attribute])

    def add_variable(self, name, data, dimensions, attributes=None, format='f4', ncopts={}):
        """
        Create a `name' variable with the given `attributes' and `data'.

        Parameters
        ----------
        name : str
            Variable name to add.
        data : np.ndararay, list, float, str
            Data to add to the netCDF file object.
        dimensions : list, tuple
            List of dimension names to apply to the new variable.
        attributes : dict, optional
            Attributes to add to the netCDF variable object.
        format : str, optional
            Data format for the new variable. Defaults to 'f4' (float32).
        ncopts : dict
            Dictionary of options to use when creating the netCDF variables.

        """

        var = self.nc.createVariable(name, format, dimensions, **ncopts)
        if attributes:
            for attribute in attributes:
                setattr(var, attribute, attributes[attribute])

        var[:] = data

        setattr(self, name, var)

    def write_fvcom_time(self, time, **kwargs):
        """
        Write the four standard FVCOM time variables (time, Times, Itime, Itime2) for the given time series.

        Parameters
        ----------
        time : np.ndarray, list, tuple
            Times as datetime objects.

        """

        mjd = date2num(time, units='days since 1858-11-17 00:00:00')
        Itime = np.floor(mjd)  # integer Modified Julian Days
        Itime2 = (mjd - Itime) * 24 * 60 * 60 * 1000  # milliseconds since midnight
        Times = [t.strftime('%Y-%m-%dT%H:%M:%S.%f') for t in time]

        # time
        atts = {'units': 'days since 1858-11-17 00:00:00',
                'format': 'modified julian day (MJD)',
                'long_name': 'time',
                'time_zone': 'UTC'}
        self.add_variable('time', mjd, ['time'], attributes=atts, **kwargs)
        # Itime
        atts = {'units': 'days since 1858-11-17 00:00:00',
                'format': 'modified julian day (MJD)',
                'time_zone': 'UTC'}
        self.add_variable('Itime', Itime, ['time'], attributes=atts, format='i', **kwargs)
        # Itime2
        atts = {'units': 'msec since 00:00:00', 'time_zone': 'UTC'}
        self.add_variable('Itime2', Itime2, ['time'], attributes=atts, format='i', **kwargs)
        # Times
        atts = {'long_name': 'Calendar Date', 'format': 'String: Calendar Time', 'time_zone': 'UTC'}
        self.add_variable('Times', Times, ['time', 'DateStrLen'], format='c', attributes=atts, **kwargs)


    def write_fvcom_grid(self, grid, subset_nodes: NDArray[np.bool_]=None,
                subset_elements: NDArray[np.bool_]=None, ncopts=None, depth:bool=True):
        """Add fvcom grid data

        Args:
            grid: Grid object from which to write the coordinates (can't type argument as circular dependency)
            subset_nodes: Boolean array to subset the nodes, if provided, otherwise all are written
            subset_elementss: Boolean array to subset the elements, if provided, otherwise all are written
            depth: Boolean, include the depth related variables (sigma layers, levels, h etc), defaults to true
        """

        if subset_nodes is None:
            subset_nodes = np.ones(len(grid.x_nodes), dtype=bool)
        if subset_elements is None:
            subset_elements = np.ones(len(grid.x_elements), dtype=bool)

        if ncopts is None:
            ncopts = {}

        atts = {'units': 'meters', 'long_name': 'nodal x-coordinate'}
        self.add_variable('x', grid.x_nodes[subset_nodes], ['node'],
                attributes=atts, ncopts=ncopts)

        atts = {'units': 'meters', 'long_name': 'nodal y-coordinate'}
        self.add_variable('y', grid.y_nodes[subset_nodes], ['node'],
                attributes=atts, ncopts=ncopts)

        atts = {'units': 'degrees_east', 'standard_name': 'longitude',
                'long_name': 'nodal longitude'}
        self.add_variable('lon', grid.lon_nodes[subset_nodes], ['node'],
                attributes=atts, ncopts=ncopts)

        atts = {'units': 'degrees_north', 'standard_name': 'latitude',
                'long_name': 'nodal latitude'}
        self.add_variable('lat', grid.lat_nodes[subset_nodes], ['node'],
                attributes=atts, ncopts=ncopts)

        atts = {'units': 'meters', 'long_name': 'zonal x-coordinate'}
        self.add_variable('xc', grid.x_elements[subset_elements], ['nele'],
                attributes=atts, ncopts=ncopts)

        atts = {'units': 'meters', 'long_name': 'zonal y-coordinate'}
        self.add_variable('yc', grid.y_elements[subset_elements], ['nele'],
                attributes=atts, ncopts=ncopts)

        atts = {'units': 'degrees_east', 'standard_name': 'longitude',
                'long_name': 'zonal longitude'}
        self.add_variable('lonc', grid.lon_elements[subset_elements],
                ['nele'], attributes=atts, ncopts=ncopts)

        atts = {'units': 'degrees_north', 'standard_name': 'latitude',
                'long_name': 'zonal latitude'}
        self.add_variable('latc', grid.lat_elements[subset_elements],
                ['nele'], attributes=atts, ncopts=ncopts)

        atts = {'long_name': 'nodes surrounding element'}
        nv = grid.triangles[subset_elements, :].T + 1  # FVCOM uses 1-based indexing

        self.add_variable('nv', nv,
                    ['three', 'nele'], format='i4', attributes=atts,
                    ncopts=ncopts)

        if depth:
            atts = {'long_name': 'Sigma Layers',
                    'standard_name': 'ocean_sigma/general_coordinate',
                    'positive': 'up',
                    'valid_min': -1.,
                    'valid_max': 0.,
                    'formula_terms': 'sigma: siglay eta: zeta depth: h'}
            self.add_variable('siglay', grid.sigma_layers[subset_nodes, :].T,
                    ['siglay', 'node'], attributes=atts, ncopts=ncopts)

            atts = {'long_name': 'Sigma Levels',
                    'standard_name': 'ocean_sigma/general_coordinate',
                    'positive': 'up',
                    'valid_min': -1.,
                    'valid_max': 0.,
                    'formula_terms': 'sigma: siglev eta: zeta depth: h'}
            self.add_variable('siglev', grid.sigma_levels[subset_nodes, :].T,
                    ['siglev', 'node'], attributes=atts, ncopts=ncopts)

            atts = {'long_name': 'Sigma Layers',
                    'standard_name': 'ocean_sigma/general_coordinate',
                    'positive': 'up',
                    'valid_min': -1.,
                    'valid_max': 0.,
                    'formula_terms': 'sigma: siglay_center eta: '
                    + 'zeta_center depth: h_center'}
            self.add_variable('siglay_center', grid.sigmac_layers[
                    subset_elements, :].T, ['siglay', 'nele'], attributes=atts,
                    ncopts=ncopts)

            atts = {'long_name': 'Sigma Levels',
                    'standard_name': 'ocean_sigma/general_coordinate',
                    'positive': 'up',
                    'valid_min': -1.,
                    'valid_max': 0.,
                    'formula_terms': 'sigma: siglev_center eta: '
                    + 'zeta_center depth: h_center'}
            self.add_variable('siglev_center', grid.sigmac_levels[
                    subset_elements, :].T, ['siglev', 'nele'], attributes=atts,
                    ncopts=ncopts)

            atts = {'long_name': 'Bathymetry',
                    'standard_name': 'sea_floor_depth_below_geoid',
                    'units': 'm',
                    'positive': 'down',
                    'grid': 'Bathymetry_mesh',
                    'coordinates': 'x y',
                    'type': 'data'}
            self.add_variable('h', grid.bathy_nodes[subset_nodes], ['node'],
                    attributes=atts, ncopts=ncopts)

            atts = {'long_name': 'Bathymetry',
                    'standard_name': 'sea_floor_depth_below_geoid',
                    'units': 'm',
                    'positive': 'down',
                    'grid': 'grid1 grid3',
                    'coordinates': 'latc lonc',
                    'grid_location': 'center'}
            self.add_variable('h_center', grid.bathy_elements[subset_elements],
                    ['nele'], attributes=atts, ncopts=ncopts)


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Tidy up the netCDF file handle. """
        self.nc.close()


