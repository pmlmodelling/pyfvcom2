import os
import numpy as np
from datetime import datetime
from typing import Optional

from pyfvcom2.fvcom_writer import FVCOMWriter

from .version import full_version
from .interpolation_coordinates import InterpolationCoordinates
from .interpolation import Interpolator
from .weights_calculator import get_weights_calculator
from .grid import Grid, OpenBoundary
from .grid import find_connected_elements
from .coordinates import sigma_to_z_coords
from .ocean import zbar
from .exceptions import PyFVCOM2ValueError


__all__ = ["GridBand", "Nest", "NestManager"]


class GridBand:
    """Grid band with adjoining nodes and elements and vertical sigma structure.

    To be used when constructing an FVCOM nest.
    """
    
    def __init__(self, nodes: np.ndarray, elements: np.ndarray, 
                 node_weights: np.ndarray = None, element_weights: np.ndarray = None):
        """Initialize 3D GridBand.
        
        Args:
            nodes: Array of node coordinates.
            elements: Array of element connectivity.
            element_weights: Array of element weights.
            node_weights: Array of node weights.
        """
        self._nodes = nodes
        self._elements = elements
        self._node_weights = node_weights
        self._element_weights = element_weights

    @property
    def nodes(self) -> np.ndarray:
        """Get node coordinates.
        
        Returns:
            Array of node coordinates.
        """
        return self._nodes

    @property
    def elements(self) -> np.ndarray:
        """Get element connectivity.
        
        Returns:
            Array of element connectivity.
        """
        return self._elements

    @property
    def element_weights(self) -> np.ndarray:
        """Get element weights.
        
        Returns:
            Array of element weights.
        """
        return self._element_weights

    @property
    def node_weights(self) -> np.ndarray:
        """Get node weights.

        Returns:
            Array of node weights.
        """
        return self._node_weights


class Nest:
    """Representation of an FVCOM nest
    
    A nest consists of an open boundary and one or more grid bands which extend away
    from the open boundary into the interior of the domain.

    Attributes:
        open_boundary: OpenBoundary instance defining the nest's open boundary.
        open_boundary_weights: Weights for the open boundary nodes.
        grid_bands: List of GridBand instances defining the nest's grid bands.
    """

    def __init__(self, open_boundary: OpenBoundary):
        """Initialize Nest with an OpenBoundary and an optional list of GridBands."""
        self.open_boundary = open_boundary
        self.open_boundary_weights = np.ones((open_boundary.nnodes), dtype=np.float32)
        self.grid_bands = []

    def add_grid_band(self, grid_band: GridBand):
        """Add a GridBand to the nest.

        Args:
            grid_band: GridBand instance to add.
        """
        self.grid_bands.append(grid_band)

    def get_grid_bands(self) -> list:
        """Get the list of GridBands in the nest.

        Returns:
            List of GridBand instances.
        """
        return self.grid_bands


class NestManager:
    """Manager for FVCOM nests

    """
    def __init__(self, grid: Grid, num_grid_bands=1,
                 weights_calculation_method: str='linear'):
        self._grid_ref = grid

        # Method for calculating the weight of different grid bands
        self._weights_calculator = get_weights_calculator(weights_calculation_method)

        # Make nests
        self.nests = []
        self.make_nests(num_grid_bands)

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
            print(f'Updating NestManager dates and purging old forcing data for the previous dates.')

        self._forcing_data = {}
        self._dates = dates

    def clear_nests(self) -> None:
        """ Clear all nests from the manager """
        self.nests = []
        self.foricing_data = {}

    def make_nests(self, num_grid_bands: int) -> None:
        """ Make nests for each open boundary in the grid
        
        Args:
            grid: Grid instance containing open boundaries.
            num_grid_bands: Number of grid bands to create for each nest.
        """
        if len(self.nests) > 0:
            # Purge existing nests
            print("Purging existing nests from NestManager")
            self.clear_nests()

        # Keep track of all nodes and elements used in the nest so far to avoid duplication
        all_nodes = []
        all_elements = []

        # If there are no open boundaries, flag that no nests can be created
        if len(self._grid_ref.open_boundaries) == 0:
            print("Warning: No open boundaries found in the grid. No nests can be created.")
            return

        # Create new nest objects for each open boundary
        for ob in self._grid_ref.open_boundaries:
            nest = Nest(open_boundary=ob)

            # Add new OB nodes to all nodes
            new_nodes = np.setdiff1d(ob.node_indices, all_nodes).tolist()
            all_nodes.extend(new_nodes)
            
            # Set of nodes that are used to locate elements in the next grid band. When beginning
            # to add grid bands, as we are here, we start with the open boundary nodes. As we successively
            # add grid bands, we update this to be the nodes in the most recently added grid band.
            reference_nodes = new_nodes

            for i in range(num_grid_bands):
                # First, find the elements that make up the grid band adjoining the current, inner
                # most set of nodes in the nest. If no grid bands have been added yet, this will
                # be the nodes that define the open boundary.
                elements = find_connected_elements(reference_nodes, self._grid_ref.triangles)

                # Only use unique elements that have not already been used in previous grid bands
                unique_elements = np.setdiff1d(elements, all_elements).tolist()

                # Get the nodes connected to the elements we've extracted.
                nodes = np.unique(self._grid_ref.triangles[unique_elements, :])

                # Remove ones we already have in the nest.
                unique_nodes = np.setdiff1d(nodes, all_nodes).tolist()

                # Calculate weights for the nodes and elements in this grid band
                node_weights = self._weights_calculator.calculate_weights(
                    n=len(unique_nodes),
                    n_bands=num_grid_bands,
                    grid_band_index=i+1 # +1 as open boundary is band 0
                )
                element_weights = self._weights_calculator.calculate_weights(
                    n=len(unique_elements),
                    n_bands=num_grid_bands,
                    grid_band_index=i
                )

                # Create the grid band
                grid_band = GridBand(unique_nodes, unique_elements, node_weights, element_weights)
                nest.add_grid_band(grid_band)

                # Update reference vars
                reference_nodes = unique_nodes

                # Keep track of all nodes and elements used in the nest so far
                all_nodes.extend(unique_nodes)
                all_elements.extend(unique_elements)

            # Add nest
            self.add_nest(nest)

    def add_nest(self, nest: Nest):
        """Add a Nest to the manager.

        Args:
            nest: Nest instance to add.
        """
        self.nests.append(nest)

    def get_nests(self) -> list:
        """Get the list of Nests managed by the manager.

        Returns:
            List of Nest instances.
        """
        return self.nests

    def get_all_nest_nodes(self) -> np.ndarray:
        """Get all unique node indices used in all nests.

        Returns:
            Array of unique node indices.
        """
        all_nodes = []
        for nest in self.nests:
            # Add open boundary nodes
            all_nodes.extend(nest.open_boundary.node_indices.tolist())

            # Add nodes from each grid band
            for band in nest.grid_bands:
                all_nodes.extend(band.nodes)

        # Return unique nodes as a numpy array
        return np.array(all_nodes, dtype=np.int32)

    def get_all_nest_elements(self) -> np.ndarray:
        """Get all unique element indices used in all nests.

        Returns:
            Array of unique element indices.
        """
        all_elements = []
        for nest in self.nests:
            # Add elements from each grid band
            for band in nest.grid_bands:
                all_elements.extend(band.elements)

        # Return unique elements as a numpy array
        return np.array(all_elements, dtype=np.int32)
    
    def get_all_node_weights(self) -> np.ndarray:
        """Get all node weights for all nests.

        Returns:
            Array of node weights.
        """
        all_node_weights = []
        for nest in self.nests:
            # Add open boundary weights
            all_node_weights.extend(nest.open_boundary_weights.tolist())

            # Add weights from each grid band
            for band in nest.grid_bands:
                all_node_weights.extend(band.node_weights.tolist())

        return np.array(all_node_weights, dtype=np.float32)
    
    def get_all_element_weights(self) -> np.ndarray:
        """Get all element weights for all nests.

        Returns:
            Array of element weights.
        """
        all_element_weights = []
        for nest in self.nests:
            # Add weights from each grid band
            for band in nest.grid_bands:
                all_element_weights.extend(band.element_weights.tolist())

        return np.array(all_element_weights, dtype=np.float32)

    def get_interpolation_coordinates(self, horizontal_position: str,
                                      vertical_position: str,
                                      horizontal_coordinate_system: Optional[str] = "geographic",
                                      vertical_coordinate_system: Optional[str] = "z") -> InterpolationCoordinates:
        """Get interpolation coordinates for a specific grid position.

        Args:
            horizontal_position: Whether coordinates are at mesh nodes or element centres ('node' or 'element').
            vertical_position: Whether depth coordinates are at layer centres or layer interfaces
                ('layer_centre' or 'layer_interface').
            horizontal_coordinate_system: The coordinate system ("geographic" or "cartesian") for the interpolation coordinates.
            vertical_coordinate_system: The vertical coordinate system ("z" or "sigma") for the interpolation coordinates.

        Returns:
            InterpolationCoordinates: The interpolation coordinates for the specified grid position.
        """
        if horizontal_position not in ['node', 'element']:
          raise PyFVCOM2ValueError("horizontal_position must be either 'node' or 'element'")

        if vertical_position not in ['layer_centre', 'layer_interface']:
            raise PyFVCOM2ValueError("vertical_position must be either 'layer_centre' or 'layer_interface'")

        if horizontal_coordinate_system not in ["geographic", "cartesian"]:
            raise PyFVCOM2ValueError("horizontal_coordinate_system must be either 'geographic' or 'cartesian'")

        if vertical_coordinate_system not in ["z", "sigma"]:
            raise PyFVCOM2ValueError("vertical_coordinate_system must be either 'z' or 'sigma'")

        if horizontal_position == 'node':
            indices = self.get_all_nest_nodes()
            if horizontal_coordinate_system == "geographic":
                x1 = self._grid_ref.lon_nodes[indices]
                x2 = self._grid_ref.lat_nodes[indices]
            else:  # cartesian
                x1 = self._grid_ref.x[indices]
                x2 = self._grid_ref.y[indices]
            if vertical_position == 'layer_interface':
                if vertical_coordinate_system == 'z':
                    x3 = self._grid_ref.sigma_levels_z[indices, :].T
                else:  # 'sigma'
                    x3 = self._grid_ref.sigma_levels[indices, :].T
            else:  # 'layer_centre'
                if vertical_coordinate_system == 'z':
                    x3 = self._grid_ref.sigma_layers_z[indices, :].T
                else:  # 'sigma'
                    x3 = self._grid_ref.sigma_layers[indices, :].T
        else:  # horizontal_position == 'element'
            indices = self.get_all_nest_elements()
            if horizontal_coordinate_system == "geographic":
                x1 = self._grid_ref.lon_elements[indices]
                x2 = self._grid_ref.lat_elements[indices]
            else:  # cartesian
                x1 = self._grid_ref.xc[indices]
                x2 = self._grid_ref.yc[indices]
            if vertical_position == 'layer_interface':
                if vertical_coordinate_system == 'z':
                    x3 = self._grid_ref.sigmac_levels_z[indices, :].T
                else:  # 'sigma'
                    x3 = self._grid_ref.sigmac_levels[indices, :].T
            else:  # 'layer_centre'
                if vertical_coordinate_system == 'z':
                    x3 = self._grid_ref.sigmac_layers_z[indices, :].T
                else:  # 'sigma'
                    x3 = self._grid_ref.sigmac_layers[indices, :].T

        return InterpolationCoordinates(self._dates, x3, x2, x1, horizontal_coordinate_system=horizontal_coordinate_system, vertical_coordinate_system=vertical_coordinate_system)

    def add_forcing_data(self, interpolator: Interpolator, fvcom_var_name: str, horizontal_position: str) -> None:
        """ Add forcing data for the nests

        Args:
            interpolator: Interpolator instance to use for interpolation.
            fvcom_var_name: FVCOM name for the forcing variable.
            horizontal_position: Whether coordinates are at mesh nodes or element centres ('node' or 'element').
        """
        interpolation_coords = self.get_interpolation_coordinates(horizontal_position, 'layer_centre')
        forcing_data = interpolator.interpolate(interpolation_coords, fvcom_var_name)
        self._forcing_data[fvcom_var_name] = forcing_data

        # For velocities, calculate the vertically averaged component
        if fvcom_var_name in ['u', 'v']:
            indices = self.get_all_nest_elements()
            layer_thickness = (self._grid_ref.sigma_levels.T[0:-1, indices] - self._grid_ref.sigma_levels.T[1:, indices])
            self._forcing_data[f'{fvcom_var_name}a'] = zbar(forcing_data, layer_thickness)

    def create_forcing_file(self, output_path: str, nest_type: int, format='NETCDF4', **kwargs) -> None:
        """ Write the nest forcing data to a NetCDF file

        Args:
            output_path: Path to the output NetCDF file.
            nest_type: Type of model nesting.
            format: NetCDF format to use. Defaults to 'NETCDF4'.
            **kwargs: Additional keyword arguments for writing the forcing file.
        """
        ncfile = os.path.basename(output_path)

        nodes = self.get_all_nest_nodes()
        elements = self.get_all_nest_elements()
        
        # Counters
        n_dates = len(self._dates)
        n_nodes = len(nodes)
        n_elements = len(elements)
        n_sigma_layers = self._grid_ref.n_sigma_layers
        n_sigma_levels = self._grid_ref.n_sigma_levels

        # Prepare the data.
        hyw = np.zeros((n_dates, n_sigma_levels, n_nodes)) # Always zero, for now at least

        # Set weights if bdy type is 3 
        if nest_type == 3:
            node_weights = self.get_all_node_weights()
            node_weights = np.tile(node_weights, [n_dates, 1])

            element_weights = self.get_all_element_weights()
            element_weights = np.tile(element_weights, [n_dates, 1])
        else:
            node_weights = None
            element_weights = None

        # Options for compression etc.
        if 'ncopts' in kwargs:
            ncopts = kwargs.pop('ncopts')
        else:
            ncopts = {}

        # Define the global attributes
        globals = {'type': 'FVCOM nestING TIME SERIES FILE',
                   'title': f'FVCOM nestING TYPE {nest_type} TIME SERIES data for open boundary',
                   'history': f'File created using PyFVCOM2 version {full_version}',
                   'filename': str(ncfile),
                   'Conventions': 'CF-1.0'}

        # Dimensions
        dims = {'nele': n_elements, 'node': n_nodes, 'time': 0, 
                'DateStrLen': 26, 'three': 3,
                'siglay': n_sigma_layers, 'siglev': n_sigma_levels}

        with FVCOMWriter(str(ncfile), dims, global_attributes=globals, 
                clobber=True, format=format, **kwargs) as nest_ncfile:

            # Add standard times
            # ------------------
            nest_ncfile.write_fvcom_time(self._dates, ncopts=ncopts)

            # Add space variables
            # -------------------
            atts = {'units': 'meters', 'long_name': 'nodal x-coordinate'}
            nest_ncfile.add_variable('x', self._grid_ref.x[nodes], ['node'], 
                    attributes=atts, ncopts=ncopts)

            atts = {'units': 'meters', 'long_name': 'nodal y-coordinate'}
            nest_ncfile.add_variable('y', self._grid_ref.y[nodes], ['node'], 
                    attributes=atts, ncopts=ncopts)

            atts = {'units': 'degrees_east', 'standard_name': 'longitude', 
                    'long_name': 'nodal longitude'}
            nest_ncfile.add_variable('lon', self._grid_ref.lon[nodes], ['node'], 
                    attributes=atts, ncopts=ncopts)

            atts = {'units': 'degrees_north', 'standard_name': 'latitude', 
                    'long_name': 'nodal latitude'}
            nest_ncfile.add_variable('lat', self._grid_ref.lat[nodes], ['node'], 
                    attributes=atts, ncopts=ncopts)

            atts = {'units': 'meters', 'long_name': 'zonal x-coordinate'}
            nest_ncfile.add_variable('xc', self._grid_ref.xc[elements], ['nele'], 
                    attributes=atts, ncopts=ncopts)

            atts = {'units': 'meters', 'long_name': 'zonal y-coordinate'}
            nest_ncfile.add_variable('yc', self._grid_ref.yc[elements], ['nele'], 
                    attributes=atts, ncopts=ncopts)

            atts = {'units': 'degrees_east', 'standard_name': 'longitude', 
                    'long_name': 'zonal longitude'}
            nest_ncfile.add_variable('lonc', self._grid_ref.lonc[elements], 
                    ['nele'], attributes=atts, ncopts=ncopts)

            atts = {'units': 'degrees_north', 'standard_name': 'latitude', 
                    'long_name': 'zonal latitude'}
            nest_ncfile.add_variable('latc', self._grid_ref.latc[elements], 
                    ['nele'], attributes=atts, ncopts=ncopts)

            atts = {'long_name': 'nodes surrounding element'}
            nv = self._grid_ref.triangles[elements, :].T + 1  # FVCOM uses 1-based indexing
            nest_ncfile.add_variable('nv', nv, 
                    ['three', 'nele'], format='i4', attributes=atts, 
                    ncopts=ncopts)

            atts = {'long_name': 'Sigma Layers',
                    'standard_name': 'ocean_sigma/general_coordinate',
                    'positive': 'up',
                    'valid_min': -1.,
                    'valid_max': 0.,
                    'formula_terms': 'sigma: siglay eta: zeta depth: h'}
            nest_ncfile.add_variable('siglay', self._grid_ref.sigma_layers[nodes, :].T, 
                    ['siglay', 'node'], attributes=atts, ncopts=ncopts)

            atts = {'long_name': 'Sigma Levels',
                    'standard_name': 'ocean_sigma/general_coordinate',
                    'positive': 'up',
                    'valid_min': -1.,
                    'valid_max': 0.,
                    'formula_terms': 'sigma: siglev eta: zeta depth: h'}
            nest_ncfile.add_variable('siglev', self._grid_ref.sigma_levels[nodes, :].T, 
                    ['siglev', 'node'], attributes=atts, ncopts=ncopts)

            atts = {'long_name': 'Sigma Layers',
                    'standard_name': 'ocean_sigma/general_coordinate',
                    'positive': 'up',
                    'valid_min': -1.,
                    'valid_max': 0.,
                    'formula_terms': 'sigma: siglay_center eta: '
                    + 'zeta_center depth: h_center'}
            nest_ncfile.add_variable('siglay_center', self._grid_ref.sigmac_layers[
                    elements, :].T, ['siglay', 'nele'], attributes=atts, 
                    ncopts=ncopts)

            atts = {'long_name': 'Sigma Levels',
                    'standard_name': 'ocean_sigma/general_coordinate',
                    'positive': 'up',
                    'valid_min': -1.,
                    'valid_max': 0.,
                    'formula_terms': 'sigma: siglev_center eta: '
                    + 'zeta_center depth: h_center'}
            nest_ncfile.add_variable('siglev_center', self._grid_ref.sigmac_levels[
                    elements, :].T, ['siglev', 'nele'], attributes=atts, 
                    ncopts=ncopts)

            atts = {'long_name': 'Bathymetry',
                    'standard_name': 'sea_floor_depth_below_geoid',
                    'units': 'm',
                    'positive': 'down',
                    'grid': 'Bathymetry_mesh',
                    'coordinates': 'x y',
                    'type': 'data'}
            nest_ncfile.add_variable('h', self._grid_ref.h[nodes], ['node'], 
                    attributes=atts, ncopts=ncopts)

            atts = {'long_name': 'Bathymetry',
                    'standard_name': 'sea_floor_depth_below_geoid',
                    'units': 'm',
                    'positive': 'down',
                    'grid': 'grid1 grid3',
                    'coordinates': 'latc lonc',
                    'grid_location': 'center'}
            nest_ncfile.add_variable('h_center', self._grid_ref.hc[elements], 
                    ['nele'], attributes=atts, ncopts=ncopts)

            if nest_type == 3:
                atts = {'long_name': 'Weights for nodes in relaxation zone',
                        'units': 'no units',
                        'grid': 'fvcom_grid',
                        'type': 'data'}
                nest_ncfile.add_variable('weight_node', node_weights, 
                        ['time', 'node'], attributes=atts, ncopts=ncopts)

                atts = {'long_name': 'Weights for elements in relaxation zone',
                        'units': 'no units',
                        'grid': 'fvcom_grid',
                        'type': 'data'}
                nest_ncfile.add_variable('weight_cell', element_weights, 
                        ['time', 'nele'], attributes=atts, ncopts=ncopts)

            # Now all the data
            # ----------------

            atts = {'long_name': 'Water Surface Elevation',
                    'units': 'meters',
                    'positive': 'up',
                    'standard_name': 'sea_surface_height_above_geoid',
                    'grid': 'Bathymetry_Mesh',
                    'coordinates': 'time lat lon',
                    'type': 'data',
                    'location': 'node'}
            nest_ncfile.add_variable('zeta', self._forcing_data['zeta'], 
                    ['time', 'node'], attributes=atts, ncopts=ncopts)

            atts = {'long_name': 'Vertically Averaged x-velocity',
                    'units': 'meters  s-1',
                    'grid': 'fvcom_grid',
                    'type': 'data'}
            nest_ncfile.add_variable('ua', self._forcing_data['ua'], 
                    ['time', 'nele'], attributes=atts, ncopts=ncopts)

            atts = {'long_name': 'Vertically Averaged y-velocity',
                    'units': 'meters  s-1',
                    'grid': 'fvcom_grid',
                    'type': 'data'}
            nest_ncfile.add_variable('va', self._forcing_data['va'], 
                    ['time', 'nele'], attributes=atts, ncopts=ncopts)

            atts = {'long_name': 'Eastward Water Velocity',
                    'units': 'meters  s-1',
                    'standard_name': 'eastward_sea_water_velocity',
                    'grid': 'fvcom_grid',
                    'coordinates': 'time siglay latc lonc',
                    'type': 'data',
                    'location': 'face'}
            nest_ncfile.add_variable('u', self._forcing_data['u'], 
                    ['time', 'siglay', 'nele'], attributes=atts, ncopts=ncopts)

            atts = {'long_name': 'Northward Water Velocity',
                    'units': 'meters  s-1',
                    'standard_name': 'Northward_sea_water_velocity',
                    'grid': 'fvcom_grid',
                    'coordinates': 'time siglay latc lonc',
                    'type': 'data',
                    'location': 'face'}
            nest_ncfile.add_variable('v', self._forcing_data['v'], 
                    ['time', 'siglay', 'nele'], attributes=atts, ncopts=ncopts)

            atts = {'long_name': 'Temperature',
                    'standard_name': 'sea_water_temperature',
                    'units': 'degrees Celcius',
                    'grid': 'fvcom_grid',
                    'coordinates': 'time siglay lat lon',
                    'type': 'data',
                    'location': 'node'}
            nest_ncfile.add_variable('temp', self._forcing_data['temp'], 
                    ['time', 'siglay', 'node'], attributes=atts, ncopts=ncopts)

            atts = {'long_name': 'Salinity',
                    'standard_name': 'sea_water_salinity',
                    'units': '1e-3',
                    'grid': 'fvcom_grid',
                    'coordinates': 'time siglay lat lon',
                    'type': 'data',
                    'location': 'node'}
            nest_ncfile.add_variable('salinity', self._forcing_data['salinity'], 
                    ['time', 'siglay', 'node'], attributes=atts, ncopts=ncopts)

            atts = {'long_name': 'hydro static vertical velocity',
                    'units': 'meters s-1',
                    'grid': 'fvcom_grid',
                    'type': 'data',
                    'coordinates': 'time siglev lat lon'}
            nest_ncfile.add_variable('hyw', hyw, 
                    ['time', 'siglev', 'node'], attributes=atts, ncopts=ncopts)
