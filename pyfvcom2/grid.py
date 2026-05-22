from __future__ import annotations

import numpy as np
from typing import Optional

from .mesh_reader import MeshData, read_mesh_file
from .sigma import SigmaConfig, SigmaData, process_sigma_config, read_sigma_file, write_sigma_file
from .coordinates import lonlat_from_utm, utm_from_lonlat, sigma_to_z_coords
from .exceptions import PyFVCOM2ValueError
from .interpolation_coordinates import InterpolationCoordinates
from .fvcom_writer import FVCOMWriter

__all__ = ["OpenBoundary", "Grid", "connectivity", "nodes2elems", "find_connected_elements"]


class OpenBoundary:
    """Represents an open boundary in the mesh.

    Args:
        bdy_id: Unique identifier for the boundary.
        node_indices: Array of node indices that form this boundary.
        sigma_levels: Sigma level coordinates for boundary nodes.
        sigma_layers: Sigma layer coordinates for boundary nodes.
    """
    
    def __init__(self, bdy_id: int, node_indices: np.ndarray,
                 sigma_levels: np.ndarray, sigma_layers: np.ndarray):
        self._bdy_id = bdy_id
        self._node_indices = node_indices
        self._sigma_levels = sigma_levels
        self._sigma_layers = sigma_layers
        self._sponge_radius: Optional[np.ndarray] = None
        self._sponge_coefficient: Optional[np.ndarray] = None

    @property
    def bdy_id(self) -> int:
        """Get the boundary ID.
        
        Returns:
            Unique identifier for this boundary.
        """
        return self._bdy_id

    @property
    def nnodes(self) -> int:
        """Get the number of nodes in this boundary.
        
        Returns:
            Number of nodes that form this boundary.
        """
        return len(self._node_indices)

    @property
    def node_indices(self) -> np.ndarray:
        """Get the node indices for this boundary.
        
        Returns:
            Array of node indices that form this boundary.
        """
        return self._node_indices

    @property
    def sigma_levels(self) -> np.ndarray:
        """Get sigma levels for boundary nodes.
        
        Returns:
            Sigma level coordinates for boundary nodes.
        """
        return self._sigma_levels

    @property
    def sigma_layers(self) -> np.ndarray:
        """Get sigma layers for boundary nodes.

        Returns:
            Sigma layer coordinates for boundary nodes.
        """
        return self._sigma_layers

    @property
    def sponge_radius(self) -> Optional[np.ndarray]:
        """Sponge layer radius at each boundary node (m), or None if not set."""
        return self._sponge_radius

    @property
    def sponge_coefficient(self) -> Optional[np.ndarray]:
        """Sponge layer coefficient at each boundary node, or None if not set."""
        return self._sponge_coefficient

    def set_sponge(self, radius: float | np.ndarray,
                   coefficient: float | np.ndarray) -> None:
        """Set the sponge layer parameters for all nodes on this boundary.

        A scalar radius or coefficient is broadcast to every node on the
        boundary. Per-node arrays must have length equal to ``nnodes``.

        Args:
            radius: Sponge layer radius in metres. A single float applies the
                same value to every node; an array must match ``nnodes``.
            coefficient: Sponge layer coefficient (dimensionless relaxation
                strength). Same broadcasting rules as *radius*.

        Raises:
            PyFVCOM2ValueError: If an array argument does not match ``nnodes``.
        """
        n = self.nnodes
        for name, val in (('radius', radius), ('coefficient', coefficient)):
            if np.isscalar(val):
                val = np.full(n, float(val))
            else:
                val = np.asarray(val, dtype=float)
                if val.shape != (n,):
                    raise PyFVCOM2ValueError(
                        f"Sponge {name} array length {len(val)} does not match "
                        f"the number of boundary nodes ({n})."
                    )
            if name == 'radius':
                self._sponge_radius = val
            else:
                self._sponge_coefficient = val


class Grid:
    """A class to represent a triangular mesh.

    Attributes:
        n_nodes (int): Number of nodes in the mesh.
        n_elements (int): Number of elements in the mesh.
        n_sigma_levels (int): Number of sigma levels in the vertical grid.
        n_sigma_layers (int): Number of sigma layers in the vertical grid.
        n_open_boundaries (int): Number of open boundaries.
        epsg_code (str): EPSG code used for coordinate transformations.
        lon_nodes (np.ndarray): Longitude of nodes.
        lat_nodes (np.ndarray): Latitude of nodes.
        lon_elements (np.ndarray): Longitude of element centroids.
        lat_elements (np.ndarray): Latitude of element centroids.
        x_nodes (np.ndarray): Cartesian x-coordinates of nodes.
        y_nodes (np.ndarray): Cartesian y-coordinates of nodes.
        x_elements (np.ndarray): Cartesian x-coordinates of element centroids.
        y_elements (np.ndarray): Cartesian y-coordinates of element centroids.
        bathy_nodes (np.ndarray): Bathymetry at nodes.
        bathy_elements (np.ndarray): Bathymetry at element centroids.
        triangles (np.ndarray): Triangle connectivity array of shape (n_elements, 3), 0-indexed.
        types_bdy (np.ndarray): Boundary node type codes.
        nodes_bdy (np.ndarray): Boundary node indices.
        sigma_config (SigmaConfig): Sigma coordinate configuration.
        sigma_levels (np.ndarray): Sigma level coordinates at nodes, shape (n_nodes, n_levels).
        sigma_layers (np.ndarray): Sigma layer coordinates at nodes, shape (n_nodes, n_layers).
        sigmac_levels (np.ndarray): Sigma level coordinates at element centroids, shape (n_elements, n_levels).
        sigmac_layers (np.ndarray): Sigma layer coordinates at element centroids, shape (n_elements, n_layers).
        sigma_levels_nodes (np.ndarray): Sigma level coordinates at nodes, shape (n_levels, n_nodes).
        sigma_layers_nodes (np.ndarray): Sigma layer coordinates at nodes, shape (n_layers, n_nodes).
        sigma_levels_elements (np.ndarray): Sigma level coordinates at element centroids, shape (n_levels, n_elements).
        sigma_layers_elements (np.ndarray): Sigma layer coordinates at element centroids, shape (n_layers, n_elements).
        z_layers_static (np.ndarray): Static z-coordinates at layer centres for nodes, shape (n_nodes, n_layers).
        zc_layers_static (np.ndarray): Static z-coordinates at layer centres for element centroids, shape (n_elements, n_layers).
        z_levels_static (np.ndarray): Static z-coordinates at layer interfaces for nodes, shape (n_nodes, n_levels).
        zc_levels_static (np.ndarray): Static z-coordinates at layer interfaces for element centroids, shape (n_elements, n_levels).
        open_boundaries (list[OpenBoundary]): List of open boundary objects.
    """

    def __init__(
        self,
        mesh_data: MeshData,
        sigma_data: SigmaData,
        coordinate_system: str,
        epsg_code: Optional[str] = None,
    ):
        self._triangles = mesh_data.triangle
        self._types_bdy = mesh_data.types_bdy
        self._nodes_bdy = mesh_data.nodes_bdy

        self._n_nodes = mesh_data.nodes.shape[0]
        self._n_elements = mesh_data.triangle.shape[0]

        if coordinate_system == "cartesian":
            self._x = mesh_data.x1
            self._y = mesh_data.x2
            self.epsg_code = epsg_code
            self._lon, self._lat = lonlat_from_utm(self._x, self._y, epsg_code)
        elif coordinate_system == "geographic":
            self._lon = mesh_data.x1
            self._lat = mesh_data.x2
            self._x, self._y, self.epsg_code = utm_from_lonlat(self._lon, self._lat, epsg_code)
        else:
            raise PyFVCOM2ValueError(
                "coordinate_system must be either 'cartesian' or 'geographic'. "\
                f"Received '{coordinate_system}'."
            )

        # Element centre coordinates
        self._xc = nodes2elems(self._x, self._triangles)
        self._yc = nodes2elems(self._y, self._triangles)
        self._lonc, self._latc = lonlat_from_utm(self._xc, self._yc, self.epsg_code)

        # Bathymetry at nodes and elements
        self._h = mesh_data.x3
        self._hc = nodes2elems(self._h, self._triangles)

        # Vertical grid
        # -------------
        self._add_sigma_coordinates(sigma_data)

        # Bed roughness
        # -------------
        self._z0b: Optional[np.ndarray] = None
        self._cbcmin: Optional[np.ndarray] = None

        # Open boundaries
        # ---------------
        self.open_boundaries = []
        if mesh_data.nodes_bdy is not None:
            for bdy_id, bdy_nodes in enumerate(mesh_data.nodes_bdy):
                # Convert to numpy array if it isn't already
                bdy_node_indices = np.asarray(bdy_nodes)
                
                # Extract sigma levels and layers for boundary nodes
                bdy_sigma_levels = self._sigma_levels[bdy_node_indices, :]
                bdy_sigma_layers = self._sigma_layers[bdy_node_indices, :]
 
                # Create OpenBoundary object
                open_boundary = OpenBoundary(
                    bdy_id=bdy_id,
                    node_indices=bdy_node_indices,
                    sigma_levels=bdy_sigma_levels,
                    sigma_layers=bdy_sigma_layers
                )
                self.open_boundaries.append(open_boundary)

    # Add property decorators for retrieving class attributes
    @property
    def n_nodes(self):
        """Get the number of nodes in the mesh."""
        return self._n_nodes

    @property
    def n_elements(self):
        """Get the number of elements in the mesh."""
        return self._n_elements

    @property
    def n_sigma_levels(self):
        """Get the number of sigma levels in the vertical grid."""
        return self._n_sigma_levels

    @property
    def n_sigma_layers(self):
        """Get the number of sigma layers in the vertical grid."""
        return self._n_sigma_layers

    @property
    def lon_nodes(self):
        """Get the longitude values at nodes."""
        return self._lon

    @property
    def lat_nodes(self):
        """Get the latitude values at nodes."""
        return self._lat

    @property
    def lon_elements(self):
        """Get the longitude values of element centroids."""
        return self._lonc

    @property
    def lat_elements(self):
        """Get the latitude values of element centroids."""
        return self._latc

    @property
    def x_nodes(self):
        """Get the Cartesian x-coordinates at nodes."""
        return self._x

    @property
    def y_nodes(self):
        """Get the Cartesian y-coordinates at nodes."""
        return self._y

    @property
    def x_elements(self):
        """Get the Cartesian x-coordinates of element centroids."""
        return self._xc

    @property
    def y_elements(self):
        """Get the Cartesian y-coordinates of element centroids."""
        return self._yc

    @property
    def sigma_layers_nodes(self):
        """Get the sigma layer values at nodes (n_layers, n_nodes)."""
        return self._sigma_layers.T

    @property
    def sigma_layers_elements(self):
        """Get the sigma layer values at element centroids (n_layers, n_elements)."""
        return self._sigmac_layers.T

    @property
    def sigma_levels_nodes(self):
        """Get the sigma level values at nodes (n_levels, n_nodes)."""
        return self._sigma_levels.T

    @property
    def sigma_levels_elements(self):
        """Get the sigma level values at element centroids (n_levels, n_elements)."""
        return self._sigmac_levels.T

    @property
    def bathy_nodes(self):
        """Get the bathymetry values at nodes."""
        return self._h

    @property
    def bathy_elements(self):
        """Get the bathymetry values at element centroids."""
        return self._hc


    @property
    def triangles(self):
        """Get the triangle connectivity array (n_elements, 3), 0-indexed."""
        return self._triangles

    @property
    def types_bdy(self):
        """Get the boundary node types array."""
        return self._types_bdy

    @property
    def nodes_bdy(self):
        """Get the boundary node indices."""
        return self._nodes_bdy

    @property
    def sigma_config(self):
        """Get the sigma configuration."""
        return self._sigma_config

    @property
    def sigma_levels(self):
        """Get sigma level coordinates at nodes (n_nodes, n_levels)."""
        return self._sigma_levels

    @property
    def sigma_layers(self):
        """Get sigma layer coordinates at nodes (n_nodes, n_layers)."""
        return self._sigma_layers

    @property
    def sigmac_levels(self):
        """Get sigma level coordinates at element centroids (n_elements, n_levels)."""
        return self._sigmac_levels

    @property
    def sigmac_layers(self):
        """Get sigma layer coordinates at element centroids (n_elements, n_layers)."""
        return self._sigmac_layers

    @property
    def z_layers_static(self):
        """Get static z-coordinates at layer centres for nodes (n_nodes, n_layers)."""
        return self._z_layers_static

    @property
    def zc_layers_static(self):
        """Get static z-coordinates at layer centres for element centroids (n_elements, n_layers)."""
        return self._zc_layers_static

    @property
    def z_levels_static(self):
        """Get static z-coordinates at layer interfaces for nodes (n_nodes, n_levels)."""
        return self._z_levels_static

    @property
    def zc_levels_static(self):
        """Get static z-coordinates at layer interfaces for element centroids (n_elements, n_levels)."""
        return self._zc_levels_static

    @property
    def n_open_boundaries(self):
        """Get the number of open boundaries."""
        return len(self.open_boundaries)

    def _add_sigma_coordinates(self, sigma_data: SigmaData):
        """ Add sigma coordinates from a sigma configuration.
        
        Args:
            sigma_data: Sigma data.
        """
        self._sigma_config = sigma_data.sigma_config
        self._sigma_levels = sigma_data.sigma_levels

        # Create a sigma layer variable (i.e. midpoint in the sigma levels).
        self._sigma_layers = self._sigma_levels[:, 0:-1] + (
            np.diff(self._sigma_levels, axis=1) / 2
        )

        self._n_sigma_levels = self._sigma_levels.shape[1]
        self._n_sigma_layers = self._sigma_layers.shape[1]

        # Create a sigma layer variable (i.e. midpoint in the sigma levels).
        self._sigmac_levels = nodes2elems(self._sigma_levels.T, self._triangles).T
        self._sigmac_layers = nodes2elems(self._sigma_layers.T, self._triangles).T

        # Compute static z coordinates for sigma levels and layers, which can be used
        # when interpolating data onto FVCOM's grid. We assume a value of zero for zeta
        # and generate a set of "corrected" values for the bathymetry by replacing all
        # negative h values (indicating a position above the zero geoid, which is most-likely
        # intertidal) with a value of 1 m. When interpolating from coarse cmems data, for example,
        # this ensures we can fill data arrays with sensible values.
        bathy_nodes_corrected = np.where(self._h < 0, 1.0, self._h)
        bathy_elements_corrected = np.where(self._hc < 0, 1.0, self._hc)
        self._z_layers_static = bathy_nodes_corrected[:, np.newaxis] * self._sigma_layers
        self._zc_layers_static = bathy_elements_corrected[:, np.newaxis] * self._sigmac_layers
        self._z_levels_static = bathy_nodes_corrected[:, np.newaxis] * self._sigma_levels
        self._zc_levels_static = bathy_elements_corrected[:, np.newaxis] * self._sigmac_levels

    def get_interpolation_coordinates(self, horizontal_position: str, vertical_position: str,
                                      horizontal_coordinate_system: str = "geographic",
                                      vertical_coordinate_system: str = "z",
                                      dates: Optional[np.ndarray] = None) -> InterpolationCoordinates:
        """Get interpolation coordinates for a specific grid position.

        Args:
            horizontal_position: Whether coordinates are at mesh nodes or element centres ('node' or 'element').
            vertical_position: Whether depth coordinates are at layer centres or layer interfaces
                ('layer_centre' or 'layer_interface').
            horizontal_coordinate_system: The coordinate system ("geographic" or "cartesian") for the interpolation coordinates.
            vertical_coordinate_system: The vertical coordinate system ("z" or "sigma") for the interpolation coordinates.
            dates: Array of datetime objects for temporal interpolation. If None, returns empty array.

        Returns:
            InterpolationCoordinates: The interpolation coordinates for the specified grid position.
        """
        if horizontal_position not in ['node', 'element']:
            raise PyFVCOM2ValueError("horizontal_position must be either 'node' or 'element'")

        if vertical_position not in ['layer_centre', 'layer_interface']:
            raise PyFVCOM2ValueError("vertical_position must be either 'layer_centre' or 'layer_interface'")

        if horizontal_coordinate_system not in ['geographic', 'cartesian']:
            raise PyFVCOM2ValueError("coordinate_system must be either 'geographic' or 'cartesian'")

        if vertical_coordinate_system not in ['z', 'sigma']:
            raise PyFVCOM2ValueError("vertical_coordinate_system must be either 'z' or 'sigma'")

        if horizontal_position == 'node':
            if horizontal_coordinate_system == 'geographic':
                x1 = self.lon_nodes
                x2 = self.lat_nodes
            else:  # cartesian
                x1 = self._x
                x2 = self._y
            if vertical_position == 'layer_interface':
                if vertical_coordinate_system == 'z':
                    x3 = self._z_levels_static.T
                else:  # 'sigma'
                    x3 = self._sigma_levels.T
            else:  # 'layer_centre'
                if vertical_coordinate_system == 'z':
                    x3 = self._z_layers_static.T
                else:  # 'sigma'
                    x3 = self._sigma_layers.T
        else:  # horizontal_position == 'element'
            if horizontal_coordinate_system == 'geographic':
                x1 = self.lon_elements
                x2 = self.lat_elements
            else:  # cartesian
                x1 = self._xc
                x2 = self._yc
            if vertical_position == 'layer_interface':
                if vertical_coordinate_system == 'z':
                    x3 = self._zc_levels_static.T
                else:  # 'sigma'
                    x3 = self._sigmac_levels.T
            else:  # 'layer_centre'
                if vertical_coordinate_system == 'z':
                    x3 = self._zc_layers_static.T
                else:  # 'sigma'
                    x3 = self._sigmac_layers.T

        # Use provided dates or empty array
        if dates is None:
            dates = np.array([])

        return InterpolationCoordinates(dates, x3, x2, x1, horizontal_coordinate_system=horizontal_coordinate_system, vertical_coordinate_system=vertical_coordinate_system)

    def write_grid(self, grid_file: str, coordinate_system: str, depth_file: Optional[str] = None) -> None:
        """Write the unstructured grid to an FVCOM-formatted ASCII file.

        Args:
            grid_file: Path to the output grid file.
            coordinate_system: Which coordinates to write, either 'geographic' (lon/lat) or 'cartesian' (x/y).
            depth_file: If given, also write a separate FVCOM depth file.
        """
        import warnings
        if coordinate_system == 'geographic':
            x, y = self._lon, self._lat
        elif coordinate_system == 'cartesian':
            x, y = self._x, self._y
        else:
            raise PyFVCOM2ValueError(
                "coordinate_system must be either 'geographic' or 'cartesian'. "
                f"Received '{coordinate_system}'."
            )

        depth = self._h.copy()
        if np.sum(depth < 0) > np.sum(depth > 0):
            depth = -depth
            warnings.warn('Flipping depths to be positive down since mostly negative depths were supplied.')

        nodes = np.arange(self.n_nodes) + 1
        with open(grid_file, 'w') as f:
            f.write('Node Number = {:d}\n'.format(self.n_nodes))
            f.write('Cell Number = {:d}\n'.format(self.n_elements))
            for i, triangle in enumerate(self._triangles, 1):
                f.write('{node:d} {:d} {:d} {:d} {node:d}\n'.format(node=i, *triangle + 1))
            for node in zip(nodes, x, y, depth):
                f.write('{:d} {:.6f} {:.6f} {:.6f}\n'.format(*node))

        if depth_file is not None:
            with open(depth_file, 'w') as f:
                f.write('Node Number = {:d}\n'.format(self.n_nodes))
                for row in zip(x, y, depth):
                    f.write('{:.6f} {:.6f} {:.6f}\n'.format(*row))

    def write_coriolis(self, coriolis_file: str, coordinate_system: str) -> None:
        """Write an FVCOM-formatted Coriolis file.

        Args:
            coriolis_file: Path to the output Coriolis file.
            coordinate_system: Which horizontal coordinates to write alongside latitude,
                either 'geographic' (lon/lat) or 'cartesian' (x/y).
        """
        if coordinate_system == 'geographic':
            x, y = self._lon, self._lat
        elif coordinate_system == 'cartesian':
            x, y = self._x, self._y
        else:
            raise PyFVCOM2ValueError(
                "coordinate_system must be either 'geographic' or 'cartesian'. "
                f"Received '{coordinate_system}'."
            )

        with open(coriolis_file, 'w') as f:
            f.write('Node Number = {:d}\n'.format(self.n_nodes))
            for row in zip(x, y, self._lat):
                f.write('{:.6f} {:.6f} {:.6f}\n'.format(*row))

    def write_obc(self, obc_file: str) -> None:
        """Write open boundary node IDs and types to an FVCOM-formatted ASCII file.

        Args:
            obc_file: Path to the output open boundary file.
        """
        ids = []
        types = []
        for boundary in self.open_boundaries:
            ids.extend(boundary.node_indices.tolist())
            types.extend([boundary.bdy_id] * boundary.nnodes)

        with open(obc_file, 'w') as f:
            f.write('OBC Node Number = {:d}\n'.format(len(ids)))
            for count, node, obc_type in zip(np.arange(len(ids)) + 1, ids, types):
                f.write('{} {:d} {:d}\n'.format(count, int(node) + 1, int(obc_type)))

    def write_sponge(self, sponge_file: str) -> None:
        """Write sponge layer parameters to an FVCOM-formatted ASCII file.

        Collects sponge radius and coefficient values from every open boundary
        and writes the ``Sponge Node Number`` ASCII file consumed by FVCOM's
        ``SPONGE_FILE`` namelist entry.

        Args:
            sponge_file: Path to the output sponge file.

        Raises:
            PyFVCOM2ValueError: If any open boundary has not had
                :meth:`OpenBoundary.set_sponge` called.
        """
        nodes, radii, coefficients = [], [], []
        for boundary in self.open_boundaries:
            if boundary.sponge_radius is None or boundary.sponge_coefficient is None:
                raise PyFVCOM2ValueError(
                    f"Open boundary {boundary.bdy_id} has no sponge layer set. "
                    "Call boundary.set_sponge(radius, coefficient) first."
                )
            nodes.extend(boundary.node_indices.tolist())
            radii.extend(boundary.sponge_radius.tolist())
            coefficients.extend(boundary.sponge_coefficient.tolist())

        with open(sponge_file, 'w') as f:
            f.write('Sponge Node Number = {:d}\n'.format(len(nodes)))
            for node, r, c in zip(nodes, radii, coefficients):
                f.write('{:d} {:.6f} {:.6f}\n'.format(int(node) + 1, r, c))

    def set_bed_roughness(self, z0b: float | np.ndarray, cbcmin: Optional[float | np.ndarray] = None) -> None:
        """Set the bed roughness parameters for the grid.

        Args:
            z0b: Bottom roughness in metres. A scalar is broadcast to all
                elements; an array must have length ``n_elements``.
            cbcmin: Minimum bottom drag coefficient (dimensionless), optional.
                Same broadcast/validation rules as ``z0b``. If not supplied,
                ``cbcmin`` is not written by :meth:`write_bed_roughness`.

        Raises:
            PyFVCOM2ValueError: If an array argument has the wrong length.
        """
        def _to_element_array(value, name):
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 0:
                return np.full(self._n_elements, arr.item())
            if arr.shape != (self._n_elements,):
                raise PyFVCOM2ValueError(
                    f"'{name}' must be a scalar or an array of length n_elements "
                    f"({self._n_elements}); got shape {arr.shape}."
                )
            return arr

        self._z0b = _to_element_array(z0b, 'z0b')
        self._cbcmin = _to_element_array(cbcmin, 'cbcmin') if cbcmin is not None else None

    def write_bed_roughness(self, roughness_file: str) -> None:
        """Write the bed roughness to a NetCDF4 file.

        Writes the ``z0b`` variable (bottom roughness in metres, dimensioned
        ``nele``) and, if :meth:`set_bed_roughness` was called with a
        ``cbcmin`` value, a ``cbcmin`` variable (minimum drag coefficient,
        dimensionless).

        The output file is consumed by FVCOM via the ``BEDFRICFILE`` namelist
        entry.

        Args:
            roughness_file: Path to the output NetCDF file.

        Raises:
            PyFVCOM2ValueError: If :meth:`set_bed_roughness` has not been
                called.
        """
        if self._z0b is None:
            raise PyFVCOM2ValueError(
                "No bed roughness set. Call set_bed_roughness(z0b) first."
            )

        dims = {'nele': self._n_elements}
        global_attrs = {'title': 'bottom roughness'}
        ncopts = {'zlib': True, 'complevel': 7}

        with FVCOMWriter(roughness_file, dims, global_attributes=global_attrs,
                         format='NETCDF4') as writer:
            writer.add_variable(
                'z0b', self._z0b, ['nele'],
                attributes={'long_name': 'bottom roughness', 'units': 'm'},
                ncopts=ncopts,
            )
            if self._cbcmin is not None:
                writer.add_variable(
                    'cbcmin', self._cbcmin, ['nele'],
                    attributes={'long_name': 'bottom roughness minimum', 'units': '1'},
                    ncopts=ncopts,
                )

    def write_sigma(self, sigma_file: str) -> None:
        """Write the sigma coordinate configuration to an FVCOM-formatted ASCII file.

        Args:
            sigma_file: Path to the output sigma file.
        """
        write_sigma_file(self._sigma_config, sigma_file)


def create_grid(grid_file: str, mesh_type: str, sigma_file: str, coordinate_system: str,
                epsg_code: Optional[str] = None, **kwargs) -> Grid:
    """Create a Grid object from mesh and sigma files.

    Args:
        grid_file: Path to the mesh file.
        mesh_type: Type of the mesh file (e.g., 'fvcom', 'gmsh').
        sigma_file: Path to the sigma file.
        coordinate_system: Coordinate system of the mesh ('cartesian' or 'geographic').
        epsg_code: EPSG code for coordinate transformations (optional).
        **kwargs: Additional keyword arguments for mesh reading functions.
    Returns:
        Grid: The created Grid object.
    """
    # Read mesh data
    mesh_data = read_mesh_file(grid_file, mesh_type=mesh_type, **kwargs)

    # Read sigma data
    sigma_config = read_sigma_file(sigma_file)
    sigma_data = process_sigma_config(sigma_config, mesh_data.x3)

    return Grid(mesh_data, sigma_data, coordinate_system, epsg_code)


def connectivity(p, t):
    """
    Assemble connectivity data for a triangular mesh.

    The edge based connectivity is built for a triangular mesh and the boundary
    nodes identified. This data should be useful when implementing FE/FV
    methods using triangular meshes.

    Args:
    p : np.ndarray
        Nx2 array of nodes coordinates, [[x1, y1], [x2, y2], etc.]
    t : np.ndarray
        Mx3 array of triangles as indices, [[n11, n12, n13], [n21, n22, n23],
        etc.]

    Returns:
    e : np.ndarray
        Kx2 array of unique mesh edges - [[n11, n12], [n21, n22], etc.]
    te : np.ndarray
        Mx3 array of triangles as indices into e, [[e11, e12, e13], [e21, e22,
        e23], etc.]
    e2t : np.ndarray
        Kx2 array of triangle neighbours for unique mesh edges - [[t11, t12],
        [t21, t22], etc]. Each row has two entries corresponding to the
        triangle numbers associated with each edge in e. Boundary edges have
        e2t[i, 1] = -1.
    bnd : np.ndarray, bool
        Nx1 logical array identifying boundary nodes. p[i, :] is a boundary
        node if bnd[i] = True.

    Notes:
    Python translation of the MATLAB MESH2D connectivity function by Darren
    Engwirda. See: https://github.com/dengwirda/MESH2D. Code translated by
    Pierre Cazenave, PML.

    References:
    .. [1] Darren Engwirda, Locally-optimal Delaunay-refinement and optimisation-based
    mesh generation, Ph.D. Thesis, School of Mathematics and Statistics,
    The University of Sydney, September 2014.

    """

    def _unique_rows(A, return_index=False, return_inverse=False):
        """
        Similar to MATLAB's unique(A, 'rows'), this returns B, I, J
        where B is the unique rows of A and I and J satisfy
        A = B[J, :] and B = A[I, :]

        Returns I if return_index is True
        Returns J if return_inverse is True

        Taken from https://github.com/numpy/numpy/issues/2871

        """
        A = np.require(A, requirements="C")
        assert A.ndim == 2, "array must be 2-dim'l"

        B = np.unique(
            A.view([("", A.dtype)] * A.shape[1]),
            return_index=return_index,
            return_inverse=return_inverse,
        )

        if return_index or return_inverse:
            return (B[0].view(A.dtype).reshape((-1, A.shape[1]), order="C"),) + B[1:]
        else:
            return B.view(A.dtype).reshape((-1, A.shape[1]), order="C")

    if p.shape[-1] != 2:
        raise Exception("p must be an Nx2 array")
    if t.shape[-1] != 3:
        raise Exception("t must be an Mx3 array")
    if np.any(t.ravel() < 0) or t.max() > p.shape[0] - 1:
        raise Exception("Invalid t")

    # Unique mesh edges as indices into p
    numt = t.shape[0]
    # Triangle indices
    vect = np.arange(numt)
    # Edges - not unique
    e = np.vstack(([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]))
    # Unique edges
    e, j = _unique_rows(np.sort(e, axis=1), return_inverse=True)
    # Unique edges in each triangle
    te = np.column_stack((j[vect], j[vect + numt], j[vect + (2 * numt)]))

    # Edge-to-triangle connectivity
    # Each row has two entries corresponding to the triangle numbers
    # associated with each edge. Boundary edges have e2t[i, 1] = -1.
    nume = e.shape[0]
    e2t = np.zeros((nume, 2)).astype(int) - 1
    for k in range(numt):
        for j in range(3):
            ce = te[k, j]
            if e2t[ce, 0] == -1:
                e2t[ce, 0] = k
            else:
                e2t[ce, 1] = k

    # Flag boundary nodes
    bnd = np.zeros((p.shape[0],)).astype(bool)
    # True for bnd nodes
    bnd[e[e2t[:, 1] == -1, :]] = True

    return e, te, e2t, bnd


def find_connected_nodes(n, triangles):
    """Return the IDs of the nodes surrounding node number `n'.

    Args:
    n : int
        Node ID around which to find the connected nodes.
    triangles : np.ndarray
        Triangulation matrix to find the connected nodes. Shape is [nele,
        3].

    Returns
    -------
    surroundingidx : np.ndarray
        Indices of the surrounding nodes.

    See Also
    --------
    PyFVCOM.grid.find_connected_elements().

    Notes
    -----

    Check it works with:
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from scipy.spatial import Delaunay
    >>> x, y = np.meshgrid(np.arange(25), np.arange(100, 125))
    >>> x = x.flatten() + np.random.randn(x.size) * 0.1
    >>> y = y.flatten() + np.random.randn(y.size) * 0.1
    >>> tri = Delaunay(np.array((x, y)).transpose())
    >>> for n in np.linspace(1, len(x) - 1, 5).astype(int):
    ...     aa = surrounders(n, tri.vertices)
    ...     plt.figure()
    ...     plt.triplot(x, y, tri.vertices, zorder=20, alpha=0.5)
    ...     plt.plot(x[n], y[n], 'ro', label='central node')
    ...     plt.plot(x[aa], y[aa], 'ko', label='connected nodes')
    ...     plt.xlim(x[aa].min() - 1, x[aa].max() + 1)
    ...     plt.ylim(y[aa].min() - 1, y[aa].max() + 1)
    ...     plt.legend(numpoints=1)

    """

    eidx = np.max((np.abs(triangles - n) == 0), axis=1)
    surroundingidx = np.unique(triangles[eidx][triangles[eidx] != n])

    return surroundingidx


def find_connected_elements(n, triangles):
    """
    Return the IDs of the elements connected to node number `n'.

    Parameters
    ----------
    n : int or iterable
        Node ID(s) around which to find the connected elements. If more than
        one node is given, the unique elements for all nodes are returned.
        Order of results is not maintained.
    triangles : np.ndarray
        Triangulation matrix to find the connected elements. Shape is [nele,
        3].

    Returns
    -------
    surroundingidx : np.ndarray
        Indices of the surrounding elements.

    See Also
    --------
    PyFVCOM.grid.find_connected_nodes().

    """

    try:
        surroundingidx = []
        for ni in n:
            idx = np.argwhere(triangles == ni)[:, 0]
            surroundingidx.append(idx)
        surroundingidx = np.asarray(
            [item for sublist in surroundingidx for item in sublist]
        )
        surroundingidx = np.unique(surroundingidx)
    except TypeError:
        surroundingidx = np.argwhere(triangles == n)[:, 0]

    return surroundingidx


def nodes2elems(nodes, tri):
    """
    Calculate an element-centre value based on the average value for the
    nodes from which it is formed. This involves an average, so the
    conversion from nodes to elements cannot be reversed without smoothing.

    Parameters
    ----------
    nodes : np.ndarray
        Array of unstructured grid node values to move to the element
        centres.
    tri : np.ndarray
        Array of shape (nelem, 3) comprising the list of connectivity
        for each element.

    Returns
    -------
    elems : np.ndarray
        Array of values at the grid nodes.

    """

    if np.ndim(nodes) == 1:
        elems = nodes[tri].mean(axis=-1)
    else:
        elems = nodes[..., tri].mean(axis=-1)

    return elems
