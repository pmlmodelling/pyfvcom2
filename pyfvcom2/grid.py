import numpy as np
from typing import Optional

from .mesh_reader import MeshData, read_mesh_file
from .sigma_reader import SigmaConfig, SigmaData, process_sigma_config, read_sigma_file
from .coordinates import lonlat_from_utm, utm_from_lonlat, sigma_to_z_coords
from .exceptions import PyFVCOM2ValueError
from .interpolation_coordinates import InterpolationCoordinates

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


class Grid:
    """A class to represent a triangular mesh.

    Attributes:
        nodes (np.ndarray): N array of node coordinates.
        triangles (np.ndarray): (n_elem, 3) array of triangle vertex indices.
        x (np.ndarray): x coordinates of nodes.
        y (np.ndarray): y coordinates of nodes.
        h (np.ndarray): bathymetry at nodes.
        xc (np.ndarray): x coordinates of triangle centroids.
        yc (np.ndarray): y coordinates of triangle centroids.
        hc (np.ndarray): bathymetry at triangle centroids.
        lon (np.ndarray): Longitude of nodes.
        lat (np.ndarray): Latitude of nodes.
        lonc (np.ndarray): Longitude of triangle centroids.
        latc (np.ndarray): Latitude of triangle centroids.
        open_boundaries[OpenBoundary]): List of open boundary objects.
    """

    def __init__(
        self,
        mesh_data: MeshData,
        sigma_data: SigmaData,
        coordinate_system: str,
        epsg_code: Optional[str] = None,
    ):
        self.triangles = mesh_data.triangle
        self.types_bdy = mesh_data.types_bdy
        self.nodes_bdy = mesh_data.nodes_bdy

        self._n_nodes = mesh_data.nodes.shape[0]
        self._n_elements = mesh_data.triangle.shape[0]

        if coordinate_system == "cartesian":
            self.x = mesh_data.x1
            self.y = mesh_data.x2
            self.epsg_code = epsg_code
            self.lon, self.lat = lonlat_from_utm(self.x, self.y, epsg_code)
        elif coordinate_system == "geographic":
            self.lon = mesh_data.x1
            self.lat = mesh_data.x2
            self.x, self.y, self.epsg_code = utm_from_lonlat(self.lon, self.lat, epsg_code)
        else:
            raise PyFVCOM2ValueError(
                "coordinate_system must be either 'cartesian' or 'geographic'"
            )

        # Element centre coordinates
        self.xc = nodes2elems(self.x, self.triangles)
        self.yc = nodes2elems(self.y, self.triangles)
        self.lonc, self.latc = lonlat_from_utm(self.xc, self.yc, self.epsg_code)

        # Bathymetry at nodes and elements
        self.h = mesh_data.x3
        self.hc = nodes2elems(self.h, self.triangles)

        # Vertical grid
        # -------------
        self._add_sigma_coordinates(sigma_data)

        # Open boundaries
        # ---------------
        self.open_boundaries = []
        if mesh_data.nodes_bdy is not None:
            for bdy_id, bdy_nodes in enumerate(mesh_data.nodes_bdy):
                # Convert to numpy array if it isn't already
                bdy_node_indices = np.asarray(bdy_nodes)
                
                # Extract sigma levels and layers for boundary nodes
                bdy_sigma_levels = self.sigma_levels[bdy_node_indices, :]
                bdy_sigma_layers = self.sigma_layers[bdy_node_indices, :]
 
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
        return self.lon

    @property
    def lat_nodes(self):
        """Get the latitude values at nodes."""
        return self.lat

    @property
    def lon_elements(self):
        """Get the longitude values of element centroids."""
        return self.lonc

    @property
    def lat_elements(self):
        """Get the latitude values of element centroids."""
        return self.latc

    @property
    def sigma_layers_nodes(self):
        """Get the sigma layer values at nodes."""
        return self.sigma_layers

    @property
    def sigma_layers_elements(self):
        """Get the sigma layer values at element centroids."""
        return self.sigmac_layers

    @property
    def bathy_nodes(self):
        """Get the bathymetry values at nodes."""
        return self.h

    @property
    def bathy_elements(self):
        """Get the bathymetry values at element centroids."""
        return self.hc

    @property
    def n_open_boundaries(self):
        """Get the number of open boundaries."""
        return len(self.open_boundaries)

    def _add_sigma_coordinates(self, sigma_data: SigmaData):
        """ Add sigma coordinates from a sigma configuration.
        
        Args:
            sigma_data: Sigma data.
        """
        self.sigma_config = sigma_data.sigma_config
        self.sigma_levels = sigma_data.sigma_levels

        # Create a sigma layer variable (i.e. midpoint in the sigma levels).
        self.sigma_layers = self.sigma_levels[:, 0:-1] + (
            np.diff(self.sigma_levels, axis=1) / 2
        )

        self._n_sigma_levels = self.sigma_levels.shape[1]
        self._n_sigma_layers = self.sigma_layers.shape[1]

        # Create a sigma layer variable (i.e. midpoint in the sigma levels).
        self.sigmac_levels = nodes2elems(self.sigma_levels.T, self.triangles).T
        self.sigmac_layers = nodes2elems(self.sigma_layers.T, self.triangles).T

        # Depth levels in z coordinates
        self.sigma_layers_z = self.h[:, np.newaxis] * self.sigma_layers
        self.sigmac_layers_z = self.hc[:, np.newaxis] * self.sigmac_layers
        self.sigma_levels_z = self.h[:, np.newaxis] * self.sigma_levels
        self.sigmac_levels_z = self.hc[:, np.newaxis] * self.sigmac_levels

    def get_interpolation_coordinates(self, grid_position: str, dates: Optional[np.ndarray] = None) -> InterpolationCoordinates:
        """Get interpolation coordinates for a specific grid position.

        Args:
            grid_position: The grid position ('node' or 'element') for which to retrieve
                interpolation coordinates.
            dates: Array of datetime objects for temporal interpolation. If None, returns empty array.

        Returns:
            InterpolationCoordinates: The interpolation coordinates for the specified grid position.
        """
        if grid_position not in ['node', 'element']:
            raise ValueError("grid_position must be either 'node' or 'element'")

        if grid_position == 'node':
            lons = self.lon
            lats = self.lat
            sigma_layers = self.sigma_layers.T
            bathy = self.h
        else:  # grid_position == 'element'
            lons = self.lonc
            lats = self.latc
            sigma_layers = self.sigmac_layers.T
            bathy = self.hc

        # Set zeta to zero for depth calculation (no free surface displacement)
        zeta = np.zeros_like(bathy)

        # Compute depths from sigma coordinates
        depths = sigma_to_z_coords(sigma_layers, zeta, bathy)

        # Use provided dates or empty array
        if dates is None:
            dates = np.array([])

        return InterpolationCoordinates(dates, depths, lats, lons)


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
