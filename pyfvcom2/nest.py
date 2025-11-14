import numpy as np

from .weights_calculator import get_weights_calculator
from .grid import Grid, OpenBoundary
from .grid import find_connected_elements


class GridBand:
    """Grid band with adjoining nodes and elements and vertical sigma structure.

    To be used when constructing an FVCOM nest.
    """
    
    def __init__(self, nodes: np.ndarray, elements: np.ndarray, 
                 element_weights: np.ndarray = None, node_weights: np.ndarray = None):
        """Initialize 3D GridBand.
        
        Args:
            nodes: Array of node coordinates.
            elements: Array of element connectivity.
            element_weights: Array of element weights.
            node_weights: Array of node weights.
        """
        self._nodes = nodes
        self._elements = elements
        self._element_weights = element_weights
        self._node_weights = node_weights

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

    Notes:
    - Using grid and nests, we can calculate all coordinate variables and weights.
    - This leaves just the forcing data and time data. Add variables for these here?
    - I think so, although we may want all the calculations to happen in a separate calculator
    (e.g. the interpolation). Or could it happen here, given an argument which permits all forcing data
    to be read and interpolated?
    - Need to add on tide data if needed, and may need a tide manager class.
    - Will then just need to implement a method which writes all the nest forcing data to file.
    """
    def __init__(self, grid: Grid, weights_calculation_method: str = 'linear'):
        self._grid_ref = grid # Reference to the full grid
        self.nests = []
        self.forcing_data = {}
        self.weights_calculator = get_weights_calculator(weights_calculation_method)

    def clear_nests(self) -> None:
        """ Clear all nests from the manager """
        self.nests = []
        self.foricing_data = {}

    def make_nests(self, grid: Grid, num_grid_bands: int) -> None:
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

        # Create new nest objects for each open boundary
        for ob in grid.open_boundaries:
            nest = Nest(open_boundary=ob)

            # Set of nodes that are used to locate elements in the next grid band. When beginning
            # to add grid bands, as were are here, we start with the open boundary nodes. As we successively
            # add grid bands, we update this to be the nodes in the most recently added grid band.
            reference_nodes = ob.node_indices
            
            for i in range(num_grid_bands):
                # First, find the elements that make up the grid band adjoining the current, inner
                # most set of nodes in the nest. If no grid bands have been added yet, this will
                # be the nodes that define the open boundary.
                elements = find_connected_elements(reference_nodes, grid.triangles) 

                # Only use unique elements that have not already been used in previous grid bands
                unique_elements = np.setdiff1d(elements, all_elements).tolist()

                # Get the nodes connected to the elements we've extracted.
                nodes = np.unique(grid.triangles[unique_elements, :])

                # Remove ones we already have in the nest.
                unique_nodes = np.setdiff1d(nodes, all_nodes).tolist()

                # Calculate weights for the nodes and elements in this grid band
                node_weights = self.weights_calculator.calculate_weights(
                    n=len(unique_nodes),
                    n_bands=num_grid_bands,
                    grid_band_index=i+1 # +1 as open boundary is band 0
                )
                element_weights = self.weights_calculator.calculate_weights(
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
                all_nodes.extend(band.nodes.tolist())

        # Return unique nodes as a numpy array
        return np.unique(np.array(all_nodes, dtype=np.int32))

    def get_all_nest_elements(self) -> np.ndarray:
        """Get all unique element indices used in all nests.

        Returns:
            Array of unique element indices.
        """
        all_elements = []
        for nest in self.nests:
            # Add elements from each grid band
            for band in nest.grid_bands:
                all_elements.extend(band.elements.tolist())

        # Return unique elements as a numpy array
        return np.unique(np.array(all_elements, dtype=np.int32))
    
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


class NestForcingCalculator:
    """ Class to calculate forcing data for nests """
    pass


class NestForcingWriter:
    """ Class to write nest forcing data to file """
    pass
