from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from .exceptions import PyFVCOM2ValueError

if TYPE_CHECKING:
    from .grid import Grid


__all__ = [
    "BathymetrySmoother",
    "GlobalBathymetrySmoother",
    "LocalBathymetrySmoother",
]


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _edge_r_factors(h_nodes: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Compute the Haney r-factor for every inter-element edge in the mesh.

    The r-factor for an edge shared by two adjacent elements is:

    .. math::

        r = \\frac{|h_i - h_j|}{h_i + h_j}

    where :math:`h_i` and :math:`h_j` are the mean node depths of the two
    elements sharing that edge.

    Args:
        h_nodes: Node depths, shape ``(n_nodes,)``.
        triangles: Element connectivity (0-indexed), shape ``(n_elements, 3)``.

    Returns:
        r-factor for each unique inter-element edge, shape ``(n_edges,)``.
    """
    # Element-mean depths
    h_elems = h_nodes[triangles].mean(axis=1)

    # Build shared-edge pairs: each triangle has 3 edges; collect adjacent
    # element pairs by finding triangles that share two nodes.
    # Represent each edge as a frozenset of two node indices → map to element.
    edge_to_elem: dict[frozenset, int] = {}
    pairs: list[tuple[int, int]] = []

    for ei, tri in enumerate(triangles):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = frozenset((a, b))
            if key in edge_to_elem:
                pairs.append((edge_to_elem[key], ei))
            else:
                edge_to_elem[key] = ei

    if not pairs:
        return np.empty(0)

    pairs_arr = np.array(pairs, dtype=np.intp)
    hi = h_elems[pairs_arr[:, 0]]
    hj = h_elems[pairs_arr[:, 1]]
    denom = hi + hj
    # Guard against zero-depth sums (should not occur in valid meshes)
    safe = denom > 0
    r = np.zeros(len(pairs_arr))
    r[safe] = np.abs(hi[safe] - hj[safe]) / denom[safe]
    return r


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BathymetrySmoother(ABC):
    """Abstract base class for FVCOM bathymetry smoothers.

    All concrete smoothers must implement :meth:`smooth`.  The shared
    :meth:`r_factor` method computes the Haney r-factor distribution for the
    current grid bathymetry.

    .. note::

        Only cold-start workflows are supported.  Smoothing the bathymetry
        after a hot-start restart file has been generated would produce an
        internally inconsistent restart because the stored 3-D fields are
        referenced to the original sigma-coordinate depths.  Apply smoothing
        before generating any forcing or restart files.
    """

    @abstractmethod
    def smooth(self, grid: Grid) -> None:
        """Smooth the bathymetry of *grid* in-place.

        Modifies ``grid._h`` (node depths) and ``grid._hc`` (element-centroid
        depths) directly.

        Args:
            grid: :class:`~pyfvcom2.grid.Grid` instance to modify.
        """

    def r_factor(self, grid: Grid) -> np.ndarray:
        """Return the Haney r-factor for every inter-element edge.

        Args:
            grid: :class:`~pyfvcom2.grid.Grid` instance to evaluate.

        Returns:
            Array of r-factor values, one per shared edge.  Values near zero
            indicate a smooth transition; values approaching 1 indicate a step
            change.  A commonly used threshold is :math:`r < 0.2`.
        """
        return _edge_r_factors(grid.bathy_nodes, grid.triangles)


# ---------------------------------------------------------------------------
# Global smoother
# ---------------------------------------------------------------------------

class GlobalBathymetrySmoother(BathymetrySmoother):
    """Global ping-pong bathymetry smoother.

    Applies one or more passes of a two-step linear interpolation:

    1. Interpolate node depths → element-centroid depths
       (``LinearNDInterpolator`` over the node positions).
    2. Interpolate the smoothed element depths → node depths
       (``LinearNDInterpolator`` over the centroid positions).

    Each pass is mathematically equivalent to one iteration of graph-Laplacian
    smoothing, reducing sharp depth gradients uniformly across the entire mesh.
    Boundary nodes that fall outside the convex hull of the centroids (step 2)
    retain their original depth.

    Args:
        passes: Number of complete node→centroid→node round-trips to apply.
            More passes produce stronger smoothing but progressively erode
            bathymetric features. Defaults to ``1``.
    """

    def __init__(self, passes: int = 1) -> None:
        if passes < 1:
            raise PyFVCOM2ValueError("'passes' must be >= 1.")
        self._passes = passes

    def smooth(self, grid: Grid) -> None:
        """Apply global ping-pong smoothing to *grid* in-place.

        Args:
            grid: :class:`~pyfvcom2.grid.Grid` instance to modify.
        """
        h = grid.bathy_nodes.copy()
        xn, yn = grid.x_nodes, grid.y_nodes
        xc, yc = grid.x_elements, grid.y_elements

        for _ in range(self._passes):
            # Step 1: nodes → centroids
            interp_to_centroids = LinearNDInterpolator((xn, yn), h)
            h_centroids = interp_to_centroids((xc, yc))

            # Fill any NaNs at centroid positions (shouldn't occur for interior
            # elements, but guard defensively)
            nan_mask = np.isnan(h_centroids)
            if nan_mask.any():
                h_centroids[nan_mask] = grid.bathy_elements[nan_mask]

            # Step 2: centroids → nodes
            interp_to_nodes = LinearNDInterpolator((xc, yc), h_centroids)
            h_new = interp_to_nodes((xn, yn))

            # Restore any boundary nodes outside the centroid convex hull
            nan_mask = np.isnan(h_new)
            h_new[nan_mask] = h[nan_mask]
            h = h_new

        grid._h = h
        grid._hc = h[grid.triangles].mean(axis=1)


# ---------------------------------------------------------------------------
# Local (targeted) smoother
# ---------------------------------------------------------------------------

class LocalBathymetrySmoother(BathymetrySmoother):
    """Targeted r-factor-driven bathymetry smoother.

    Rather than smoothing the entire mesh, this smoother identifies only the
    nodes that touch edges violating the Haney criterion and applies a
    weighted-average update to those nodes.  The process repeats until all
    edges satisfy :math:`r \\leq r_{max}` or ``max_iter`` iterations are
    reached.

    At each iteration:

    1. Compute the r-factor for every inter-element edge.
    2. Collect the set of nodes that belong to any non-compliant edge.
    3. For each such node, replace its depth with the mean depth of all
       nodes it is directly connected to (i.e. one graph-Laplacian step,
       applied only where needed).
    4. Recompute element-centroid depths from the updated node depths.

    This is the minimum smoothing necessary to satisfy the r-factor criterion
    and leaves unaffected regions of the mesh untouched.

    Args:
        r_max: Maximum permitted Haney r-factor. Edges exceeding this value
            are targeted for smoothing. Defaults to ``0.2``, the commonly
            used threshold for sigma-coordinate ocean models.
        max_iter: Maximum number of smoothing iterations before giving up.
            A :class:`~pyfvcom2.exceptions.PyFVCOM2ValueError` is raised if
            convergence is not achieved. Defaults to ``100``.
    """

    def __init__(self, r_max: float = 0.2, max_iter: int = 100) -> None:
        if r_max <= 0:
            raise PyFVCOM2ValueError("'r_max' must be positive.")
        if max_iter < 1:
            raise PyFVCOM2ValueError("'max_iter' must be >= 1.")
        self._r_max = r_max
        self._max_iter = max_iter

    def smooth(self, grid: Grid) -> None:
        """Apply targeted r-factor smoothing to *grid* in-place.

        Args:
            grid: :class:`~pyfvcom2.grid.Grid` instance to modify.

        Raises:
            PyFVCOM2ValueError: If the mesh does not converge to
                :math:`r \\leq r_{max}` within ``max_iter`` iterations.
        """
        triangles = grid.triangles
        h = grid.bathy_nodes.copy()

        # Pre-build node → neighbouring nodes lookup (shared-edge adjacency)
        node_neighbours: dict[int, set[int]] = {i: set() for i in range(grid.n_nodes)}
        for tri in triangles:
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                node_neighbours[int(a)].add(int(b))
                node_neighbours[int(b)].add(int(a))

        # Pre-build edge list with the two node indices for each shared edge
        # so we can identify which nodes to update.
        edge_to_elem: dict[frozenset, int] = {}
        edge_nodes: list[tuple[int, int]] = []
        edge_elem_pairs: list[tuple[int, int]] = []

        for ei, tri in enumerate(triangles):
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = frozenset((int(a), int(b)))
                if key in edge_to_elem:
                    edge_nodes.append((int(a), int(b)))
                    edge_elem_pairs.append((edge_to_elem[key], ei))
                else:
                    edge_to_elem[key] = ei

        edge_nodes_arr = np.array(edge_nodes, dtype=np.intp)
        edge_elem_pairs_arr = np.array(edge_elem_pairs, dtype=np.intp)

        for iteration in range(self._max_iter):
            h_elems = h[triangles].mean(axis=1)

            hi = h_elems[edge_elem_pairs_arr[:, 0]]
            hj = h_elems[edge_elem_pairs_arr[:, 1]]
            denom = hi + hj
            r = np.zeros(len(edge_elem_pairs_arr))
            safe = denom > 0
            r[safe] = np.abs(hi[safe] - hj[safe]) / denom[safe]

            bad_edges = np.where(r > self._r_max)[0]
            if len(bad_edges) == 0:
                break

            # Collect unique nodes on non-compliant edges
            bad_nodes = np.unique(edge_nodes_arr[bad_edges].ravel())

            # Laplacian update: replace each bad node's depth with the mean
            # of its direct neighbours' current depths.
            h_new = h.copy()
            for node in bad_nodes:
                neighbours = node_neighbours[node]
                if neighbours:
                    h_new[node] = np.mean(h[list(neighbours)])

            h = h_new
        else:
            max_r = r.max() if len(r) else 0.0
            raise PyFVCOM2ValueError(
                f"Bathymetry smoothing did not converge after {self._max_iter} "
                f"iterations. Maximum r-factor remaining: {max_r:.4f} "
                f"(target: {self._r_max:.4f}). "
                "Try increasing max_iter or r_max."
            )

        grid._h = h
        grid._hc = h[triangles].mean(axis=1)
