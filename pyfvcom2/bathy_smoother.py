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
]


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _edge_r_factors(
    h_nodes: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """Compute the Haney r-factor for every inter-element edge.

    The r-factor for an edge shared by two adjacent elements is:

    .. math::

        r = \\frac{|h_i - h_j|}{h_i + h_j}

    where :math:`h_i` and :math:`h_j` are element-mean depths.

    Edges where either adjacent element has a mean depth at or below the datum
    (i.e. intertidal or exposed elements) are **masked** and returned as
    :data:`numpy.nan`.  The Haney criterion is a measure of the sigma-layer
    pressure-gradient error: it is only meaningful for fully-submerged elements
    where the sigma layers span a positive water column.  FVCOM handles
    wetting/drying through a separate momentum cut-off, so including intertidal
    edges would produce physically meaningless and numerically singular r values.

    The returned array has the same length as the number of unique shared edges
    (i.e. one entry per pair of adjacent elements), so it can be indexed
    directly against any parallel per-edge array (e.g. edge midpoint
    coordinates) that is built with the same iteration order.

    Args:
        h_nodes: Node depths, shape ``(n_nodes,)``.  May contain negative
            values (intertidal / above-water nodes).
        triangles: Element connectivity (0-indexed), shape ``(n_elements, 3)``.

    Returns:
        r-factor for each unique inter-element edge, shape ``(n_edges,)``.
        Entries for edges adjacent to at least one intertidal element are
        :data:`numpy.nan`.
    """
    h_elems = h_nodes[triangles].mean(axis=1)
    wet = h_elems > 0  # element is fully submerged at mean depth

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
    ei0, ei1 = pairs_arr[:, 0], pairs_arr[:, 1]
    both_wet = wet[ei0] & wet[ei1]

    r = np.full(len(pairs_arr), np.nan)
    hi = h_elems[ei0[both_wet]]
    hj = h_elems[ei1[both_wet]]
    # Both hi and hj are > 0 by construction, so denominator is always positive.
    r[both_wet] = np.abs(hi - hj) / (hi + hj)
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

        Edges adjacent to at least one intertidal element (element-mean depth
        ≤ 0) are returned as :data:`numpy.nan` and excluded from the
        assessment.  Use :func:`numpy.nanmax` / :func:`numpy.nanmean` to
        compute statistics, and note that ``r > threshold`` comparisons
        naturally treat NaN as ``False``.

        Args:
            grid: :class:`~pyfvcom2.grid.Grid` instance to evaluate.

        Returns:
            Array of r-factor values, one per shared edge.  Values near zero
            indicate a smooth transition; values near 1 indicate a near-step
            change.  Intertidal edges are :data:`numpy.nan`.  A commonly used
            threshold is :math:`r < 0.2`.
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

