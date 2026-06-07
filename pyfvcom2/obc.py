from __future__ import annotations

import os
import numpy as np
from datetime import datetime
from typing import Optional

from .fvcom_writer import FVCOMWriter
from .version import full_version
from .grid import Grid
from .tide import TideManager
from .exceptions import PyFVCOM2ValueError


__all__ = ["OBCManager"]


class OBCManager:
    """Manager for FVCOM open boundary tidal elevation forcing.

    Predicts tidal elevation at open boundary nodes using a TideManager and
    writes an FVCOM-format OBC elevation forcing file suitable for driving a
    tide-only model run. The resulting file is typically used to generate a
    tidal harmonics file, which is then used alongside a nest forcing file
    in a full baroclinic run.

    Attributes:
        grid: Grid instance containing the open boundaries.
    """

    def __init__(self, grid: Grid) -> None:
        """Initialise the OBCManager.

        Args:
            grid: Grid instance whose open_boundaries define the OBC nodes.
        """
        self._grid_ref = grid
        self._dates = []
        self._zeta = None

    def set_dates(self, dates: list[datetime]) -> None:
        """Set the dates for which to generate forcing.

        Args:
            dates: List of datetime objects.
        """
        self._zeta = None
        self._dates = dates

    def add_tidal_data(self, tide_manager: TideManager) -> None:
        """Predict tidal elevation at all OBC nodes.

        Args:
            tide_manager: TideManager with an interpolator registered for
                'zeta'. Call tide_manager.add_interpolator('zeta', ...) first.
        """
        if not self._dates:
            raise PyFVCOM2ValueError(
                "Dates must be set before adding tidal data. Call set_dates first."
            )

        node_indices = self._get_obc_node_indices()
        lons = self._grid_ref.lon_nodes[node_indices]
        lats = self._grid_ref.lat_nodes[node_indices]

        datetimes = np.array(self._dates)
        self._zeta = tide_manager.predict('zeta', datetimes, lons, lats)

    def _get_obc_node_indices(self) -> np.ndarray:
        """Return all OBC node indices (0-based) across all open boundaries."""
        indices = []
        for boundary in self._grid_ref.open_boundaries:
            indices.extend(boundary.node_indices.tolist())
        return np.array(indices, dtype=np.int32)

    def create_forcing_file(self, output_path: str, format: str = 'NETCDF4',
                            **kwargs) -> None:
        """Write the OBC tidal elevation forcing file.

        Args:
            output_path: Path to the output NetCDF file.
            format: NetCDF format string. Defaults to 'NETCDF4'.
            **kwargs: Additional keyword arguments passed to FVCOMWriter.
                Pass ncopts (dict) to control compression etc.
        """
        if self._zeta is None:
            raise PyFVCOM2ValueError(
                "No tidal data available. Call add_tidal_data first."
            )

        node_indices = self._get_obc_node_indices()
        n_nodes = len(node_indices)
        n_times = len(self._dates)

        if 'ncopts' in kwargs:
            ncopts = kwargs.pop('ncopts')
        else:
            ncopts = {}

        global_attributes = {
            'type': 'FVCOM TIME SERIES ELEVATION FORCING FILE',
            'title': 'Tidal elevation open boundary forcing',
            'history': f'File created using PyFVCOM2 version {full_version}',
            'filename': os.path.basename(output_path),
            'Conventions': 'CF-1.0',
        }

        dims = {'nobc': n_nodes, 'time': 0, 'DateStrLen': 26}

        with FVCOMWriter(str(output_path), dims,
                         global_attributes=global_attributes,
                         clobber=True, format=format, **kwargs) as ncfile:

            atts = {'long_name': 'Open Boundary Node Number', 'grid': 'obc_grid'}
            ncfile.add_variable('obc_nodes', node_indices + 1, ['nobc'],
                                attributes=atts, ncopts=ncopts, format='i4')

            atts = {'long_name': 'internal mode iteration number'}
            ncfile.add_variable('iint', np.arange(n_times, dtype=np.int32),
                                ['time'], attributes=atts, ncopts=ncopts,
                                format='i4')

            ncfile.write_fvcom_time(self._dates, ncopts=ncopts)

            atts = {'long_name': 'Open Boundary Elevation', 'units': 'meters'}
            ncfile.add_variable('elevation', self._zeta, ['time', 'nobc'],
                                attributes=atts, ncopts=ncopts)

