"""Plotting subpackage for pyfvcom2

This module exports the plotting classes and helper functions.
Users can import from `pyfvcom2.plotting` for a cleaner API.
"""

from .plot import PyFVCOM2Plotter, FVCOMPlotter, create_figure

__all__ = ["PyFVCOM2Plotter", "FVCOMPlotter", "create_figure"]
