"""Plotting subpackage for pyfvcom2

This module exports the plotting classes and helper functions.
Users can import from `pyfvcom2.plotting` for a cleaner API.
"""

from .plot import PyFVCOM2Plotter, FVCOMPlotter, CMEMSPlotter, create_figure, create_cbar_ax, cm2inch, colourmap

__all__ = ["PyFVCOM2Plotter", "FVCOMPlotter", "CMEMSPlotter", "create_figure", "create_cbar_ax", "cm2inch", "colourmap"]
