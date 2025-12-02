"""FVCOM Plotting Functions"""

from typing import Optional, Union
import numpy as np
from pathlib import Path
from netCDF4 import Dataset

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm as mplcm
from matplotlib.tri import Triangulation
from matplotlib.collections import PolyCollection
from matplotlib import quiver as mpl_quiver
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cmocean import cm
from abc import abstractmethod

from ..grid import Grid

__all__ = [
    "PyFVCOM2Plotter",
    "FVCOMPlotter",
    "CMEMSPlotter",
    "create_figure",
    "create_cbar_ax",
    "cm2inch",
    "colourmap"
]


class PyFVCOM2Plotter:
    """Base class for PyFVCOM2 plotters

    Class to assist in the creation of plots and animations. The class can
    be used to create a set of basic plot objects. Plots that overlay
    particle trajectories on top of underlying field data should be created
    using the appropriate derived class.

    Args:
        geographic_coords (bool, optional): Boolean specifying whether or not to use cartopy to create a 2D map
            on top of which the data will be plotted. The default option is
            `True`. If `False`, a simple Cartesian grid is drawn instead.
        font_size (int, optional): Font size to use when rendering plot text
        line_width (float, optional): Default line width to use when plotting

    """

    def __init__(
        self,
        geographic_coords: Optional[bool] = True,
        font_size: Optional[int] = 10,
        line_width: Optional[float] = 0.2,
    ):

        self.geographic_coords = geographic_coords

        self.font_size = font_size

        self.line_width = line_width

    def _add_colour_bar(
        self,
        figure: matplotlib.figure.Figure,
        axes: matplotlib.axes.Axes,
        plot: PolyCollection,
        cb_label: Optional[str] = None,
    ):
        # Add colobar scaled to axis width
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
        cbar = figure.colorbar(plot, cax=cax)
        cbar.ax.tick_params(labelsize=self.font_size)
        if cb_label:
            cbar.set_label(cb_label, size=self.font_size)
        return

    @abstractmethod
    def plot_field(
        self,
        ax: matplotlib.axes.Axes,
        field: np.ndarray,
        **kwargs
    ) -> matplotlib.axes.Axes:
        """Map the supplied field

        Additional plotting options are passed to `matplotlib.pyplot.pcolormesh`. See the matplotlib documentation
        for a full list of supported options.

        Args:
            ax (matplotlib.axes.Axes): Axes object
            field (np.ndarray): The field to plot.
        Returns:
            matplotlib.axes.Axes: Axes object
        """
        pass

    def plot_lines(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        transform: Optional[ccrs.Projection] = None,
        **kwargs,
    ):
        """Plot path lines.

        In addition to the listed parameters, the function accepts all keyword arguments taken by the Matplotlib
        plot command.

        Args:
            ax (matplotlib.axes.Axes): Axes object
            x (np.ndarray): Array of x coordinates to plot.
            y (np.ndarray): Array of y coordinates to plot.
            transform (ccrs.Projection, optional): Transform for geographic projections
            **kwargs: Additional keyword arguments passed to matplotlib plot

        Returns:
            tuple: (axes, line_plots) - Axes object and list of line plot objects
        """
        transform = self._check_transform(transform)

        # Use some better default attributes if they have not been supplied
        alpha = kwargs.pop("alpha", 0.25)
        color = kwargs.pop("color", "r")
        linewidth = kwargs.pop("linewidth", 1.0)

        line_plots = ax.plot(
            x,
            y,
            alpha=alpha,
            color=color,
            linewidth=linewidth,
            transform=transform,
            **kwargs,
        )

        return ax, line_plots

    def _check_transform(self, transform: Optional[ccrs.Projection] = None):
        # If geographic coords, set the transform
        _transform = transform
        if self.geographic_coords and (transform is None):
            print(
                f"Plotting in geographic coordinates but no transform supplied. Using PlateCarree. "
                f"You can override this by supplying a transform argument."
            )
            _transform = ccrs.PlateCarree()

        return _transform



    def remove_line_plots(self, line_plots: list):
        """Remove line plots

        Useful when updating plots for animations.

        Args:
            line_plots (list): List of line plot objects created during call to plot_lines()
        """
        while line_plots:
            line_plots.pop(0).remove()

        return

    @abstractmethod
    def scatter(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        c: Optional[np.ndarray] = None,
        configure: Optional[bool] = False,
        extents: Optional[list] = None,
        transform: Optional[ccrs.Projection] = None,
        draw_coastlines: Optional[bool] = False,
        resolution: Optional[str] = "10m",
        tick_inc: Optional[bool] = False,
        **kwargs,
    ):
        """Create a scatter plot using the provided x and y values

        If geographic_coords is True, x and y should be geographic (lat, lon) coordinates. If not, x any y should
        be given as cartesian coordinates.

        See Matplotlib's scatter documentation for a list of additional key
        word arguments.

        Args:
            ax (matplotlib.axes.Axes): Axes object
            x (np.ndarray): Array of 'x' positions. If plotting in geographic coords, these should be longitudes.
            y (np.ndarray): Array of 'y' positions. If plotting in geographic coords, these should be latitudes.
            c (np.ndarray, optional): Array of colour values for each point. If provided, this will be used to
            colour the points in the scatter plot.
            configure (bool, optional): If true, configure the plot by setting plot extents, drawing coastlines etc. Default: False.
            extents (list, optional): Four element list giving lon/lat limits (e.g. [-4.56, -3.76, 55.12, 55.84])
            transform (ccrs.Projection, optional): The type of transform to perform if geographic_coords is True.
            draw_coastlines (bool, optional): Draw coastlines? Only used if geographic_coords is True.
            resolution (str, optional): Resolution to use when plotting the coastline. Only used when draw_coastline=True. Default: '10m'.
            tick_inc (bool, optional): Draw ticks? Only used if geographic_coords is True.
            **kwargs: Additional keyword arguments passed to matplotlib scatter

        Returns:
            tuple: (ax, scatter_plot) - Axes object and scatter plot collection
        """
        pass

    @abstractmethod
    def _get_default_extents(self):
        """Get the default plot extents
        
        Returns:
            np.ndarray: Array of [xmin, xmax, ymin, ymax]
        """
        pass

    def set_title(self, ax, title):
        """Set the title

        Args:
            ax (matplotlib.axes.Axes): Axes object
            title (str): Plot title
        """
        ax.set_title(title, fontsize=self.font_size)

    def _add_ticks(self, ax):
        gl = ax.gridlines(
            linewidth=self.line_width,
            draw_labels={"bottom": "x", "left": "y"},
            linestyle="--",
            color="k",
        )

        gl.xlabel_style = {"fontsize": self.font_size}
        gl.ylabel_style = {"fontsize": self.font_size}

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER


class FVCOMPlotter(PyFVCOM2Plotter):
    """Create FVCOM plot objects based on FVCOM model outputs or Grid objects

    Class to assist in the creation of plots and animations based on FVCOM
    data. Grid information can be provided either from a FVCOM NetCDF file
    or from a pre-existing Grid object.

    Args:
        fvcom_source (Union[str, Path, Grid]): Either a path to a FVCOM NetCDF file 
            (as string or Path object) or a Grid object containing the mesh information.
        geographic_coords (bool, optional): Whether to use geographic coordinates. 
            Default True. Ignored if fvcom_source is a Grid object.
        font_size (int, optional): Font size for plot text. Default 10.
        line_width (float, optional): Default line width for plotting. Default 0.2.
    """

    def __init__(
        self,
        fvcom_source: Union[str, Path, Grid],
        geographic_coords: Optional[bool] = True,
        font_size: Optional[int] = 10,
        line_width: Optional[float] = 0.2,
    ):
        # Initialise base class
        super().__init__(geographic_coords, font_size, line_width)

        # Check if input is a Grid object or file path
        if isinstance(fvcom_source, Grid):
            self._read_grid_from_object(fvcom_source)
        elif isinstance(fvcom_source, (str, Path)):
            # Open the NetCDF file for reading
            with Dataset(str(fvcom_source), "r") as ds:
                # Read grid information
                self._read_grid_information(ds)
        else:
            raise TypeError(
                "fvcom_source must be either a file path (str or Path) or a Grid object"
            )

    def _read_grid_from_object(self, grid: Grid):
        """Read grid information from a Grid object.
        
        Args:
            grid (Grid): Grid object containing mesh information.
        """
        # Read in the required grid variables from Grid object
        self.n_nodes = grid.n_nodes
        self.n_elems = grid.n_elements
        
        # Grid triangles need to be converted to the format expected by matplotlib
        # Grid.triangles is 0-indexed, which is what matplotlib expects
        self.nv = grid.triangles
        
        # Coordinates - Grid object always has both geographic and cartesian
        if self.geographic_coords:
            self.x = grid.lon
            self.y = grid.lat
            self.xc = grid.lonc
            self.yc = grid.latc
            self.transform = ccrs.PlateCarree()
        else:
            self.x = grid.x
            self.y = grid.y
            self.xc = grid.xc
            self.yc = grid.yc
            self.transform = None

        # Triangles for matplotlib
        self.triangles = self.nv

        # Store triangulation
        self.tri = Triangulation(self.x, self.y, self.triangles)

    def _read_grid_information(self, ds):
        # Read in the required grid variables
        self.n_nodes = ds.dimensions["node"].size
        self.n_elems = ds.dimensions["nele"].size
        self.nv = ds.variables["nv"][:] - 1  # Adjust for Fortran indexing

        if self.geographic_coords:
            self.x = ds.variables["lon"][:]
            self.y = ds.variables["lat"][:]
            self.xc = ds.variables["lonc"][:]
            self.yc = ds.variables["latc"][:]
            self.transform = ccrs.PlateCarree()
        else:
            self.x = ds.variables["x"][:]
            self.y = ds.variables["y"][:]
            self.xc = ds.variables["xc"][:]
            self.yc = ds.variables["yc"][:]

            self.transform = None

        # Triangles
        self.triangles = self.nv.transpose()

        # Store triangulation
        self.tri = Triangulation(self.x, self.y, self.triangles)

    def _get_default_extents(self):
        return np.array([self.x.min(), self.x.max(), self.y.min(), self.y.max()])

    def plot_field(
        self,
        ax: matplotlib.axes.Axes,
        field: np.ndarray,
        update: Optional[bool] = False,
        configure: Optional[bool] = True,
        add_colour_bar: Optional[bool] = True,
        cb_label: Optional[str] = None,
        tick_inc: Optional[bool] = True,
        extents: Optional[list] = None,
        draw_coastlines: Optional[bool] = False,
        resolution: Optional[str] = "10m",
        **kwargs
    ):
        """Map the supplied field

        The field must be defined on the same triangular mesh that was used to initialise the plotter.

        Additional plotting options are passed to `matplotlib.pyplot.pcolormesh`. See the matplotlib documentation
        for a full list of supported options.

        Args:
            ax (matplotlib.axes.Axes): Axes object
            field (np.ndarray): The field to plot.
            update (bool, optional): If true, update the existing plot. Specifically, the axes will be checked to see if it contains a
                PolyCollection object, as generated by tripcolor. If found, the associated data array will be
                updated with the supplied field data. This is faster than drawing a new map
            configure (bool, optional): If true, configure the plot by setting plot extents, drawing coastlines etc. This can be
                useful when overlaying plots, and you only want to incur the cost of configuring the plot
                once. The default is True, with the expectation that in most circumstances users will
                draw any underlying field data before overlaying particle tracks. Default: True.
            add_colour_bar (bool, optional): If true, draw a colour bar.
            cb_label (str, optional): The colour bar label.
            tick_inc (bool, optional): Add coordinate axes (i.e. lat/long).
            extents (list, optional): Four element numpy array giving lon/lat limits (e.g. [-4.56, -3.76, 49.96, 50.44])
            draw_coastlines (bool, optional): Draw coastlines. Default False.
            resolution (str, optional): Resolution to use when plotting the coastline. Only used when draw_coastline=True. Default: '10m'.
            **kwargs: Additional keyword arguments passed to matplotlib tripcolor

        Returns:
            tuple: (axes, plot) - Axes object and PolyCollection plot object
        """
        if update:
            for collection in ax.collections:
                if isinstance(collection, PolyCollection):
                    field_masked = field[~self.tri.mask]
                    collection.set_array(field_masked)
                    return ax
            raise RuntimeError(
                "update=True but no existing PolyCollection object found on the axes"
            )

        # If not configuring the plot, simply plot the field and return
        if self.geographic_coords:
            plot = ax.tripcolor(
                self.tri, field, transform=self.transform, **kwargs
            )
        else:
            plot = ax.tripcolor(self.tri, field, **kwargs)

        if not configure:
            return ax, plot

        # Set extents
        if extents is None:
            extents = self._get_default_extents()

        # Create plot
        if self.geographic_coords:
            ax.set_extent(extents, self.transform)

            if draw_coastlines:
                ax.coastlines(resolution=resolution, linewidth=self.line_width)

            if tick_inc:
                self._add_ticks(ax)

            ax.set_xlabel("Longitude (E)", fontsize=self.font_size)
            ax.set_ylabel("Longitude (N)", fontsize=self.font_size)
        else:
            ax.set_xlim(extents[0], extents[1])
            ax.set_ylim(extents[2], extents[3])
            ax.set_xlabel("x (m)", fontsize=self.font_size)
            ax.set_ylabel("y (m)", fontsize=self.font_size)

        # Add colour bar
        if add_colour_bar:
            figure = ax.get_figure()
            self._add_colour_bar(figure, ax, plot, cb_label)

        return ax, plot

    def scatter(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        c: Optional[np.ndarray] = None,
        configure: Optional[bool] = False,
        extents: Optional[list] = None,
        transform: Optional[ccrs.Projection] = None,
        draw_coastlines: Optional[bool] = False,
        resolution: Optional[str] = "10m",
        tick_inc: Optional[bool] = False,
        **kwargs,
    ):
        """Create an FVCOM-specific scatter plot using the provided x and y values

        Args:
            ax (matplotlib.axes.Axes): Axes object
            x (np.ndarray): Array of 'x' positions. If plotting in geographic coords, these should be longitudes.
            y (np.ndarray): Array of 'y' positions. If plotting in geographic coords, these should be latitudes.
            c (np.ndarray, optional): Array of colour values for each point.
            configure (bool, optional): If true, configure the plot by setting extents, coastlines etc.
            extents (list, optional): Four element list giving lon/lat limits
            transform (ccrs.Projection, optional): Transform for geographic projections
            draw_coastlines (bool, optional): Draw coastlines? Only used if geographic_coords is True.
            resolution (str, optional): Coastline resolution. Default: '10m'.
            tick_inc (bool, optional): Draw ticks? Only used if geographic_coords is True.
            **kwargs: Additional keyword arguments passed to matplotlib scatter

        Returns:
            tuple: (ax, scatter_plot) - Axes object and scatter plot collection
        """
        transform = self._check_transform(transform)

        # Check to see if a field has already been plotted
        if not configure:
            if self.geographic_coords:
                scatter_plot = ax.scatter(
                    x, y, c=c, transform=transform, **kwargs
                )
            else:
                scatter_plot = ax.scatter(x, y, c=c, **kwargs)
            return ax, scatter_plot

        # Create a new plot with full configuration
        if extents is None:
            extents = self._get_default_extents()

        if self.geographic_coords:
            scatter_plot = ax.scatter(
                x, y, c=c, transform=transform, **kwargs
            )
            ax.set_extent(extents, transform)

            if draw_coastlines:
                ax.coastlines(resolution=resolution, linewidth=self.line_width)

            if tick_inc:
                self._add_ticks(ax)

            ax.set_xlabel("Longitude (E)", fontsize=self.font_size)
            ax.set_ylabel("Latitude (N)", fontsize=self.font_size)
        else:
            scatter_plot = ax.scatter(x, y, c=c, **kwargs)
            ax.set_xlim(extents[:2])
            ax.set_ylim(extents[2:])

            ax.set_xlabel("x (m)", fontsize=self.font_size)
            ax.set_ylabel("y (m)", fontsize=self.font_size)

        return ax, scatter_plot

    def plot_quiver(
        self,
        ax: matplotlib.axes.Axes,
        u: np.ndarray,
        v: np.ndarray,
        configure: Optional[bool] = True,
        update: Optional[bool] = False,
        tick_inc: Optional[bool] = True,
        extents: Optional[np.ndarray] = None,
        draw_coastlines: Optional[bool] = False,
        resolution: Optional[str] = "10m",
        point_res: Optional[int] = 1,
        scale: Optional[float] = 0.5,
        quiver_key_x: Optional[float] = 0.9,
        quiver_key_y: Optional[float] = 0.9,
        quiver_key_value: Optional[float] = 0.5,
        quiver_key_label: Optional[str] = None,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Produce a quiver plot of the supplied velocity field.

        Args:
            ax (matplotlib.axes.Axes): Axes object on which to plot.
            u (np.ndarray): 1D array of u velocity components defined at element centres.
            v (np.ndarray): 1D array of v velocity components defined at element centres.
            configure (bool, optional): If True, configure the plot by setting plot extents, drawing coastlines etc. Default: True.
            update (bool, optional): If True, update the existing plot. Specifically, the axes will be checked to see if
                it contains a Quiver object. If found, the associated data arrays will be
                updated with the supplied u and v data. This is faster than drawing a new map. Default: False.
            tick_inc (bool, optional): Add coordinate axes (i.e. lat/long). Default: True.
            extents (np.ndarray, optional): Four element numpy array giving lon/lat limits (e.g. [-4.56, -3.76, 49.96, 50.44]).
                If None, will use default extents from the grid. Default: None.
            draw_coastlines (bool, optional): Draw coastlines. Only used if geographic_coords is True. Default: False.
            resolution (str, optional): Resolution to use when plotting the coastline. Only used when draw_coastlines=True.
                Default: '10m'.
            point_res (int, optional): Plot every n-th arrow, where n = point_res. Default: 1 (plot every arrow).
            scale (float, optional): Scaling factor for quiver plot. Default: 0.5.
            quiver_key_x (float, optional): X position for quiver key in axes coordinates. Default: 0.9.
            quiver_key_y (float, optional): Y position for quiver key in axes coordinates. Default: 0.9.
            quiver_key_value (float, optional): Reference velocity value for the quiver key. Default: 0.5.
            quiver_key_label (str, optional): Custom label for the quiver key. If None, will use default format. Default: None.
            **kwargs: Additional keyword arguments passed to matplotlib's quiver function.

        Returns:
            matplotlib.axes.Axes: The axes object with the quiver plot.

        Raises:
            ValueError: If u and v arrays have different shapes, are not 1D, or don't match the number of elements.
            RuntimeError: If update=True but no existing Quiver object is found on the axes.
        """
        # Validate input arrays
        if u.shape != v.shape:
            raise ValueError(f"u and v shapes do not match: {u.shape} vs {v.shape}")

        if len(u.shape) != 1:
            raise ValueError(f"Expected 1D u/v arrays. Array has shape {u.shape}.")

        if u.shape[0] != self.n_elems:
            raise ValueError(
                f"Array size {u.shape[0]} does not match number of elements {self.n_elems}"
            )

        # Set spacing to plot 1 in n arrows where n = point_res
        points = slice(None, None, point_res)

        # Handle updates to existing quiver plots
        if update:
            for collection in ax.collections:
                if isinstance(collection, mpl_quiver.Quiver):
                    collection.set_UVC(u, v)
                    return ax
            raise RuntimeError(
                "update=True but no existing Quiver object found on the axes"
            )

        # Create the quiver plot
        quiver = ax.quiver(
            self.xc[points],
            self.yc[points],
            u[points],
            v[points],
            transform=self.transform,
            units="inches",
            scale_units="inches",
            scale=scale,
            **kwargs,
        )

        # Add quiver key with configurable parameters
        key_label = quiver_key_label or f"{quiver_key_value} " + r"$\mathrm{ms^{-1}}$"
        plt.quiverkey(
            quiver,
            quiver_key_x,
            quiver_key_y,
            quiver_key_value,
            key_label,
            coordinates="axes",
        )

        # If not configuring the rest of the plot return to caller
        if not configure:
            return ax

        # Set extents
        if extents is None:
            extents = self._get_default_extents()

        ax.set_extent(extents, self.transform)

        if draw_coastlines:
            ax.coastlines(resolution=resolution, linewidth=self.line_width)

        if tick_inc:
            self._add_ticks(ax)

        ax.set_xlabel("Longitude (E)", fontsize=self.font_size)
        ax.set_ylabel("Longitude (N)", fontsize=self.font_size)

        return ax

    def draw_grid(self, ax: matplotlib.axes.Axes, **kwargs):
        """Draw the underlying grid or mesh

        Args:
            ax (matplotlib.axes.Axes): Axes object
            **kwargs: Additional keyword arguments passed to matplotlib triplot

        Returns:
            matplotlib.axes.Axes: Axes object
        """
        ax.triplot(self.tri, **kwargs)

        return ax


class CMEMSPlotter(PyFVCOM2Plotter):
    """Class for plotting CMEMS data

    Args:
        cmems_file_name (str): Path to a CMEMS NetCDF file.
        font_size (int, optional): Font size for plot text. Default 10.
        line_width (float, optional): Default line width for plotting. Default 0.2.
    """
    
    def __init__(
        self,
        cmems_file_name: str,
        font_size: Optional[int] = 10,
        line_width: Optional[float] = 0.2,
    ):
        # Initialise base class
        super().__init__(geographic_coords=True, font_size=font_size, line_width=line_width)

        # Open the NetCDF file for reading
        with Dataset(cmems_file_name, "r") as ds:
            # Read grid information
            self._read_grid_information(ds)
        
        # Set the transform
        self.transform = ccrs.PlateCarree()

    def _read_grid_information(self, ds):
        # Read in the required grid variables
        self.lon = ds.variables["longitude"][:]
        self.lat = ds.variables["latitude"][:]
        try:
            self.depth = ds.variables["depth"][:]
        except KeyError:
            self.depth = None

    def _get_default_extents(self):
        return np.array([self.lon.min(), self.lon.max(), self.lat.min(), self.lat.max()])

    def plot_field(
        self,
        ax: matplotlib.axes.Axes,
        field: np.ndarray,
        configure: Optional[bool] = True,
        add_colour_bar: Optional[bool] = True,
        cb_label: Optional[str] = None,
        tick_inc: Optional[bool] = True,
        extents: Optional[list] = None,
        draw_coastlines: Optional[bool] = False,
        resolution: Optional[str] = "10m",
        **kwargs
    ):
        """Map the supplied field

        Additional plotting options are passed to `matplotlib.pyplot.pcolormesh`. See the matplotlib documentation
        for a full list of supported options.

        Args:
            ax (matplotlib.axes.Axes): Axes object.
            field (np.ndarray): The field to plot.
            configure (bool, optional): If true, configure the plot by setting plot extents, drawing coastlines etc. Default: True.
            add_colour_bar (bool, optional): If true, draw a colour bar. Default: True.
            cb_label (str, optional): The colour bar label.
            tick_inc (bool, optional): Add coordinate axes (i.e. lat/long). Default: True.
            extents (list, optional): Four element numpy array giving lon/lat limits (e.g. [-4.56, -3.76, 49.96, 50.44]). Default: None.
            draw_coastlines (bool, optional): Draw coastlines. Default: False.
            resolution (str, optional): Resolution to use when plotting the coastline. Only used when draw_coastline=True. Default: '10m'.
            **kwargs: Additional keyword arguments passed to matplotlib pcolormesh.
        Returns:
            tuple: (axes, plot) - Axes object and PolyCollection plot object.
        """
        # Create the plot
        plot = ax.pcolormesh(
            self.lon,
            self.lat,
            field,
            transform=self.transform,
            **kwargs
        )

        if not configure:
            return ax, plot

        # Set extents
        if extents is None:
            extents = self._get_default_extents()

        # Create plot
        ax.set_extent(extents, self.transform)

        if draw_coastlines:
            ax.coastlines(resolution=resolution, linewidth=self.line_width)

        if tick_inc:
            self._add_ticks(ax)

        ax.set_xlabel("Longitude (E)", fontsize=self.font_size)
        ax.set_ylabel("Longitude (N)", fontsize=self.font_size)

        # Add colour bar
        if add_colour_bar:
            figure = ax.get_figure()
            self._add_colour_bar(figure, ax, plot, cb_label)

        return ax, plot

    def scatter(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        c: Optional[np.ndarray] = None,
        configure: Optional[bool] = False,
        extents: Optional[list] = None,
        transform: Optional[ccrs.Projection] = None,
        draw_coastlines: Optional[bool] = False,
        resolution: Optional[str] = "10m",
        tick_inc: Optional[bool] = False,
        **kwargs,
    ):
        """Create a CMEMS-specific scatter plot using the provided x and y values

        Args:
            ax (matplotlib.axes.Axes): Axes object
            x (np.ndarray): Array of longitude positions.
            y (np.ndarray): Array of latitude positions.
            c (np.ndarray, optional): Array of colour values for each point.
            configure (bool, optional): If true, configure the plot by setting extents, coastlines etc.
            extents (list, optional): Four element list giving lon/lat limits
            transform (ccrs.Projection, optional): Transform for geographic projections (uses self.transform if None)
            draw_coastlines (bool, optional): Draw coastlines.
            resolution (str, optional): Coastline resolution. Default: '10m'.
            tick_inc (bool, optional): Draw ticks.
            **kwargs: Additional keyword arguments passed to matplotlib scatter

        Returns:
            tuple: (ax, scatter_plot) - Axes object and scatter plot collection
        """
        if transform is None:
            transform = self.transform

        # Simple scatter if not configuring
        if not configure:
            scatter_plot = ax.scatter(
                x, y, c=c, transform=transform, **kwargs
            )
            return ax, scatter_plot

        # Full plot configuration
        if extents is None:
            extents = self._get_default_extents()

        scatter_plot = ax.scatter(
            x, y, c=c, transform=transform, **kwargs
        )
        
        ax.set_extent(extents, transform)

        if draw_coastlines:
            ax.coastlines(resolution=resolution, linewidth=self.line_width)

        if tick_inc:
            self._add_ticks(ax)

        ax.set_xlabel("Longitude (E)", fontsize=self.font_size)
        ax.set_ylabel("Latitude (N)", fontsize=self.font_size)

        return ax, scatter_plot


def create_figure(
    figure_size: Optional[tuple] = (10.0, 10.0),
    font_size: Optional[int] = 10,
    axis_position: Optional[list] = None,
    projection: Optional[ccrs.Projection] = None,
    bg_color: Optional[str] = "white",
):
    """Create a Figure object

    Args:
        figure_size (tuple, optional): Figure size in cm. This is only used if a new Figure object is created.
        font_size (int, optional): Font size to use for axis labels
        axis_position (list, optional): Array giving axis dimensions
        projection (ccrs.Projection, optional): Cartopy projection to use for the plot. If None, a projection will not be used.
        bg_color (str, optional): Colour to use for the axis background. Default is `white`. When
            creating a figure for plotting FVCOM outputs, it can be useful
            to set this to `gray`. When FVCOM is fitted to a coastline, the
            gray areas mark the land boundary used by the model. This provides
            a fast alternative to plotting a high resolution (e.g. `res` = `f`)
            land boundary using methods provided by the Basemap class instance.
    
    Returns:
        tuple: (figure, axes) - Figure and Axes objects
    """
    figure_size_inches = (cm2inch(figure_size[0]), cm2inch(figure_size[1]))
    figure = plt.figure(figsize=figure_size_inches)
    figure.set_facecolor("white")

    axes = figure.add_subplot(1, 1, 1, projection=projection, facecolor=bg_color)

    if axis_position is not None:
        axes.set_position(axis_position)

    axes.tick_params(axis="both", which="major", labelsize=font_size)
    axes.tick_params(axis="both", which="minor", labelsize=font_size)

    return figure, axes


def create_cbar_ax(ax: matplotlib.axes.Axes):
    """Create colorbar axis alligned with plot axis y limits

    Args:
        ax (matplotlib.axes.Axes): Plot axes instance

    Returns:
        matplotlib.axes.Axes: Colorbar plot axis
    """
    divider = make_axes_locatable(ax)
    return divider.append_axes("right", size="5%", pad=0.05)


def cm2inch(value: float) -> float:
    """Convert centimetres to inches.

    Args:
        value (float): Length in cm.

    Returns:
        float: Length in inches.
    """
    return value / 2.54


def colourmap(variable: str) -> matplotlib.colors.Colormap:
    """Use a predefined colour map for a given variable.

    Leverages the cmocean package for perceptually uniform colour maps.

    Args:
        variable (str): For the given variable name, return the appropriate colour
            palette from the cmocean/matplotlib colour maps. If the
            variable is not in the pre-defined variables here, the
            returned values will be `viridis`.

    Returns:
        matplotlib.colors.Colormap: The colour map for the variable given.

    """

    default_cmap = mplcm.get_cmap("viridis")

    cmaps = {
        "q2": cm.dense,
        "l": cm.dense,
        "q2l": cm.dense,
        "tke": cm.dense,
        "viscofh": cm.dense,
        "kh": cm.dense,
        "nuh": cm.dense,
        "teps": cm.dense,
        "tauc": cm.dense,
        "temp": cm.thermal,
        "sst": cm.thermal,
        "salinity": cm.haline,
        "zeta": cm.balance,
        "ww": cm.balance,
        "omega": cm.balance,
        "uv": cm.speed,
        "uava": cm.speed,
        "speed": cm.speed,
        "u": cm.delta,
        "v": cm.delta,
        "ua": cm.delta,
        "va": cm.delta,
        "uvanomaly": cm.delta,
        "direction": cm.phase,
        "uvdir": cm.phase,
        "h_morpho": cm.deep,
        "h": cm.deep,
        "h_r": cm.deep_r,
        "bathymetry": cm.deep,
        "bathymetry_r": cm.deep_r,
        "taub_total": cm.thermal,
        "mud_1": cm.turbid,
        "mud_2": cm.turbid,
        "sand_1": cm.turbid,
        "sand_2": cm.turbid,
        "todal_ssc": cm.turbid,
        "total_ssc": cm.turbid,
        "mud_1_bedfrac": cm.dense,
        "mud_2_bedfrac": cm.dense,
        "sand_1_bedfrac": cm.dense,
        "sand_2_bedfrac": cm.dense,
        "mud_1_bedload": cm.dense,
        "mud_2_bedload": cm.dense,
        "sand_1_bedload": cm.dense,
        "sand_2_bedload": cm.dense,
        "bed_thick": cm.deep,
        "bed_age": cm.tempo,
        "bed_por": cm.turbid,
        "bed_diff": cm.haline,
        "bed_btcr": cm.thermal,
        "bot_sd50": cm.turbid,
        "bot_dens": cm.thermal,
        "bot_wsed": cm.turbid,
        "bot_nthck": cm.matter,
        "bot_lthck": cm.matter,
        "bot_dthck": cm.matter,
        "bot_morph": cm.deep,
        "bot_tauc": cm.thermal,
        "bot_rlen": cm.dense,
        "bot_rhgt": cm.dense,
        "bot_bwav": cm.turbid,
        "bot_zdef": cm.dense,
        "bot_zapp": cm.dense,
        "bot_zNik": cm.dense,
        "bot_zbio": cm.dense,
        "bot_zbfm": cm.dense,
        "bot_zbld": cm.dense,
        "bot_zwbl": cm.dense,
        "bot_actv": cm.deep,
        "bot_shgt": cm.deep_r,
        "bot_maxD": cm.deep,
        "bot_dnet": cm.matter,
        "bot_doff": cm.thermal,
        "bot_dslp": cm.amp,
        "bot_dtim": cm.haline,
        "bot_dbmx": cm.dense,
        "bot_dbmm": cm.dense,
        "bot_dbzs": cm.dense,
        "bot_dbzm": cm.dense,
        "bot_dbzp": cm.dense,
        "wet_nodes": cm.amp,
        "tracer1_c": cm.dense,
        "DYE": cm.dense,
    }

    if variable in cmaps:
        colourmaps = cmaps[variable]
    else:
        colourmaps = default_cmap

    return colourmaps
