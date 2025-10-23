"""FVCOM Plotting Functions"""

from typing import Optional
import numpy as np
from netCDF4 import Dataset
import stripy as stripy

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib import cm as mplcm
from matplotlib import quiver as mpl_quiver
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cmocean import cm


class PyFVCOM2Plotter:
    """Base class for PyFVCOM2 plotters

    Class to assist in the creation of plots and animations. The class can
    be used to create a set of basic plot objects. Plots that overlay
    particle trajectories on top of underlying field data should be created
    using the appropriate derived class.

    Parameters
    ----------
    geographic_coords : boolean, optional
        Boolean specifying whether or not to use cartopy to create a 2D map
        on top of which the data will be plotted. The default option is
        `True`. If `False`, a simple Cartesian grid is drawn instead.

    font_size : int, optional
        Font size to use when rendering plot text

    line_width : float, optional
        Default line width to use when plotting

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

        self.current_zorder = 1

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

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object

        x : ND array
            Array of x coordinates to plot.

        y : ND array
            Array of y coordinates to plot.

        Returns
        -------
        axes : matplotlib.axes.Axes
            Axes object

        line_plot : matplotlib.collections.Line2D
            The plot object
        """
        transform = self._check_transform(transform)

        # Use some better default attributes if they have not been supplied
        alpha = kwargs.pop("alpha", 0.25)
        color = kwargs.pop("color", "r")
        linewidth = kwargs.pop("linewidth", 1.0)

        line_plots = ax.plot(
            x,
            y,
            zorder=3,
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
                f"Plotting in geographic coordinates but not transform supplied. Using PlateCarree. "
                f"You can override this by supplying a transform argument."
            )
            _transform = ccrs.PlateCarree()

        return _transform

    def _get_zorder(self):
        """Get the zorder for plotting

        Returns
        -------
        zorder : int
            The zorder to use for plotting
        """
        zorder = self.current_zorder
        self.current_zorder += 1
        return zorder

    def remove_line_plots(self, line_plots: list):
        """Remove line plots

        Useful when updating plots for animations.

        Parameters
        ----------
        line_plots : list
            List of line plot objects created during call to plot_lines()
        """
        while line_plots:
            line_plots.pop(0).remove()

        return

    def scatter(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
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

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object

        x : 1D array
            Array of 'x' positions. If plotting in geographic coords, these should be longitudes.

        y : 1D array
            Array of 'y' positions. If plotting in geographic coords, these should be latitudes.

        configure : bool, optional
            If true, configure the plot by setting plot extents, drawing coastlines etc. Default: False.

        extents : list, optional
            Four element list giving lon/lat limits (e.g. [-4.56, -3.76, 55.12, 55.84])

        transform : cartopy.crs.Projection
            The type of transform to perform if geographic_coords is True. Optional.

        draw_coastlines : bool
            Draw coastlines? Only used if geographic_coords is True. Optional.

        resolution : str, optional
            Resolution to use when plotting the coastline. Only used when draw_coastline=True. Default: '10m'.

        tick_inc : bool
            Draw ticks? Only used if geographic_coords is True. Optional.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes object

        scatter_plot : matplotlib.collection.PathCollection
            The scatter plot
        """
        transform = self._check_transform(transform)

        zorder = self._get_zorder()

        # Check to see if a field has already been plotted, indicating we can simply make
        # the scatter plot without setting up the plot axes in full.
        if not configure:
            if self.geographic_coords:
                scatter_plot = ax.scatter(
                    x, y, transform=transform, zorder=zorder, **kwargs
                )
            else:
                scatter_plot = ax.scatter(x, y, zorder=zorder, **kwargs)

            return ax, scatter_plot

        # Create a new plot
        # -----------------

        # Set extents
        if extents is None:
            extents = self._get_default_extents()

        # Create plot
        if self.geographic_coords:
            scatter_plot = ax.scatter(
                x, y, transform=transform, zorder=zorder, **kwargs
            )
            ax.set_extent(extents, transform)

            if draw_coastlines:
                ax.coastlines(resolution=resolution, linewidth=self.line_width)

            if tick_inc:
                self._add_ticks(ax)
        else:
            scatter_plot = ax.scatter(x, y, zorder=zorder, **kwargs)
            ax.set_xlim(extents[:2])
            ax.set_ylim(extents[2:])

            ax.set_xlabel("x (m)", fontsize=self.font_size)
            ax.set_ylabel("y (m)", fontsize=self.font_size)

        return ax, scatter_plot

    def set_title(self, ax, title):
        """Set the title

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object

        title : str
            Plot title
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
    """Create FVCOM plot objects based on FVCOM model outputs

    Class to assist in the creation of plots and animations based on FVCOM
    data.

    Parameters
    ----------
    fvcom_file_name : Dataset or str
        Path to a FVCOM NetCDF file.
    """

    def __init__(
        self,
        fvcom_file_name: str,
        geographic_coords: Optional[bool] = True,
        font_size: Optional[int] = 10,
        line_width: Optional[float] = 0.2,
    ):
        # Initialise base class
        super().__init__(geographic_coords, font_size, line_width)

        # Open the NetCDF file for reading
        with Dataset(fvcom_file_name, "r") as ds:
            # Read grid information
            self._read_grid_information(ds)

    def _read_grid_information(self, ds):
        # Read in the required grid variables
        self.n_nodes = ds.dimensions["node"].size
        self.n_elems = ds.dimensions["element"].size
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
        **kwargs,
    ):
        """Map the supplied field

        The field must be defined on the same triangular mesh that was used to initialise the plotter.

        Additional plotting options are passed to `matplotlib.pyplot.pcolormesh`. See the matplotlib documentation
        for a full list of supported options.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object

        field : 1D NumPy array
            The field to plot.

        update : bool, optional
            If true, update the existing plot. Specifically, the axes will be checked to see if it contains a
            PolyCollection object, as generated by tripcolor. If found, the associated data array will be
            updated with the supplied field data. This is faster than drawing a new map

        configure : bool, optional
            If true, configure the plot by setting plot extents, drawing coastlines etc. This can be
            useful when overlaying plots, and you only want to incur the cost of configuring the plot
            once. The default is True, with the expectation that in most circumstances users will
            draw any underlying field data before overlaying particle tracks. Default: True.

        add_colour_bar : bool, optional
            If true, draw a colour bar.

        cb_label : str, optional
            The colour bar label.

        tick_inc : bool, optional
            Add coordinate axes (i.e. lat/long).

        extents : 1D array, optional
            Four element numpy array giving lon/lat limits (e.g. [-4.56, -3.76,
            49.96, 50.44])

        draw_coastlines : boolean, optional
            Draw coastlines. Default False.

        resolution : str, optional
            Resolution to use when plotting the coastline. Only used when draw_coastline=True. Default: '10m'.

        Returns
        -------
        axes : matplotlib.axes.Axes
            Axes object

        plot : matplotlib.collections.PolyCollection
            The plot object
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

        # Determine current zorder
        zorder = self._get_zorder()

        # If not configuring the plot, simply plot the field and return
        if self.geographic_coords:
            plot = ax.tripcolor(
                self.tri, field, transform=self.transform, zorder=zorder, **kwargs
            )
        else:
            plot = ax.tripcolor(self.tri, field, zorder=zorder, **kwargs)

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

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object on which to plot.
        u : np.ndarray
            1D array of u velocity components defined at element centres.
        v : np.ndarray
            1D array of v velocity components defined at element centres.
        configure : bool, optional
            If True, configure the plot by setting plot extents, drawing coastlines etc. Default: True.
        update : bool, optional
            If True, update the existing plot. Specifically, the axes will be checked to see if
            it contains a Quiver object. If found, the associated data arrays will be
            updated with the supplied u and v data. This is faster than drawing a new map. Default: False.
        tick_inc : bool, optional
            Add coordinate axes (i.e. lat/long). Default: True.
        extents : np.ndarray, optional
            Four element numpy array giving lon/lat limits (e.g. [-4.56, -3.76, 49.96, 50.44]).
            If None, will use default extents from the grid. Default: None.
        draw_coastlines : bool, optional
            Draw coastlines. Only used if geographic_coords is True. Default: False.
        resolution : str, optional
            Resolution to use when plotting the coastline. Only used when draw_coastlines=True.
            Default: '10m'.
        point_res : int, optional
            Plot every n-th arrow, where n = point_res. Default: 1 (plot every arrow).
        scale : float, optional
            Scaling factor for quiver plot. Default: 0.5.
        quiver_key_x : float, optional
            X position for quiver key in axes coordinates. Default: 0.9.
        quiver_key_y : float, optional
            Y position for quiver key in axes coordinates. Default: 0.9.
        quiver_key_value : float, optional
            Reference velocity value for the quiver key. Default: 0.5.
        quiver_key_label : str, optional
            Custom label for the quiver key. If None, will use default format. Default: None.
        **kwargs
            Additional keyword arguments passed to matplotlib's quiver function.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the quiver plot.

        Raises
        ------
        ValueError
            If u and v arrays have different shapes, are not 1D, or don't match the number of elements.
        RuntimeError
            If update=True but no existing Quiver object is found on the axes.
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
        zorder = self._get_zorder()
        quiver = ax.quiver(
            self.xc[points],
            self.yc[points],
            u[points],
            v[points],
            transform=self.transform,
            units="inches",
            scale_units="inches",
            scale=scale,
            zorder=zorder,
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

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes object
        """
        zorder = self._get_zorder()
        ax.triplot(self.tri, zorder=zorder, **kwargs)

        return ax


def create_figure(
    figure_size: Optional[tuple] = (10.0, 10.0),
    font_size: Optional[int] = 10,
    axis_position: Optional[list] = None,
    projection: Optional[ccrs.Projection] = None,
    bg_color: Optional[str] = "white",
):
    """Create a Figure object

    Parameters
    ----------
    figure_size : tuple(float), optional
        Figure size in cm. This is only used if a new Figure object is
        created.

    font_size : int
        Font size to use for axis labels

    axis_position : 1D array, optional
        Array giving axis dimensions

    projection : ccrs.Projection
        Cartopy projection to use for the plot. If None, a projection will not be used.

    bg_color : str, optional
        Colour to use for the axis background. Default is `white`. When
        creating a figure for plotting FVCOM outputs, it can be useful
        to set this to `gray`. When FVCOM is fitted to a coastline, the
        gray areas mark the land boundary used by the model. This provides
        a fast alternative to plotting a high resolution (e.g. `res` = `f`)
        land boundary using methods provided by the Basemap class instance.
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

    Parameters
    ----------
    ax : Axes
        Plot axes instsance

    Returns
    -------
    cax : Axes
        Colorbar plot axis
    """
    divider = make_axes_locatable(ax)
    return divider.append_axes("right", size="5%", pad=0.05)


def cm2inch(value: float) -> float:
    """Convert centimetres to inches.

    Parameters
    ----------
    value : float
        Length in cm.

    Returns
    -------
     : float
         Length in inches.
    """
    return value / 2.54


def colourmap(variable: str) -> matplotlib.colors.Colormap:
    """Use a predefined colour map for a given variable.

    Leverages the cmocean package for perceptually uniform colour maps.

    Parameters
    ----------
    variable : str
        For the given variable name, return the appropriate colour
        palette from the cmocean/matplotlib colour maps. If the
        variable is not in the pre-defined variables here, the
        returned values will be `viridis`.

    Returns
    -------
    colourmaps : matplotlib.colours.cmap
        The colour map for the variable given.

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
