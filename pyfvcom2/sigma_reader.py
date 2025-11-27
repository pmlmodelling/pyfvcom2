from typing import NamedTuple, Optional
import numpy as np
from pathlib import Path
import scipy

from pyfvcom2.exceptions import PyFVCOM2ValueError

__all__ = [
    "SigmaConfig",
    "read_sigma_file",
    "process_sigma_config", 
    "write_sigma_file",
    "sigma_generalized",
    "sigma_geometric", 
    "sigma_tanh",
    "hybrid_sigma_coordinate",
]


class SigmaConfig(NamedTuple):
    """Named tuple containing sigma coordinate configuration parameters.

    Attributes:
        nlev: Number of sigma levels.
        sigtype: Sigma coordinate type (e.g., 'uniform', 'geometric', 'tanh', 'generalized').
        sigpow: Power value for geometric sigma coordinates.
        du: Upper depth boundary thickness.
        dl: Lower depth boundary thickness.
        min_constant_depth: Minimum water depth for generalized coordinates.
        ku: Number of levels in upper boundary.
        kl: Number of levels in lower boundary.
        zku: Depths of upper boundary layers.
        zkl: Depths of lower boundary layers.
    """

    nlev: int
    sigtype: str
    sigpow: Optional[float] = None
    du: Optional[float] = None
    dl: Optional[float] = None
    min_constant_depth: Optional[float] = None
    ku: Optional[int] = None
    kl: Optional[int] = None
    zku: Optional[np.ndarray] = None
    zkl: Optional[np.ndarray] = None


class SigmaData(NamedTuple):
    """Named tuple containing sigma coordinate data.

    Attrs:
        sigma_config: Sigma configuration parameters.
        sigma_levels: 2D array of sigma levels at each node.
    """

    sigma_config: SigmaConfig
    sigma_levels: np.ndarray


def read_sigma_file(filepath: str) -> SigmaConfig:
    """Read in a sigma coordinates file and return parsed configuration.

    Args:
        filepath: Path to FVCOM sigma coordinates .dat file.

    Returns:
        Named tuple containing all sigma coordinate parameters.

    Notes:
        This is more or less a direct python translation of the original
        MATLAB fvcom-toolbox function read_sigma.m
    """

    # Initialize variables with None/default values
    nlev = None
    sigtype = None
    sigpow = None
    du = None
    dl = None
    min_constant_depth = None
    ku = None
    kl = None
    zku = None
    zkl = None

    with open(filepath, "r") as f:
        lines = f.readlines()
        for line in lines:
            if not line.strip("\n").strip("\r"):
                continue
            line = line.strip()
            option, value = line.split("=")
            option = option.strip().lower()
            value = value.strip()

            # Grab the various bits we need.
            if option == "number of sigma levels":
                nlev = int(value)
            elif option == "sigma coordinate type":
                sigtype = value.lower()
            elif option == "sigma power":
                sigpow = float(value)
            elif option == "du":
                du = float(value)
            elif option == "dl":
                dl = float(value)
            elif option == "min constant depth":
                min_constant_depth = float(value)
            elif option == "ku":
                ku = int(value)
            elif option == "kl":
                kl = int(value)
            elif option == "zku":
                s = [float(i) for i in value.split()]
                zku = np.zeros(ku)
                for i in range(ku):
                    zku[i] = s[i]
            elif option == "zkl":
                s = [float(i) for i in value.split()]
                zkl = np.zeros(kl)
                for i in range(kl):
                    zkl[i] = s[i]

    return SigmaConfig(
        nlev=nlev,
        sigtype=sigtype,
        sigpow=sigpow,
        du=du,
        dl=dl,
        min_constant_depth=min_constant_depth,
        ku=ku,
        kl=kl,
        zku=zku,
        zkl=zkl,
    )


def process_sigma_config(config: SigmaConfig, h: np.ndarray) -> SigmaData:
    """Read in a sigma coordinates file and generate sigma coordinate parameters.

    Args:
        config: Sigma configuration parameters.
        h: Bathymetry at nodes.

    Returns:
        SigmaData: A named tuple containing:
            - sigma_config (SigmaConfig): Sigma configuration as read from file.
            - sigma_levels (np.ndarray): Sigma levels at each node.
    """
    # Derive n_nodes and n_elems from h and hc
    n_nodes = h.shape[0]

    # Calculate the sigma level distributions at each grid node.
    if config.sigtype.lower() == "generalized":
        # Do some checks if we've got uniform or generalised coordinates
        # to make sure the input is correct.
        if len(config.zku) != config.ku:
            raise ValueError(
                "Number of zku values does not match the " + "number specified in ku"
            )
        if len(config.zkl) != config.kl:
            raise ValueError(
                "Number of zkl values does not match the " + "number specified in kl"
            )
        sigma_levels = np.empty((n_nodes, config.nlev)) * np.nan
        for i in range(n_nodes):
            sigma_levels[i, :] = sigma_generalized(
                config.nlev,
                config.dl,
                config.du,
                h[i],
                config.min_constant_depth,
                config.kl,
                config.ku,
                zkl=config.zkl,
                zku=config.zku,
            )
    elif config.sigtype == "uniform":
        sigma_levels = np.tile(sigma_geometric(config.nlev, 1), (n_nodes, 1))
    elif config.sigtype == "geometric":
        sigma_levels = np.tile(
            sigma_geometric(config.nlev, config.sigpow), (n_nodes, 1)
        )
    elif config.sigtype == "tanh":
        sigma_levels = np.tile(
            sigma_tanh(config.nlev, config.dl, config.du), (n_nodes, 1)
        )
    else:
        raise ValueError(
            "Unrecognised sigtype " + "{} (is it supported?)".format(config.sigtype)
        )

    return SigmaData(
        sigma_config=config,
        sigma_levels=sigma_levels
    )


def write_sigma_file(sigma_config: SigmaConfig, sigma_file: str):
    """Write the sigma distribution to file.

    Args:
        sigma_config: Sigma configuration to write.
        sigma_file: Path to which to save sigma data.

    Todo:
        Add support for writing all the sigma file formats.
    """

    sigma_file = Path(sigma_file)

    with sigma_file.open("w") as f:
        # All types of sigma distribution have the two following lines.
        f.write("NUMBER OF SIGMA LEVELS = {:d}\n".format(sigma_config.nlev))
        f.write("SIGMA COORDINATE TYPE = {}\n".format(sigma_config.sigtype))
        if sigma_config.sigtype.lower() == "generalized":
            f.write("DU = {:4.1f}\n".format(sigma_config.du))
            f.write("DL = {:4.1f}\n".format(sigma_config.dl))
            # Why do we go to all the trouble of finding the transition depth only to round it anyway?
            f.write(
                "MIN CONSTANT DEPTH = {:10.1f}\n".format(
                    np.round(sigma_config.min_constant_depth)
                )
            )
            f.write("KU = {:d}\n".format(sigma_config.ku))
            f.write("KL = {:d}\n".format(sigma_config.kl))
            # Add the thicknesses with a loop.
            f.write("ZKU = ")
            for ii in sigma_config.zku:
                f.write("{:4.1f}".format(ii))
            f.write("\n")
            f.write("ZKL = ")
            for ii in sigma_config.zkl:
                f.write("{:4.1f}".format(ii))
            f.write("\n")
        elif sigma_config.sigtype.lower() == "geometric":
            f.write("SIGMA POWER = {:.1f}\n".format(sigma_config.sigpow))


def sigma_generalized(
    levels: int,
    dl: float,
    du: float,
    h: float,
    hmin: float,
    kl: int,
    ku: int,
    zku: list = [],
    zkl: list = [],
) -> np.ndarray:
    """Generate a generalised sigma coordinate distribution.

    Args:
        levels: Number of sigma levels.
        dl: The lower depth boundary from the bottom, down to which the layers are uniform thickness.
        du: The upper depth boundary from the surface, up to which the layers are uniform thickness.
        h: Water depth (positive down).
        hmin: Minimum water depth (positive down).
        kl: Number of levels in lower boundary.
        ku: Number of levels in upper boundary.
        zku: Depths of upper boundary layers. Calculated evenly if not given.
        zkl: Depths of lower boundary layers. Calculated evenly if not given.

    Returns:
        Generalised vertical sigma coordinate distribution.
    """

    # Make sure we have positive down depths by nuking negatives.
    h = np.abs(h)
    hmin = np.abs(hmin)

    # Check that the uniform --> generalised transition makes sense
    if not hmin / (levels - 1) == dl / kl or not hmin / (levels - 1) == du / ku:
        raise PyFVCOM2ValueError("Uniform to generalised sigma layers are not matched")

    if h > hmin:
        # Fixed layers for the top and bottom, can assume they are the same thickness as this
        # is checked above
        if (len(zkl) + len(zku)) == 0:
            perc_in_boundary = (dl + du) / h
            layers_in_boundary = ku + kl
            perc_per_layer_boundary = perc_in_boundary / layers_in_boundary

            # Uniform in between
            perc_in_midlayer = (
                h - (dl + du) * (1 - 1 / layers_in_boundary)
            ) / h  # how much of the water column still to divvy up
            layers_in_mid = levels - (ku + kl) - 1  # 0 m is defined
            perc_per_layer_mid = perc_in_midlayer / layers_in_mid

            # Compile it into one set of dists
            dist = [0]
            for i in np.arange(1, ku + 1):
                dist.append(dist[-1] + perc_per_layer_boundary)
            for i in np.arange(0, layers_in_mid):
                dist.append(dist[-1] + perc_per_layer_mid)
            for i in np.arange(0, kl):
                dist.append(dist[-1] + perc_per_layer_boundary)
            dist = np.asarray(dist) * -1

        else:
            perc_zku = zku / h
            perc_zkl = zkl / h
            # Uniform in between
            layers_in_boundary = ku + kl
            perc_in_midlayer = (
                h - (dl + du) * (1 - 1 / layers_in_boundary)
            ) / h  # how much of the water column still to divvy up
            layers_in_mid = levels - (ku + kl) - 1  # 0 m is defined
            perc_per_layer_mid = perc_in_midlayer / layers_in_mid

            # Compile it into one set of dists
            dist = [0]
            for i in np.arange(0, ku):
                dist.append(dist[-1] + perc_zku[i])
            for i in np.arange(0, layers_in_mid):
                dist.append(dist[-1] + perc_per_layer_mid)
            for i in np.arange(0, kl):
                dist.append(dist[-1] + perc_zkl[i])
            dist = np.asarray(dist) * -1

    else:
        # Uniform for shallow areas
        dist = sigma_geometric(levels, 1)

    return dist


def sigma_geometric(levels: int, p_sigma: int) -> np.ndarray:
    """Generate a geometric sigma coordinate distribution.

    Args:
        levels: Number of sigma levels.
        p_sigma: Power value. 1 for uniform sigma layers, 2 for parabolic function.
            See page 308-309 in the FVCOM manual for examples.

    Returns:
        Geometric vertical sigma coordinate distribution.
    """

    dist = np.empty(levels) * np.nan

    if p_sigma == 1:
        for k in range(1, levels + 1):
            dist[k - 1] = -(((k - 1) / (levels - 1)) ** p_sigma)
    else:
        split = int(np.floor((levels + 1) / 2))
        for k in range(split):
            dist[k] = -((k / ((levels + 1) / 2 - 1)) ** p_sigma) / 2
        # Mirror the first half to make the second half of the parabola. We need to offset by one if we've got an
        # odd number of levels.
        if levels % 2 == 0:
            dist[split:] = -(1 - -dist[:split])[::-1]
        else:
            dist[split:] = -(1 - -dist[: split - 1])[::-1]

    return dist


def sigma_tanh(levels: int, dl: float, du: float) -> np.ndarray:
    """Generate a hyperbolic tangent vertical sigma coordinate distribution.

    Args:
        levels: Number of sigma levels (layers + 1).
        dl: The lower depth boundary from the bottom down to which the coordinates
            are parallel with uniform thickness.
        du: The upper depth boundary from the surface up to which the coordinates
            are parallel with uniform thickness.

    Returns:
        Hyperbolic tangent vertical sigma coordinate distribution.
    """

    kbm1 = levels - 1

    dist = np.zeros(levels)

    # Loop has to go to kbm1 + 1 (or levels) since python ranges stop before the end point.
    for k in range(1, levels):
        x1 = dl + du
        x1 = x1 * (kbm1 - k) / (kbm1)
        x1 = x1 - dl
        x1 = np.tanh(x1)
        x2 = np.tanh(dl)
        x3 = x2 + np.tanh(du)
        # k'th position starts from 1 which is right because we want the initial value to be zero for sigma levels.
        dist[k] = (x1 + x2) / x3 - 1


def hybrid_sigma_coordinate(
    levels: int,
    transition_depth: float,
    upper_layer_depth: float,
    lower_layer_depth: float,
    total_upper_layers: int,
    total_lower_layers: int,
    h: np.ndarray,
) -> SigmaData:
    """Create a hybrid vertical coordinate system.

    Args:
        levels: Number of vertical levels.
        transition_depth: Transition depth of the hybrid coordinates.
        upper_layer_depth: Upper water boundary thickness (metres).
        lower_layer_depth: Lower water boundary thickness (metres).
        total_upper_layers: Number of layers in the DU water column.
        total_lower_layers: Number of layers in the DL water column.
        h: Water depth at nodes.

    Returns:
        SigmaData: A named tuple containing:
            - sigma_config (SigmaConfig): Configuration for the sigma coordinate system.
            - sigma_levels (np.ndarray): Sigma levels at each node.
    """
    # Compute number of nodes
    n_nodes = h.shape[0]

    # Optimise the transition depth to minimise the error between the uniform region and the hybrid region.
    upper_layer_thickness = np.repeat(
        upper_layer_depth / total_upper_layers, total_upper_layers
    )
    lower_layer_thickness = np.repeat(
        lower_layer_depth / total_lower_layers, total_lower_layers
    )
    optimisation_settings = {
        "maxfun": 5000,
        "maxiter": 5000,
        "ftol": 10e-5,
        "xtol": 1e-7,
    }
    fparams = lambda depth_guess: _hybrid_coordinate_hmin(
        depth_guess,
        levels,
        upper_layer_depth,
        lower_layer_depth,
        total_upper_layers,
        total_lower_layers,
        upper_layer_thickness,
        lower_layer_thickness,
    )
    optimised_depth = scipy.optimize.fmin(
        func=fparams, x0=transition_depth, disp=False, **optimisation_settings
    )
    min_error = (
        transition_depth - optimised_depth
    )  # TODO Original comment - "this isn't right"!
    transition_depth = optimised_depth

    print(
        f"Hmin found {optimised_depth} with a maximum error in vertical distribution of {min_error} metres\n"
    )

    # Calculate the sigma level distributions at each grid node.
    sigma_levels = np.empty((n_nodes, levels)) * np.nan
    for i in range(n_nodes):
        sigma_levels[i, :] = sigma_generalized(
            levels, lower_layer_depth, upper_layer_depth, h[i], optimised_depth
        )

    # Create the sigma configuration object.
    sigma_config = SigmaConfig(
        nlev=levels,
        sigtype="generalized",
        du=upper_layer_depth,
        dl=lower_layer_depth,
        min_constant_depth=optimised_depth,
        ku=total_upper_layers,
        kl=total_lower_layers,
        zku=upper_layer_thickness,
        zkl=lower_layer_thickness,
    )

    return SigmaData(
        sigma_config=sigma_config,
        sigma_levels=sigma_levels
    )


def _hybrid_coordinate_hmin(
    h: float,
    levels: int,
    du: float,
    dl: float,
    ku: int,
    kl: int,
    zku: np.ndarray,
    zkl: np.ndarray,
):
    """Helper function to find the relevant minimum depth.

    Args:
        h: Transition depth of the hybrid coordinates.
        levels: Number of vertical levels (layers + 1).
        du: Upper water boundary thickness (metres).
        dl: Lower water boundary thickness (metres).
        ku: Layer number in the water column of DU.
        kl: Layer number in the water column of DL.
        zku: Upper layer thicknesses.
        zkl: Lower layer thicknesses.

    Returns:
        Minimum water depth.
    """
    # This is essentially identical to sigma_tanh, so we should probably just use that instead.
    z0 = sigma_tanh(levels, du, dl)
    z2 = np.zeros(levels)

    # s-coordinates
    x1 = h - du - dl
    x2 = x1 / h
    dr = x2 / (levels - ku - kl - 1)

    for k in range(1, ku + 1):
        z2[k] = z2[k - 1] - (zku[k - 1] / h)

    for k in range(ku + 2, levels - kl):
        z2[k] = z2[k - 1] - dr

    kk = 0
    for k in range(levels - kl + 1, levels):
        kk += 1
        z2[k] = z2[k - 1] - (zkl[kk] / h)

    zz = np.max(h * z0 - h * z2)

    return zz
