from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .exceptions import PyFVCOM2ValueError


__all__ = ["NamelistManager"]


class _NamelistEntry:
    """A single entry in an FVCOM namelist section.

    Args:
        name: The FVCOM namelist key name.
        value: The entry value. ``bool`` is automatically converted to
            ``'T'``/``'F'``. Set to ``None`` for entries that must be
            supplied by the user before writing.
        fmt: A Python format specification string (e.g. ``'f'``, ``'.10f'``,
            ``'d'``, ``'s'``). Defaults to ``'s'`` (plain string).
        no_quote: When ``True``, string values are written without surrounding
            quotes. ``'T'``/``'F'`` values are always unquoted regardless.
    """

    def __init__(
        self,
        name: str,
        value,
        fmt: str = "s",
        no_quote: bool = False,
    ) -> None:
        self.name = name
        self.fmt = fmt
        self._no_quote = no_quote
        self.value = value  # invoke setter

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        if isinstance(v, bool):
            v = "T" if v else "F"
        self._value = v

    def as_string(self) -> str:
        """Return the entry as a formatted FVCOM namelist line (no trailing newline)."""
        unquoted = self._no_quote or self._value in ("T", "F")
        if self.fmt == "s":
            if unquoted:
                return f" {self.name} = {self._value}"
            else:
                return f" {self.name} = '{self._value}'"
        else:
            return f" {self.name} = {self._value:{self.fmt}}"


class NamelistManager:
    """Manager for FVCOM run namelists.

    Holds a complete FVCOM namelist configuration with sensible defaults and
    exposes methods to update individual entries before writing to an ASCII
    ``.nml`` file.

    The defaults correspond to a barotropic, tide-only cold-start run with no
    surface forcing, rivers, or data assimilation.  The only mandatory entries
    that have no defaults are ``NML_CASE / START_DATE`` and
    ``NML_CASE / END_DATE``; everything else is pre-populated.

    Args:
        casename: Model case name used to construct default file-path strings.
            Defaults to ``'casename'``.
        fabm: If ``True``, add FABM output controls to the relevant sections
            and append an ``NML_FABM`` section. Defaults to ``False``.

    Examples:
        >>> nml = NamelistManager(casename='tamar_v09')
        >>> nml.update('NML_CASE', 'START_DATE', '2020-05-01 00:00:00')
        >>> nml.update('NML_CASE', 'END_DATE',   '2020-06-01 00:00:00')
        >>> nml.update('NML_INTEGRATION', 'EXTSTEP_SECONDS', 2.0)
        >>> nml.update('NML_PHYSICS', 'TEMPERATURE_ACTIVE', True)
        >>> nml.update_ramp(24)        # 24-hour ramp
        >>> nml.write('tamar_v09_run.nml')
    """

    def __init__(self, casename: str = "casename", fabm: bool = False) -> None:
        self._casename = casename
        self._fabm = fabm
        self._config: dict[str, list[_NamelistEntry]] = self._build_default_config()
        if fabm:
            self._add_fabm_config()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_default_config(self) -> dict[str, list[_NamelistEntry]]:
        c = self._casename
        E = _NamelistEntry
        return {
            "NML_CASE": [
                E("CASE_TITLE", "pyfvcom2 default run"),
                E("TIMEZONE", "UTC"),
                E("DATE_FORMAT", "YMD"),
                E("DATE_REFERENCE", "default"),
                E("START_DATE", None),
                E("END_DATE", None),
            ],
            "NML_STARTUP": [
                E("STARTUP_TYPE", "coldstart"),
                E("STARTUP_FILE", f"{c}_restart.nc"),
                E("STARTUP_UV_TYPE", "default"),
                E("STARTUP_TURB_TYPE", "default"),
                E("STARTUP_TS_TYPE", "constant"),
                E("STARTUP_T_VALS", 15.0, "f"),
                E("STARTUP_S_VALS", 35.0, "f"),
                E("STARTUP_U_VALS", 0.0, "f"),
                E("STARTUP_V_VALS", 0.0, "f"),
                E("STARTUP_DMAX", -3.0, "f"),
            ],
            "NML_IO": [
                E("INPUT_DIR", "./input"),
                E("OUTPUT_DIR", "./output"),
                E("IREPORT", 300, "d"),
                E("VISIT_ALL_VARS", "F"),
                E("WAIT_FOR_VISIT", "F"),
                E("USE_MPI_IO_MODE", "F"),
            ],
            "NML_INTEGRATION": [
                E("EXTSTEP_SECONDS", 1.0, "f"),
                E("ISPLIT", 10, "d"),
                E("IRAMP", 1, "d"),
                E("MIN_DEPTH", 0.2, "f"),
                E("STATIC_SSH_ADJ", 0.0, "f"),
            ],
            "NML_RESTART": [
                E("RST_ON", "T"),
                E("RST_FIRST_OUT", None),
                E("RST_OUT_INTERVAL", "seconds=86400."),
                E("RST_OUTPUT_STACK", 0, "d"),
            ],
            "NML_NETCDF": [
                E("NC_ON", "T"),
                E("NC_FIRST_OUT", None),
                E("NC_OUT_INTERVAL", "seconds=900."),
                E("NC_OUTPUT_STACK", 0, "d"),
                E("NC_SUBDOMAIN_FILES", "FVCOM"),
                E("NC_GRID_METRICS", "T"),
                E("NC_FILE_DATE", "T"),
                E("NC_VELOCITY", "T"),
                E("NC_SALT_TEMP", "T"),
                E("NC_TURBULENCE", "T"),
                E("NC_AVERAGE_VEL", "T"),
                E("NC_VERTICAL_VEL", "T"),
                E("NC_WIND_VEL", "F"),
                E("NC_ATM_PRESS", "F"),
                E("NC_WIND_STRESS", "F"),
                E("NC_EVAP_PRECIP", "F"),
                E("NC_SURFACE_HEAT", "F"),
                E("NC_GROUNDWATER", "F"),
                E("NC_BIO", "F"),
                E("NC_WQM", "F"),
                E("NC_VORTICITY", "F"),
            ],
            "NML_NETCDF_SURFACE": [
                E("NCSF_ON", "F"),
                E("NCSF_FIRST_OUT", None),
                E("NCSF_OUT_INTERVAL", "seconds=900."),
                E("NCSF_OUTPUT_STACK", 0, "d"),
                E("NCSF_SUBDOMAIN_FILES", "FVCOM"),
                E("NCSF_GRID_METRICS", "F"),
                E("NCSF_FILE_DATE", "F"),
                E("NCSF_VELOCITY", "F"),
                E("NCSF_SALT_TEMP", "F"),
                E("NCSF_TURBULENCE", "F"),
                E("NCSF_WIND_VEL", "F"),
                E("NCSF_ATM_PRESS", "F"),
                E("NCSF_WIND_STRESS", "F"),
                E("NCSF_WAVE_PARA", "F"),
                E("NCSF_ICE", "F"),
                E("NCSF_EVAP_PRECIP", "F"),
                E("NCSF_SURFACE_HEAT", "F"),
            ],
            "NML_NETCDF_AV": [
                E("NCAV_ON", "F"),
                E("NCAV_FIRST_OUT", None),
                E("NCAV_OUT_INTERVAL", "seconds=86400."),
                E("NCAV_OUTPUT_STACK", 0, "d"),
                E("NCAV_GRID_METRICS", "T"),
                E("NCAV_FILE_DATE", "T"),
                E("NCAV_VELOCITY", "T"),
                E("NCAV_SALT_TEMP", "T"),
                E("NCAV_TURBULENCE", "T"),
                E("NCAV_AVERAGE_VEL", "T"),
                E("NCAV_VERTICAL_VEL", "T"),
                E("NCAV_WIND_VEL", "F"),
                E("NCAV_ATM_PRESS", "F"),
                E("NCAV_WIND_STRESS", "F"),
                E("NCAV_EVAP_PRECIP", "F"),
                E("NCAV_SURFACE_HEAT", "F"),
                E("NCAV_GROUNDWATER", "F"),
                E("NCAV_BIO", "F"),
                E("NCAV_WQM", "F"),
                E("NCAV_VORTICITY", "F"),
            ],
            "NML_SURFACE_FORCING": [
                E("WIND_ON", "F"),
                E("WIND_TYPE", "speed"),
                E("WIND_FILE", f"{c}_wnd.nc"),
                E("WIND_KIND", "variable"),
                E("WIND_X", 5.0, "f"),
                E("WIND_Y", 5.0, "f"),
                E("HEATING_ON", "F"),
                E("HEATING_TYPE", "flux"),
                E("HEATING_KIND", "variable"),
                E("HEATING_FILE", f"{c}_wnd.nc"),
                E("HEATING_LONGWAVE_LENGTHSCALE", 0.7, "f"),
                E("HEATING_LONGWAVE_PERCTAGE", 10, "f"),
                E("HEATING_SHORTWAVE_LENGTHSCALE", 1.1, "f"),
                E("HEATING_RADIATION", 0.0, "f"),
                E("HEATING_NETFLUX", 0.0, "f"),
                E("PRECIPITATION_ON", "F"),
                E("PRECIPITATION_KIND", "variable"),
                E("PRECIPITATION_FILE", f"{c}_wnd.nc"),
                E("PRECIPITATION_PRC", 0.0, "f"),
                E("PRECIPITATION_EVP", 0.0, "f"),
                E("AIRPRESSURE_ON", "F"),
                E("AIRPRESSURE_KIND", "variable"),
                E("AIRPRESSURE_FILE", f"{c}_wnd.nc"),
                E("AIRPRESSURE_VALUE", 0.0, "f"),
                E("WAVE_ON", "F"),
                E("WAVE_FILE", f"{c}_wav.nc"),
                E("WAVE_KIND", "constant"),
                E("WAVE_HEIGHT", 0.0, "f"),
                E("WAVE_LENGTH", 0.0, "f"),
                E("WAVE_DIRECTION", 0.0, "f"),
                E("WAVE_PERIOD", 0.0, "f"),
                E("WAVE_PER_BOT", 0.0, "f"),
                E("WAVE_UB_BOT", 0.0, "f"),
            ],
            "NML_HEATING_CALCULATED": [
                E("HEATING_CALCULATE_ON", "F"),
                E("HEATING_CALCULATE_TYPE", "flux"),
                E("HEATING_CALCULATE_FILE", f"{c}_wnd.nc"),
                E("HEATING_CALCULATE_KIND", "variable"),
                E("HEATING_FRESHWATER", "F"),
                E("COARE_VERSION", "COARE26Z"),
                E("ZUU", 10.0, "f"),
                E("ZTT", 2.0, "f"),
                E("ZQQ", 2.0, "f"),
                E("AIR_TEMPERATURE", 0.0, "f"),
                E("RELATIVE_HUMIDITY", 0.0, "f"),
                E("SURFACE_PRESSURE", 0.0, "f"),
                E("LONGWAVE_RADIATION", 0.0, "f"),
                E("SHORTWAVE_RADIATION", 0.0, "f"),
                E("HEATING_LONGWAVE_PERCTAGE_IN_HEATFLUX", 0.78, "f"),
                E("HEATING_LONGWAVE_LENGTHSCALE_IN_HEATFLUX", 1.4, "f"),
                E("HEATING_SHORTWAVE_LENGTHSCALE_IN_HEATFLUX", 6.3, "f"),
            ],
            "NML_PHYSICS": [
                E("HORIZONTAL_MIXING_TYPE", "closure"),
                E("HORIZONTAL_MIXING_KIND", "constant"),
                E("HORIZONTAL_MIXING_COEFFICIENT", 0.1, "f"),
                E("HORIZONTAL_PRANDTL_NUMBER", 1.0, "f"),
                E("VERTICAL_MIXING_TYPE", "closure"),
                E("VERTICAL_MIXING_COEFFICIENT", 0.2, "f"),
                E("VERTICAL_PRANDTL_NUMBER", 1.0, "f"),
                E("BOTTOM_ROUGHNESS_MINIMUM", 0.0001, "f"),
                E("BOTTOM_ROUGHNESS_LENGTHSCALE", -1, "f"),
                E("BOTTOM_ROUGHNESS_KIND", "static"),
                E("BOTTOM_ROUGHNESS_TYPE", "orig"),
                E("BOTTOM_ROUGHNESS_FILE", f"{c}_roughness.nc"),
                E("CONVECTIVE_OVERTURNING", "F"),
                E("SCALAR_POSITIVITY_CONTROL", "T"),
                E("BAROTROPIC", "F"),
                E("BAROCLINIC_PRESSURE_GRADIENT", "sigma levels"),
                E("SEA_WATER_DENSITY_FUNCTION", "dens2"),
                E("RECALCULATE_RHO_MEAN", "F"),
                E("INTERVAL_RHO_MEAN", "days=1.0"),
                E("TEMPERATURE_ACTIVE", "F"),
                E("SALINITY_ACTIVE", "F"),
                E("SURFACE_WAVE_MIXING", "F"),
                E("WETTING_DRYING_ON", "T"),
                E("NOFLUX_BOT_CONDITION", "T"),
                E("ADCOR_ON", "T"),
                E("EQUATOR_BETA_PLANE", "F"),
                E("BACKWARD_ADVECTION", "F"),
                E("BACKWARD_STEP", 1, "d"),
            ],
            "NML_RIVER_TYPE": [
                E("RIVER_NUMBER", 0, "d"),
                E("RIVER_KIND", "variable"),
                E("RIVER_TS_SETTING", "calculated"),
                E("RIVER_INFLOW_LOCATION", "node"),
                E("RIVER_INFO_FILE", f"{c}_riv.nml"),
            ],
            "NML_OPEN_BOUNDARY_CONTROL": [
                E("OBC_ON", "F"),
                E("OBC_NODE_LIST_FILE", f"{c}_obc.dat"),
                E("OBC_ELEVATION_FORCING_ON", "F"),
                E("OBC_ELEVATION_FILE", f"{c}_elevtide.nc"),
                E("OBC_TS_TYPE", 3, "d"),
                E("OBC_TEMP_NUDGING", "F"),
                E("OBC_TEMP_FILE", f"{c}_tsobc.nc"),
                E("OBC_TEMP_NUDGING_TIMESCALE", 0.0001736111, ".10f"),
                E("OBC_SALT_NUDGING", "F"),
                E("OBC_SALT_FILE", f"{c}_tsobc.nc"),
                E("OBC_SALT_NUDGING_TIMESCALE", 0.0001736111, ".10f"),
                E("OBC_MEANFLOW", "F"),
                E("OBC_MEANFLOW_FILE", f"{c}_meanflow.nc"),
                E("OBC_TIDEOUT_INITIAL", 1, "d"),
                E("OBC_TIDEOUT_INTERVAL", 900, "d"),
                E("OBC_LONGSHORE_FLOW_ON", "F"),
                E("OBC_LONGSHORE_FLOW_FILE", f"{c}_lsf.dat"),
            ],
            "NML_GRID_COORDINATES": [
                E("GRID_FILE", f"{c}_grd.dat"),
                E("GRID_FILE_UNITS", "meters"),
                E("PROJECTION_REFERENCE", "proj=utm +ellps=WGS84 +zone=30"),
                E("SIGMA_LEVELS_FILE", f"{c}_sigma.dat"),
                E("DEPTH_FILE", f"{c}_dep.dat"),
                E("CORIOLIS_FILE", f"{c}_cor.dat"),
                E("SPONGE_FILE", f"{c}_spg.dat"),
            ],
            "NML_GROUNDWATER": [
                E("GROUNDWATER_ON", "F"),
                E("GROUNDWATER_TEMP_ON", "F"),
                E("GROUNDWATER_SALT_ON", "F"),
                E("GROUNDWATER_KIND", "none"),
                E("GROUNDWATER_FILE", f"{c}_groundwater.nc"),
                E("GROUNDWATER_FLOW", 0.0, "f"),
                E("GROUNDWATER_TEMP", 0.0, "f"),
                E("GROUNDWATER_SALT", 0.0, "f"),
            ],
            "NML_LAG": [
                E("LAG_PARTICLES_ON", "F"),
                E("LAG_START_FILE", f"{c}_lag_init.nc"),
                E("LAG_OUT_FILE", f"{c}_lag_out.nc"),
                E("LAG_FIRST_OUT", "cycle=0"),
                E("LAG_RESTART_FILE", f"{c}_lag_restart.nc"),
                E("LAG_OUT_INTERVAL", "cycle=30"),
                E("LAG_SCAL_CHOICE", "none"),
            ],
            "NML_ADDITIONAL_MODELS": [
                E("DATA_ASSIMILATION", "F"),
                E("DATA_ASSIMILATION_FILE", f"{c}_run.nml"),
                E("BIOLOGICAL_MODEL", "F"),
                E("STARTUP_BIO_TYPE", "observed"),
                E("SEDIMENT_MODEL", "F"),
                E("SEDIMENT_MODEL_FILE", "none"),
                E("SEDIMENT_PARAMETER_TYPE", "none"),
                E("SEDIMENT_PARAMETER_FILE", "none"),
                E("BEDFLAG_TYPE", "none"),
                E("BEDFLAG_FILE", "none"),
                E("ICING_MODEL", "F"),
                E("ICING_FORCING_FILE", "none"),
                E("ICING_FORCING_KIND", "none"),
                E("ICING_AIR_TEMP", 0.0, "f"),
                E("ICING_WSPD", 0.0, "f"),
                E("ICE_MODEL", "F"),
                E("ICE_FORCING_FILE", "none"),
                E("ICE_FORCING_KIND", "none"),
                E("ICE_SEA_LEVEL_PRESSURE", 0.0, "f"),
                E("ICE_AIR_TEMP", 0.0, "f"),
                E("ICE_SPEC_HUMIDITY", 0.0, "f"),
                E("ICE_SHORTWAVE", 0.0, "f"),
                E("ICE_CLOUD_COVER", 0.0, "f"),
            ],
            "NML_PROBES": [
                E("PROBES_ON", "F"),
                E("PROBES_NUMBER", 0, "d"),
                E("PROBES_FILE", f"{c}_probes.nml"),
            ],
            "NML_STATION_TIMESERIES": [
                E("OUT_STATION_TIMESERIES_ON", "F"),
                E("STATION_FILE", f"{c}_station.dat"),
                E("LOCATION_TYPE", "node"),
                E("OUT_ELEVATION", "F"),
                E("OUT_VELOCITY_3D", "F"),
                E("OUT_VELOCITY_2D", "F"),
                E("OUT_WIND_VELOCITY", "F"),
                E("OUT_SALT_TEMP", "F"),
                E("OUT_INTERVAL", "seconds= 360.0"),
            ],
            "NML_NESTING": [
                E("NESTING_ON", "F"),
                E("NESTING_BLOCKSIZE", 10, "d"),
                E("NESTING_TYPE", 1, "d"),
                E("NESTING_FILE_NAME", f"{c}_nest.nc"),
            ],
            "NML_NCNEST": [
                E("NCNEST_ON", "F"),
                E("NCNEST_BLOCKSIZE", 10, "d"),
                E("NCNEST_NODE_FILES", "", no_quote=True),
                E("NCNEST_OUT_INTERVAL", "seconds=900.0"),
            ],
            "NML_NCNEST_WAVE": [
                E("NCNEST_ON_WAVE", "F"),
                E("NCNEST_TYPE_WAVE", "spectral density"),
                E("NCNEST_BLOCKSIZE_WAVE", -1, "d"),
                E("NCNEST_NODE_FILES_WAVE", "none"),
            ],
            "NML_BOUNDSCHK": [
                E("BOUNDSCHK_ON", "F"),
                E("CHK_INTERVAL", 1, "d"),
                E("VELOC_MAG_MAX", 6.5, "f"),
                E("ZETA_MAG_MAX", 10.0, "f"),
                E("TEMP_MAX", 30.0, "f"),
                E("TEMP_MIN", -4.0, "f"),
                E("SALT_MAX", 40.0, "f"),
                E("SALT_MIN", -0.5, "f"),
            ],
            "NML_DYE_RELEASE": [
                E("DYE_ON", "F"),
                E("DYE_RELEASE_START", None),
                E("DYE_RELEASE_STOP", None),
                E("KSPE_DYE", 1, "d"),
                E("MSPE_DYE", 1, "d"),
                E("K_SPECIFY", 1, "d"),
                E("M_SPECIFY", 1, "d"),
                E("DYE_SOURCE_TERM", 1.0, "f"),
            ],
            "NML_PWP": [
                E("UPPER_DEPTH_LIMIT", 20.0, "f"),
                E("LOWER_DEPTH_LIMIT", 200.0, "f"),
                E("VERTICAL_RESOLUTION", 1.0, "f"),
                E("BULK_RICHARDSON", 0.65, "f"),
                E("GRADIENT_RICHARDSON", 0.25, "f"),
            ],
            "NML_SST_ASSIMILATION": [
                E("SST_ASSIM", "F"),
                E("SST_ASSIM_FILE", f"{c}_sst.nc"),
                E("SST_RADIUS", 0.0, "f"),
                E("SST_WEIGHT_MAX", 1.0, "f"),
                E("SST_TIMESCALE", 0.0, "f"),
                E("SST_TIME_WINDOW", 0.0, "f"),
                E("SST_N_PER_INTERVAL", 0.0, "f"),
            ],
            "NML_SSTGRD_ASSIMILATION": [
                E("SSTGRD_ASSIM", "F"),
                E("SSTGRD_ASSIM_FILE", f"{c}_sstgrd.nc"),
                E("SSTGRD_WEIGHT_MAX", 0.5, "f"),
                E("SSTGRD_TIMESCALE", 0.0001, "f"),
                E("SSTGRD_TIME_WINDOW", 1.0, "f"),
                E("SSTGRD_N_PER_INTERVAL", 24.0, "f"),
            ],
            "NML_SSHGRD_ASSIMILATION": [
                E("SSHGRD_ASSIM", "F"),
                E("SSHGRD_ASSIM_FILE", f"{c}_sshgrd.nc"),
                E("SSHGRD_WEIGHT_MAX", 0.0, "f"),
                E("SSHGRD_TIMESCALE", 0.0, "f"),
                E("SSHGRD_TIME_WINDOW", 0.0, "f"),
                E("SSHGRD_N_PER_INTERVAL", 0.0, "f"),
            ],
            "NML_TSGRD_ASSIMILATION": [
                E("TSGRD_ASSIM", "F"),
                E("TSGRD_ASSIM_FILE", f"{c}_tsgrd.nc"),
                E("TSGRD_WEIGHT_MAX", 0.0, "f"),
                E("TSGRD_TIMESCALE", 0.0, "f"),
                E("TSGRD_TIME_WINDOW", 0.0, "f"),
                E("TSGRD_N_PER_INTERVAL", 0.0, "f"),
            ],
            "NML_CUR_NGASSIMILATION": [
                E("CUR_NGASSIM", "F"),
                E("CUR_NGASSIM_FILE", f"{c}_cur.nc"),
                E("CUR_NG_RADIUS", 0.0, "f"),
                E("CUR_GAMA", 0.0, "f"),
                E("CUR_GALPHA", 0.0, "f"),
                E("CUR_NG_ASTIME_WINDOW", 0.0, "f"),
            ],
            "NML_CUR_OIASSIMILATION": [
                E("CUR_OIASSIM", "F"),
                E("CUR_OIASSIM_FILE", f"{c}_curoi.nc"),
                E("CUR_OI_RADIUS", 0.0, "f"),
                E("CUR_OIGALPHA", 0.0, "f"),
                E("CUR_OI_ASTIME_WINDOW", 0.0, "f"),
                E("CUR_N_INFLU", 0.0, "f"),
                E("CUR_NSTEP_OI", 0.0, "f"),
            ],
            "NML_TS_NGASSIMILATION": [
                E("TS_NGASSIM", "F"),
                E("TS_NGASSIM_FILE", f"{c}_ts.nc"),
                E("TS_NG_RADIUS", 0.0, "f"),
                E("TS_GAMA", 0.0, "f"),
                E("TS_GALPHA", 0.0, "f"),
                E("TS_NG_ASTIME_WINDOW", 0.0, "f"),
            ],
            "NML_TS_OIASSIMILATION": [
                E("TS_OIASSIM", "F"),
                E("TS_OIASSIM_FILE", f"{c}_tsoi.nc"),
                E("TS_OI_RADIUS", 0.0, "f"),
                E("TS_OIGALPHA", 0.0, "f"),
                E("TS_OI_ASTIME_WINDOW", 0.0, "f"),
                E("TS_MAX_LAYER", 0.0, "f"),
                E("TS_N_INFLU", 0.0, "f"),
                E("TS_NSTEP_OI", 0.0, "f"),
            ],
        }

    def _add_fabm_config(self) -> None:
        c = self._casename
        E = _NamelistEntry
        self._config["NML_NETCDF"].append(E("NC_FABM", "F"))
        self._config["NML_NETCDF_AV"].append(E("NCAV_FABM", "F"))
        self._config["NML_OPEN_BOUNDARY_CONTROL"] += [
            E("OBC_FABM_NUDGING", "F"),
            E("OBC_FABM_FILE", f"{c}_ERSEMobc.nc"),
            E("OBC_FABM_NUDGING_TIMESCALE", 0.0001736111, ".10f"),
        ]
        self._config["NML_NESTING"].append(E("FABM_NESTING_ON", "F"))
        self._config["NML_ADDITIONAL_MODELS"].append(E("FABM_MODEL", "F"))
        self._config["NML_FABM"] = [
            E("STARTUP_FABM_TYPE", "set values"),
            E("USE_FABM_BOTTOM_THICKNESS", "F"),
            E("USE_FABM_SALINITY", "F"),
            E("FABM_DEBUG", "F"),
            E("FABM_DIAG_OUT", "F"),
        ]

    def _index(self, section: str, entry: str) -> int:
        """Return the position of ``entry`` in ``section``'s entry list."""
        if section not in self._config:
            raise KeyError(f"{section!r} is not a valid namelist section.")
        names = [e.name for e in self._config[section]]
        try:
            return names.index(entry)
        except ValueError:
            raise KeyError(f"{entry!r} is not defined in section {section!r}.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def value(self, section: str, entry: str):
        """Return the current value of a namelist entry.

        Args:
            section: Section name (e.g. ``'NML_CASE'``). A leading ``'&'``
                is accepted and stripped.
            entry: Entry name within the section.

        Returns:
            The current value (``str``, ``int``, ``float``, or ``None``).

        Raises:
            KeyError: If ``section`` or ``entry`` does not exist.
        """
        section = section.lstrip("&")
        return self._config[section][self._index(section, entry)].value

    def update(
        self,
        section: str,
        entry: str,
        value=None,
        fmt: Optional[str] = None,
    ) -> None:
        """Update the value and/or format specifier of a namelist entry.

        Args:
            section: Section name (e.g. ``'NML_CASE'``). A leading ``'&'``
                is accepted and stripped.
            entry: Entry name within the section.
            value: New value. ``bool`` is converted to ``'T'``/``'F'``
                automatically. Pass ``None`` to leave unchanged.
            fmt: New Python format specification string (e.g. ``'f'``,
                ``'.3f'``, ``'d'``). Pass ``None`` to leave unchanged.

        Raises:
            KeyError: If ``section`` or ``entry`` does not exist.
            PyFVCOM2ValueError: If neither ``value`` nor ``fmt`` is supplied.
        """
        if value is None and fmt is None:
            raise PyFVCOM2ValueError("Supply at least one of 'value' or 'fmt'.")
        section = section.lstrip("&")
        idx = self._index(section, entry)
        if value is not None:
            self._config[section][idx].value = value
        if fmt is not None:
            self._config[section][idx].fmt = fmt

    def update_ramp(self, duration: float) -> None:
        """Set ``IRAMP`` to the number of external time steps for ``duration`` hours.

        Args:
            duration: Ramp duration in hours.
        """
        timestep = self.value("NML_INTEGRATION", "EXTSTEP_SECONDS")
        self.update("NML_INTEGRATION", "IRAMP", int(duration * 3600.0 / timestep))

    def update_nudging(self, recovery_time: float) -> None:
        """Set OBC nudging timescales from a physical recovery time.

        Computes the dimensionless FVCOM nudging timescale
        (``EXTSTEP_SECONDS / recovery_time_seconds``) and applies it to
        temperature, salinity, and — when FABM is enabled — FABM OBC nudging.

        Args:
            recovery_time: Recovery time in hours.
        """
        timestep = self.value("NML_INTEGRATION", "EXTSTEP_SECONDS")
        timescale = 1.0 / (recovery_time * 3600.0 / timestep)
        self.update("NML_OPEN_BOUNDARY_CONTROL", "OBC_TEMP_NUDGING_TIMESCALE", timescale)
        self.update("NML_OPEN_BOUNDARY_CONTROL", "OBC_SALT_NUDGING_TIMESCALE", timescale)
        if self._fabm:
            self.update("NML_OPEN_BOUNDARY_CONTROL", "OBC_FABM_NUDGING_TIMESCALE", timescale)

    def valid_nesting_timescale(self, interval: Optional[float] = None) -> bool:
        """Check whether a nesting output interval is compatible with the model.

        The simulation duration must be evenly divisible by
        ``NCNEST_OUT_INTERVAL × NCNEST_BLOCKSIZE``, and the quotient must be
        evenly divisible by ``EXTSTEP_SECONDS``.

        Args:
            interval: Candidate interval in seconds.  If ``None``, reads the
                current ``NCNEST_OUT_INTERVAL`` value from the config.

        Returns:
            ``True`` if the interval is compatible, ``False`` otherwise.
        """
        model_start = datetime.strptime(
            self.value("NML_CASE", "START_DATE"), "%Y-%m-%d %H:%M:%S"
        )
        model_end = datetime.strptime(
            self.value("NML_CASE", "END_DATE"), "%Y-%m-%d %H:%M:%S"
        )
        duration_seconds = (model_end - model_start).total_seconds()
        blocksize = self.value("NML_NCNEST", "NCNEST_BLOCKSIZE")
        timestep = self.value("NML_INTEGRATION", "EXTSTEP_SECONDS")

        if interval is None:
            raw = self.value("NML_NCNEST", "NCNEST_OUT_INTERVAL")
            units, raw_val = raw.split("=")
            interval = float(raw_val.strip())
            units = units.strip()
            if units == "minutes":
                interval *= 60.0
            elif units == "hours":
                interval *= 3600.0
            elif units == "days":
                interval *= 86400.0
            elif units == "cycles":
                interval = timestep * int(interval)

        res = duration_seconds / (interval * blocksize)
        return res % 2 == 0 and res / timestep % 2 == 0

    def update_nesting_interval(self, target_interval: int = 900) -> None:
        """Find and set a ``NCNEST_OUT_INTERVAL`` compatible with the model.

        Searches candidate intervals from 60 s up to ten times
        ``target_interval`` (in 60 s steps) and selects the valid candidate
        closest to ``target_interval``.

        Args:
            target_interval: Target interval in seconds. Defaults to 900 s.

        Raises:
            PyFVCOM2ValueError: If no suitable interval is found — try a
                different ``target_interval``, ``NCNEST_BLOCKSIZE``, or
                ``EXTSTEP_SECONDS``.
        """
        model_start = datetime.strptime(
            self.value("NML_CASE", "START_DATE"), "%Y-%m-%d %H:%M:%S"
        )
        model_end = datetime.strptime(
            self.value("NML_CASE", "END_DATE"), "%Y-%m-%d %H:%M:%S"
        )
        duration_seconds = (model_end - model_start).total_seconds()
        step = 60 if duration_seconds > 60 else 1

        candidates = [
            iv
            for iv in range(step, 10 * target_interval + step, step)
            if self.valid_nesting_timescale(iv)
        ]
        if not candidates:
            raise PyFVCOM2ValueError(
                "No suitable NCNEST_OUT_INTERVAL found. Try a different "
                "target_interval, NCNEST_BLOCKSIZE, or EXTSTEP_SECONDS."
            )

        best = candidates[
            int(np.argmin(np.abs(np.array(candidates, dtype=float) - target_interval)))
        ]
        self.update("NML_NCNEST", "NCNEST_OUT_INTERVAL", f"seconds={float(best):g}")

    def write(self, path: str) -> None:
        """Write the namelist configuration to an ASCII ``.nml`` file.

        Before writing, ``None``-valued ``*_FIRST_OUT`` entries and the dye
        release start/stop are back-filled from ``NML_CASE / START_DATE`` and
        ``NML_CASE / END_DATE``.

        Args:
            path: Output file path.

        Raises:
            PyFVCOM2ValueError: If ``START_DATE`` or ``END_DATE`` have not
                been set, if ``NCNEST_ON`` is ``'T'`` but the nesting interval
                is invalid, or if any other mandatory entry is still ``None``
                at write time.
        """
        case_start = self.value("NML_CASE", "START_DATE")
        case_end = self.value("NML_CASE", "END_DATE")

        if case_start is None:
            raise PyFVCOM2ValueError(
                "NML_CASE / START_DATE has not been set. "
                "Call update('NML_CASE', 'START_DATE', '<YYYY-MM-DD HH:MM:SS>') first."
            )
        if case_end is None:
            raise PyFVCOM2ValueError(
                "NML_CASE / END_DATE has not been set. "
                "Call update('NML_CASE', 'END_DATE', '<YYYY-MM-DD HH:MM:SS>') first."
            )

        fill_with_start = [
            ("NML_RESTART", "RST_FIRST_OUT"),
            ("NML_NETCDF", "NC_FIRST_OUT"),
            ("NML_NETCDF_SURFACE", "NCSF_FIRST_OUT"),
            ("NML_NETCDF_AV", "NCAV_FIRST_OUT"),
            ("NML_DYE_RELEASE", "DYE_RELEASE_START"),
        ]
        fill_with_end = [
            ("NML_DYE_RELEASE", "DYE_RELEASE_STOP"),
        ]
        for section, entry in fill_with_start:
            if self.value(section, entry) is None:
                self.update(section, entry, case_start)
        for section, entry in fill_with_end:
            if self.value(section, entry) is None:
                self.update(section, entry, case_end)

        if self.value("NML_NCNEST", "NCNEST_ON") == "T" and not self.valid_nesting_timescale():
            raise PyFVCOM2ValueError(
                "NCNEST_OUT_INTERVAL is not compatible with the model duration "
                "and blocksize. Call update_nesting_interval() first."
            )

        with Path(path).open("w") as f:
            for section, entries in self._config.items():
                f.write(f"&{section}\n")
                for i, entry in enumerate(entries):
                    if entry.value is None:
                        raise PyFVCOM2ValueError(
                            f"Mandatory entry {section} / {entry.name} has not been set."
                        )
                    f.write(entry.as_string())
                    f.write(",\n" if i < len(entries) - 1 else "\n")
                f.write("/\n\n")
