import xarray as xr
import numpy as np
from typing import NamedTuple, Optional, List
from abc import ABC, abstractmethod
from netCDF4 import Dataset
from .exceptions import PyFVCOM2ValueError

__all__ = [
    "FVCOMHarmonicsNames", "TPXOHarmonicsNames", "TPXOComplexHarmonicsNames", 
    "HarmonicsData", "HarmonicsReader", "FVCOMHarmonicsReader", 
    "TPXOHarmonicsReader", "TPXOComplexHarmonicsReader"
]

# Named tuples to hold variable names for different harmonics data formats
FVCOMHarmonicsNames = NamedTuple(
    "FVCOMHarmonicsNames",
    [
        ("amplitude_var_name", str),
        ("phase_var_name", str),
        ("lon_var_name", str),
        ("lat_var_name", str),
        ("constituents_var_name", str),
    ],
)


TPXOHarmonicsNames = NamedTuple(
    "TPXOHarmonicsNames",
    [
        ("amplitude_var_name", str),
        ("phase_var_name", str),
        ("lon_var_name", str),
        ("lat_var_name", str),
        ("constituents_var_name", str),
    ],
)


TPXOComplexHarmonicsNames = NamedTuple(
    "TPXOComplexHarmonicsNames",
    [
        ("part1_var_name", str),
        ("part2_var_name", str),
        ("lon_var_name", str),
        ("lat_var_name", str),
        ("constituents_var_name", str),
    ],
)


# Named tuple to hold returned harmonics data
HarmonicsData = NamedTuple(
    "HarmonicsData",
    [
        ("longitude", np.ndarray),
        ("latitude", np.ndarray),
        ("amplitudes", np.ndarray),
        ("phases", np.ndarray),
        ("constituents", List[str]),
    ],
)


class HarmonicsReader(ABC):
    """Abstract base class for reading tidal harmonics data from netCDF files."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def read_harmonics(
        self, requested_constituents: List[str], var_names: NamedTuple
    ) -> HarmonicsData:
        """Read harmonics data for the specified constituents and variable.

        Args:
            requested_constituents: List of tidal constituent names to read.
            var_names: NamedTuple containing variable names for the specific data format (FVCOM, TPXO, etc.).
        Returns:
            HarmonicsData: NamedTuple containing longitude, latitude, amplitudes, phases, and constituents
        """
        pass

    def _check_constituents(
        self, requested_constituents: List[str], constituents_var_name
    ):
        """Check if the requested constituents are available in the file.

        Args:
            requested_constituents: List of tidal constituent names to read.
            constituents_var_name: Name of the variable in the netCDF file that contains constituent names.
        Raises:
            PyFVCOM2ValueError: If any requested constituent is not available in the file.
        """
        with Dataset(str(self.file_path), "r") as tides:
            available_constituents = [
                "".join(i).upper().strip()
                for i in tides.variables[constituents_var_name][:].astype(str)
            ]

        missing = [c for c in requested_constituents if c not in available_constituents]
        if missing:
            raise PyFVCOM2ValueError(
                f"The following requested constituents are not available in the file: {missing}"
            )


class FVCOMHarmonicsReader(HarmonicsReader):
    """A class to read FVCOM harmonics data from a netCDF file."""

    def __init__(self, file_path: str):
        super().__init__(file_path)

    def read_harmonics(
        self, requested_constituents: List[str], var_names: FVCOMHarmonicsNames
    ) -> HarmonicsData:
        """Read FVCOM harmonics data for the specified constituents and variable.

        Args:
            requested_constituents: List of tidal constituent names to read.
            var_names: NamedTuple containing variable names for FVCOM data.
        Returns:
            HarmonicsData: NamedTuple containing longitude, latitude, amplitudes, phases, and constituents
        """
        self._check_constituents(
            requested_constituents, var_names.constituents_var_name
        )

        with xr.open_dataset(str(self.file_path)) as tides:
            # Read available constituents from file
            constituent_names = [
                "".join(i).upper().strip()
                for i in tides[var_names.constituents_var_name].data[:].astype(str)
            ]

            # Determine the indices of the requested constituents
            const_indices = [constituent_names.index(i) for i in requested_constituents]

            # Read coordinate data
            lons = tides[var_names.lon_var_name].data[:]
            lats = tides[var_names.lat_var_name].data[:]

            # Read amplitude and phase data
            amplitudes = tides[var_names.amplitude_var_name].isel(nconsts=const_indices)
            phases = tides[var_names.phase_var_name].isel(nconsts=const_indices)

            # If necessary, reorder the array so that constiuents are the first dimension
            amplitudes = amplitudes.transpose("nconsts", ...)
            phases = phases.transpose("nconsts", ...)

        return HarmonicsData(
            longitude=lons,
            latitude=lats,
            amplitudes=amplitudes,
            phases=phases,
            constituents=requested_constituents,
        )


class TPXOHarmonicsReader(HarmonicsReader):
    """A class to read TPXO harmonics data stored as amplitude and phase from a netCDF file."""

    def __init__(self, file_path: str):
        super().__init__(file_path)

    def read_harmonics(
        self, requested_constituents: List[str], var_names: TPXOHarmonicsNames
    ) -> HarmonicsData:
        """Read TPXO harmonics data for the specified constituents.

        Args:
            requested_constituents: List of tidal constituent names to read.
            var_names: NamedTuple containing variable names for TPXO data.
        Returns:
            HarmonicsData: NamedTuple containing longitude, latitude, amplitudes, phases, and constituents
        """
        self._check_constituents(
            requested_constituents, var_names.constituents_var_name
        )

        with xr.open_dataset(str(self.file_path)) as tides:
            # Read available constituents from file
            constituent_names = [
                "".join(i).upper().strip()
                for i in tides[var_names.constituents_var_name].data[:].astype(str)
            ]

            # Determine the indices of the requested constituents
            const_indices = [constituent_names.index(i) for i in requested_constituents]

            # Read coordinate data
            lons = tides[var_names.lon_var_name].data[:]
            lats = tides[var_names.lat_var_name].data[:]

            # Read amplitude and phase data
            amplitudes = tides[var_names.amplitude_var_name].isel(nc=const_indices)
            phases = tides[var_names.phase_var_name].isel(nc=const_indices)

            # If necessary, reorder the array so that constiuents are the first dimension
            amplitudes = amplitudes.transpose("nc", ...)
            phases = phases.transpose("nc", ...)

        return HarmonicsData(
            longitude=lons,
            latitude=lats,
            amplitudes=amplitudes,
            phases=phases,
            constituents=requested_constituents,
        )


class TPXOComplexHarmonicsReader(HarmonicsReader):
    """A class to read TPXO harmonics data stored as complex (real/imaginary) from a netCDF file."""

    def __init__(self, file_path: str):
        super().__init__(file_path)

    def read_harmonics(
        self, requested_constituents: List[str], var_names: TPXOComplexHarmonicsNames
    ) -> HarmonicsData:
        """Read TPXO complex harmonics data for the specified constituents.

        Args:
            requested_constituents: List of tidal constituent names to read.
            var_names: NamedTuple containing variable names for TPXO data.
        Returns:
            HarmonicsData: NamedTuple containing longitude, latitude, amplitudes, phases, and constituents
        """
        self._check_constituents(
            requested_constituents, var_names.constituents_var_name
        )

        with xr.open_dataset(str(self.file_path)) as tides:
            # Read available constituents from file
            constituent_names = [
                "".join(i).upper().strip()
                for i in tides[var_names.constituents_var_name].data[:].astype(str)
            ]

            # Determine the indices of the requested constituents
            const_indices = [constituent_names.index(i) for i in requested_constituents]

            # Read coordinate data
            lons = tides[var_names.lon_var_name].data[:]
            lats = tides[var_names.lat_var_name].data[:]

            # Read amplitude and phase data
            real = tides[var_names.part1_var_name].isel(nc=const_indices)
            imag = tides[var_names.part2_var_name].isel(nc=const_indices)

            # If necessary, reorder the array so that constiuents are the first dimension
            real = real.transpose("nc", ...)
            imag = imag.transpose("nc", ...)

        # Convert complex to amplitude and phase
        amplitudes = np.abs(real + 1j * imag)
        phases = (np.arctan2(-imag, real) / np.pi) * 180

        return HarmonicsData(
            longitude=lons,
            latitude=lats,
            amplitudes=amplitudes,
            phases=phases,
            constituents=requested_constituents,
        )


def create_harmonics_reader(reader_type: str, file_path: str) -> HarmonicsReader:
    """Factory function to create the appropriate harmonics reader.

    Args:
        reader_type: Type of reader to create. Options:
            - 'fvcom': FVCOMHarmonicsReader
            - 'tpxo': TPXOHarmonicsReader (amplitude/phase format)
            - 'tpxo_complex': TPXOComplexHarmonicsReader (real/imaginary format)
        file_path: Path to the harmonics data file.

    Returns:
        Instance of the appropriate reader subclass.

    Raises:
        ValueError: If reader_type is not recognized.
    """
    if reader_type.lower() == "fvcom":
        return FVCOMHarmonicsReader(file_path)
    elif reader_type.lower() == "tpxo":
        return TPXOHarmonicsReader(file_path)
    elif reader_type.lower() == "tpxo_complex":
        return TPXOComplexHarmonicsReader(file_path)
    else:
        valid_types = ["fvcom", "tpxo", "tpxo_complex"]
        raise ValueError(
            f"Unknown reader_type '{reader_type}'. Valid options: {valid_types}"
        )


def get_fvcom_harmonics_names(var_name: str) -> FVCOMHarmonicsNames:
    """Get the standard variable names for FVCOM harmonics data

    Args:
        var_name: Base variable name for the harmonics data.
    Returns:
        FVCOMHarmonicsNames: NamedTuple containing standard variable names.
    Raises:
        PyFVCOM2ValueError: If var_name is not recognized.
    """
    if var_name == "zeta":
        amplitude_name, phase_name = "z_amp", "z_phase"
        lon_name, lat_name = "lon", "lat"
    elif var_name == "u":
        amplitude_name, phase_name = "u_amp", "u_phase"
        lon_name, lat_name = "lonc", "latc"
    elif var_name == "v":
        amplitude_name, phase_name = "v_amp", "v_phase"
        lon_name, lat_name = "lonc", "latc"
    elif var_name == "ua":
        amplitude_name, phase_name = "ua_amp", "ua_phase"
        lon_name, lat_name = "lonc", "latc"
    elif var_name == "va":
        amplitude_name, phase_name = "va_amp", "va_phase"
        lon_name, lat_name = "lonc", "latc"
    else:
        raise PyFVCOM2ValueError(f"Unknown FVCOM variable name '{var_name}'")

    return FVCOMHarmonicsNames(
        amplitude_var_name=amplitude_name,
        phase_var_name=phase_name,
        lon_var_name=lon_name,
        lat_var_name=lat_name,
        constituents_var_name="z_const_names",
    )


def get_tpxo_harmonics_names(var_name: str) -> TPXOHarmonicsNames:
    """Get the standard variable names for TPXO harmonics data stored as amplitude and phase.

    Args:
        var_name: Base variable name for the harmonics data.
    Returns:
        TPXOHarmonicsNames: NamedTuple containing standard variable names.
    Raises:
        PyFVCOM2ValueError: If var_name is not recognized.
    """
    if var_name == "zeta":
        amplitude_name, phase_name = "ha", "hp"
        lon_name, lat_name = "lon_z", "lat_z"
    elif var_name == "u":
        amplitude_name, phase_name = "ua", "up"
        lon_name, lat_name = "lon_u", "lat_u"
    elif var_name == "v":
        amplitude_name, phase_name = "va", "vp"
        lon_name, lat_name = "lon_v", "lat_v"
    else:
        raise PyFVCOM2ValueError(f"Unknown TPXO variable name '{var_name}'")

    return TPXOHarmonicsNames(
        amplitude_var_name=amplitude_name,
        phase_var_name=phase_name,
        lon_var_name=lon_name,
        lat_var_name=lat_name,
        constituents_var_name="con",
    )


def get_tpxo_complex_harmonics_names(var_name: str) -> TPXOComplexHarmonicsNames:
    """Get the standard variable names for TPXO harmonics data stored as complex (real/imaginary).

    Args:
        var_name: Base variable name for the harmonics data.
    Returns:
        TPXOComplexHarmonicsNames: NamedTuple containing standard variable names.
    Raises:
        PyFVCOM2ValueError: If var_name is not recognized.
    """
    if var_name == "zeta":
        part1_name, part2_name = "hRe", "hIm"
        lon_name, lat_name = "lon_z", "lat_z"
    elif var_name == "u":
        part1_name, part2_name = "uRe", "uIm"
        lon_name, lat_name = "lon_u", "lat_u"
    elif var_name == "v":
        part1_name, part2_name = "vRe", "vIm"
        lon_name, lat_name = "lon_v", "lat_v"
    else:
        raise PyFVCOM2ValueError(f"Unknown TPXO variable name '{var_name}'")

    return TPXOComplexHarmonicsNames(
        part1_var_name=part1_name,
        part2_var_name=part2_name,
        lon_var_name=lon_name,
        lat_var_name=lat_name,
        constituents_var_name="con",
    )
