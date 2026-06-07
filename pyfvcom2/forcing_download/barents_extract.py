"""
barents_extract.py
==================
Extract data from the MET Norway Barents 2.5 km EPS THREDDS server
(https://thredds.met.no/thredds/catalog/fou-hi/barents_eps_zdepth/catalog.html)
and save as a CF-compliant NetCDF file readable by xarray.

Requirements
------------
    pip install xarray netcdf4 numpy

Usage
-----
Command-line (single date):
    python barents_extract.py \
        --lon-min 15 --lon-max 35 \
        --lat-min 70 --lat-max 80 \
        --date 2024-03-10 \
        --variables temperature salinity \
        --output barents_subset.nc

Command-line (date range):
    python barents_extract.py \
        --lon-min 15 --lon-max 35 \
        --lat-min 70 --lat-max 80 \
        --date-start 2024-03-10 --date-end 2024-03-12 \
        --variables temperature salinity \
        --output barents_subset.nc

    # Retrieve all variables (omit --variables):
    python barents_extract.py \
        --lon-min 15 --lon-max 35 --lat-min 70 --lat-max 80 \
        --date-start 2024-03-10 --date-end 2024-03-12

    # List available variables without extracting:
    python barents_extract.py --list-variables

Python API:
    from barents_extract import extract_barents

    # Single date
    ds = extract_barents(
        lon_min=15, lon_max=35,
        lat_min=70, lat_max=80,
        date="2024-03-10",
        variables=["temperature", "salinity"],   # None = all variables
        output_path="barents_subset.nc",
    )

    # Date range
    ds = extract_barents(
        lon_min=15, lon_max=35,
        lat_min=70, lat_max=80,
        date_start="2024-03-10",
        date_end="2024-03-12",
        variables=["temperature", "salinity"],
        output_path="barents_range.nc",
    )

    # ds is an xarray.Dataset ready for further analysis
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Aggregated best-effort endpoint – covers the full rolling time window.
BARENTS_OPeNDAP_BE = "https://thredds.met.no/thredds/dodsC/fou-hi/barents_eps_zdepth_be"

# Individual-file base path (files named barents_eps_zdepth_YYYYMMDDTHHZ.nc).
BARENTS_OPeNDAP_BASE = (
    "https://thredds.met.no/thredds/dodsC/fou-hi/barents_eps_zdepth/"
)

# Dimension names as they appear in the dataset (lower-cased for matching).
_X_DIMS = {"x", "xi", "xi_rho", "xc"}
_Y_DIMS = {"y", "eta", "eta_rho", "yc"}
_T_DIMS = {"time", "t"}

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_barents(
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    date: "str | datetime | None" = None,
    date_start: "str | datetime | None" = None,
    date_end: "str | datetime | None" = None,
    variables: Optional[list[str]] = None,
    output_path: "str | Path" = "barents_subset.nc",
    opendap_url: str = BARENTS_OPeNDAP_BE,
    time_tolerance_hours: float = 3.0,
) -> xr.Dataset:
    """Extract a spatial/temporal subset from the Barents 2.5 km EPS dataset.

    Provide either *date* (single time step) or *date_start* + *date_end*
    (inclusive range of all available time steps between those two datetimes).

    Parameters
    ----------
    lon_min, lon_max : float
        Longitude bounds in decimal degrees East.
    lat_min, lat_max : float
        Latitude bounds in decimal degrees North.
    date : str or datetime, optional
        Single target date/time. Mutually exclusive with *date_start*/*date_end*.
    date_start : str or datetime, optional
        Start of the date range (inclusive). Requires *date_end*.
    date_end : str or datetime, optional
        End of the date range (inclusive). Requires *date_start*.
    variables : list[str] or None
        Names of data variables to extract.  ``None`` (default) retrieves all.
    output_path : str or Path
        Destination path for the output NetCDF file.
    opendap_url : str
        OPeNDAP endpoint to use.  Defaults to the aggregated best-effort
        endpoint (~10-day rolling window). For older data, point at a specific
        file: ``BARENTS_OPeNDAP_BASE + "barents_eps_zdepth_20240310T00Z.nc"``.
    time_tolerance_hours : float
        For single-date mode only: raise ``ValueError`` if the nearest time
        step is further than this many hours from *date*.

    Returns
    -------
    xr.Dataset
        The subsetted dataset (also saved to *output_path*).
    """
    # Validate date arguments
    if date is not None and (date_start is not None or date_end is not None):
        raise ValueError("Provide either 'date' or 'date_start'+'date_end', not both.")
    if date is None and date_start is None and date_end is None:
        raise ValueError("Provide either 'date' or both 'date_start' and 'date_end'.")
    if (date_start is None) != (date_end is None):
        raise ValueError("'date_start' and 'date_end' must be provided together.")

    output_path = Path(output_path)

    # ------------------------------------------------------------------
    # 1. Open remote dataset lazily via OPeNDAP
    # ------------------------------------------------------------------
    log.info("Connecting to %s …", opendap_url)
    try:
        ds_remote = xr.open_dataset(
            opendap_url,
            engine="netcdf4",
            mask_and_scale=True,
            decode_times=True,
        )
    except Exception as exc:
        raise ConnectionError(
            f"Could not open OPeNDAP endpoint: {opendap_url}\n"
            f"Original error: {exc}\n\n"
            "Troubleshooting tips:\n"
            "  • Confirm your internet connection.\n"
            "  • The best-effort endpoint has a rolling ~10-day window; for\n"
            "    older dates use an individual file URL:\n"
            f"    {BARENTS_OPeNDAP_BASE}barents_eps_zdepth_YYYYMMDDThhZ.nc\n"
            "  • Run with --list-variables to test connectivity first."
        ) from exc

    log.info("Connected. Dataset variables: %s", list(ds_remote.data_vars))

    # ------------------------------------------------------------------
    # 2. Resolve time indices
    # ------------------------------------------------------------------
    if date is not None:
        # Single time step
        target_dt = _parse_date(date)
        time_indices = [_nearest_time_index(ds_remote, target_dt, time_tolerance_hours)]
        log.info(
            "Single time step selected: index=%d  value=%s",
            time_indices[0], ds_remote.time.values[time_indices[0]],
        )
    else:
        # Date range: all steps between start and end (inclusive)
        start_dt = _parse_date(date_start)
        end_dt = _parse_date(date_end)
        if end_dt < start_dt:
            raise ValueError(
                f"date_end ({date_end}) is before date_start ({date_start})."
            )
        time_indices = _range_time_indices(ds_remote, start_dt, end_dt)
        t_vals = ds_remote.time.values
        log.info(
            "Date range: %d time steps selected  [%s → %s]",
            len(time_indices),
            str(t_vals[time_indices[0]])[:19],
            str(t_vals[time_indices[-1]])[:19],
        )

    # ------------------------------------------------------------------
    # 3. Resolve variable list
    # ------------------------------------------------------------------
    all_vars = set(ds_remote.data_vars)
    skip = _auxiliary_vars(ds_remote)

    if variables is None:
        selected_vars = sorted(all_vars - skip)
        log.info("Retrieving all %d data variables.", len(selected_vars))
    else:
        unknown = set(variables) - all_vars
        if unknown:
            raise ValueError(
                f"Requested variable(s) not in dataset: {sorted(unknown)}\n"
                f"Available: {sorted(all_vars - skip)}"
            )
        selected_vars = list(variables)
        log.info("Retrieving variables: %s", selected_vars)

    # ------------------------------------------------------------------
    # 4. Determine spatial (x/y) index slices from bounding box
    # ------------------------------------------------------------------
    lon2d, lat2d = _extract_lon_lat(ds_remote)
    x_slice, y_slice = _bbox_slices(lon2d, lat2d, lon_min, lon_max, lat_min, lat_max)
    log.info(
        "Spatial subset: x%s  y%s  (%d × %d points)",
        x_slice, y_slice,
        x_slice.stop - x_slice.start,
        y_slice.stop - y_slice.start,
    )

    # ------------------------------------------------------------------
    # 5. Download subset into memory
    # ------------------------------------------------------------------
    log.info("Downloading subset …")
    subset = _build_subset(ds_remote, selected_vars, time_indices, x_slice, y_slice)

    # ------------------------------------------------------------------
    # 6. Write NetCDF
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing %s …", output_path)
    subset.to_netcdf(output_path)
    log.info("Done – saved to %s", output_path)

    ds_remote.close()
    return subset


def list_variables(opendap_url: str = BARENTS_OPeNDAP_BE) -> list[dict]:
    """Return metadata for all data variables in the dataset.

    Each entry is a dict with keys: name, long_name, units, dims, shape.
    """
    ds = xr.open_dataset(opendap_url, engine="netcdf4", mask_and_scale=True)
    skip = _auxiliary_vars(ds)
    result = []
    for name in sorted(set(ds.data_vars) - skip):
        v = ds[name]
        result.append({
            "name": name,
            "long_name": v.attrs.get("long_name", ""),
            "units": v.attrs.get("units", ""),
            "dims": v.dims,
            "shape": v.shape,
        })
    ds.close()
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_date(date: "str | datetime") -> datetime:
    """Normalise *date* to a timezone-aware UTC datetime."""
    if isinstance(date, datetime):
        dt = date
    else:
        date_str = str(date)
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(
                f"Cannot parse date string: {date_str!r}. "
                "Expected ISO format, e.g. '2024-03-10' or '2024-03-10T06:00:00'."
            )
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _nearest_time_index(ds: xr.Dataset, target: datetime, tol_h: float) -> int:
    """Index of the time step closest to *target* (within *tol_h* hours)."""
    times = ds.time.values  # numpy datetime64 array
    target_np = np.datetime64(target.replace(tzinfo=None), "ns")
    deltas = np.abs(times - target_np)
    idx = int(np.argmin(deltas))
    delta_h = float(deltas[idx]) / 3_600_000_000_000  # nanoseconds → hours
    if delta_h > tol_h:
        first = str(times[0])[:19]
        last = str(times[-1])[:19]
        raise ValueError(
            f"Nearest time step is {delta_h:.1f} h from {target.isoformat()} "
            f"(tolerance: {tol_h} h).\n"
            f"Available range: {first} – {last} UTC.\n"
            "Use an individual-file URL for older data."
        )
    return idx


def _range_time_indices(ds: xr.Dataset, start: datetime, end: datetime) -> list[int]:
    """Return indices of all time steps in the inclusive [start, end] range."""
    times = ds.time.values
    start_np = np.datetime64(start.replace(tzinfo=None), "ns")
    end_np   = np.datetime64(end.replace(tzinfo=None),   "ns")
    indices = [int(i) for i, t in enumerate(times) if start_np <= t <= end_np]
    if not indices:
        first = str(times[0])[:19]
        last  = str(times[-1])[:19]
        raise ValueError(
            f"No time steps found between {start.isoformat()} and {end.isoformat()}.\n"
            f"Available range: {first} – {last} UTC.\n"
            "The best-effort endpoint covers only the last ~10 days. "
            "For older data use individual file URLs."
        )
    return indices


def _extract_lon_lat(ds: xr.Dataset):
    """Return 2-D (y, x) longitude and latitude arrays."""
    for lon_name in ("longitude", "lon", "LONGITUDE", "LON"):
        if lon_name in ds or lon_name in ds.coords:
            lon2d = (ds[lon_name] if lon_name in ds else ds.coords[lon_name]).values
            break
    else:
        raise KeyError(
            "Longitude variable not found. Available names: "
            + str(list(ds.coords) + list(ds.data_vars))
        )

    for lat_name in ("latitude", "lat", "LATITUDE", "LAT"):
        if lat_name in ds or lat_name in ds.coords:
            lat2d = (ds[lat_name] if lat_name in ds else ds.coords[lat_name]).values
            break
    else:
        raise KeyError("Latitude variable not found.")

    # Broadcast 1-D arrays if needed (rectangular grids)
    if lon2d.ndim == 1 and lat2d.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon2d, lat2d)

    return lon2d, lat2d


def _bbox_slices(
    lon2d, lat2d,
    lon_min, lon_max,
    lat_min, lat_max,
) -> tuple:
    """Return (x_slice, y_slice) bounding the grid points inside the bbox."""
    mask = (
        (lon2d >= lon_min) & (lon2d <= lon_max) &
        (lat2d >= lat_min) & (lat2d <= lat_max)
    )
    y_idx, x_idx = np.where(mask)

    if x_idx.size == 0:
        raise ValueError(
            f"Bounding box lon=[{lon_min}, {lon_max}] lat=[{lat_min}, {lat_max}] "
            "contains no grid points. "
            "Verify the box overlaps the Barents Sea domain "
            "(roughly lon 10–80°E, lat 68–83°N)."
        )

    return (
        slice(int(x_idx.min()), int(x_idx.max()) + 1),
        slice(int(y_idx.min()), int(y_idx.max()) + 1),
    )


def _auxiliary_vars(ds: xr.Dataset) -> set:
    """Return variable names that are not primary data fields."""
    skip = {"longitude", "latitude", "lon", "lat", "time", "depth",
            "ensemble_member", "forecast_reference_time", "projection_lcc",
            "projection_stere", "angle"}

    for var in ds.data_vars.values():
        gm = var.attrs.get("grid_mapping")
        if gm:
            skip.add(gm)

    spatial = _X_DIMS | _Y_DIMS
    for name, var in ds.data_vars.items():
        var_dims_lower = {d.lower() for d in var.dims}
        if not var_dims_lower.intersection(spatial):
            skip.add(name)

    return skip


def _build_subset(
    ds: xr.Dataset,
    variables: list[str],
    time_indices: list[int],
    x_slice: slice,
    y_slice: slice,
) -> xr.Dataset:
    """Download and assemble the requested subset.

    *time_indices* is a list of integer positions along the time dimension.
    A single-element list gives a dataset with one time step (time dim preserved);
    a multi-element list gives a dataset spanning those steps.
    """
    # Use a slice when indices are contiguous to minimise OPeNDAP requests.
    if _is_contiguous(time_indices):
        time_sel: "slice | list[int]" = slice(time_indices[0], time_indices[-1] + 1)
    else:
        time_sel = time_indices

    def _make_indexer(dims: tuple) -> dict:
        idx: dict = {}
        for dim in dims:
            dl = dim.lower()
            if dl in _T_DIMS:
                idx[dim] = time_sel
            elif dl in _X_DIMS:
                idx[dim] = x_slice
            elif dl in _Y_DIMS:
                idx[dim] = y_slice
        return idx

    # Download data variables
    data_arrays: dict[str, xr.DataArray] = {}
    for vname in variables:
        log.info("  ↓ %s", vname)
        var = ds[vname]
        indexer = _make_indexer(var.dims)
        data_arrays[vname] = var.isel(indexer).load()

    # Download matching coordinates
    coords: dict[str, xr.DataArray] = {}
    for cname, coord in ds.coords.items():
        indexer = _make_indexer(coord.dims)
        try:
            coords[cname] = coord.isel(indexer).load() if indexer else coord.load()
        except Exception:
            pass

    subset = xr.Dataset(data_arrays, coords=coords, attrs=dict(ds.attrs))
    subset.attrs.update({
        "source_url": opendap_url if (opendap_url := BARENTS_OPeNDAP_BE) else "",
        "extraction_time_utc": datetime.now(timezone.utc).isoformat(),
        "bbox_lon": f"[{x_slice.start}, {x_slice.stop}]",
        "bbox_lat": f"[{y_slice.start}, {y_slice.stop}]",
        "time_steps_extracted": len(time_indices),
    })
    return subset


def _is_contiguous(indices: list[int]) -> bool:
    """Return True if *indices* form a consecutive integer sequence."""
    return len(indices) > 1 and all(
        indices[i] + 1 == indices[i + 1] for i in range(len(indices) - 1)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract data from MET Norway Barents 2.5 km EPS via OPeNDAP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Spatial bounds
    p.add_argument("--lon-min", type=float, help="West longitude bound (°E)")
    p.add_argument("--lon-max", type=float, help="East longitude bound (°E)")
    p.add_argument("--lat-min", type=float, help="South latitude bound (°N)")
    p.add_argument("--lat-max", type=float, help="North latitude bound (°N)")

    # Time selection (mutually exclusive groups)
    time_group = p.add_mutually_exclusive_group()
    time_group.add_argument(
        "--date", metavar="DATE",
        help="Single target date/time, e.g. 2024-03-10 or 2024-03-10T06:00:00.",
    )
    p.add_argument(
        "--date-start", metavar="DATE",
        help="Start of date range (inclusive), e.g. 2024-03-10. Use with --date-end.",
    )
    p.add_argument(
        "--date-end", metavar="DATE",
        help="End of date range (inclusive), e.g. 2024-03-12. Use with --date-start.",
    )

    # Variables / output
    p.add_argument(
        "--variables", nargs="+", default=None, metavar="VAR",
        help="Variable names to extract. Omit to retrieve all variables.",
    )
    p.add_argument(
        "--output", default="barents_subset.nc", metavar="PATH",
        help="Output NetCDF file path.",
    )
    p.add_argument(
        "--url", default=BARENTS_OPeNDAP_BE, metavar="URL",
        help="Override the OPeNDAP endpoint URL.",
    )
    p.add_argument(
        "--time-tolerance", type=float, default=3.0, metavar="HOURS",
        help="(Single-date mode) Max allowed gap (hours) between requested date and nearest step.",
    )
    p.add_argument(
        "--list-variables", action="store_true",
        help="Print available variables and exit (no extraction performed).",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return p


def main(argv=None):
    args = _cli().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_variables:
        log.info("Fetching variable list from %s …", args.url)
        variables = list_variables(args.url)
        print(f"\n{'Name':<30s}  {'Units':<12s}  {'Long name'}")
        print("-" * 72)
        for v in variables:
            print(f"{v['name']:<30s}  {v['units']:<12s}  {v['long_name']}")
        print(f"\n{len(variables)} data variables available.")
        return

    # Validate spatial args
    required_spatial = ["lon_min", "lon_max", "lat_min", "lat_max"]
    missing = [r for r in required_spatial if getattr(args, r) is None]
    if missing:
        _cli().error(
            "Missing spatial arguments: "
            + ", ".join(f"--{m.replace('_', '-')}" for m in missing)
        )

    # Validate time args
    has_date = args.date is not None
    has_range = args.date_start is not None or args.date_end is not None
    if not has_date and not has_range:
        _cli().error("Provide --date (single step) or --date-start and --date-end (range).")
    if has_range and (args.date_start is None or args.date_end is None):
        _cli().error("--date-start and --date-end must be used together.")

    extract_barents(
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        date=args.date,
        date_start=args.date_start,
        date_end=args.date_end,
        variables=args.variables,
        output_path=args.output,
        opendap_url=args.url,
        time_tolerance_hours=args.time_tolerance,
    )


if __name__ == "__main__":
    main()
