from typing import Optional

from .mesh_reader import read_mesh_file
from .sigma_reader import read_sigma_file
from .grid import Grid



def build_fvcom_grid(fvcom_mesh_filename: str, fvcom_obc_filename: str, sigma_filename: str,
                     coordinate_system: str, epsg_code: Optional[int]=None) -> Grid:
    """Build a Grid object from FVCOM mesh and sigma files.

    Args:
        fvcom_mesh_filename (str): Path to the FVCOM mesh file.
        fvcom_obc_filename (str): Path to the FVCOM OBC file.
        sigma_filename (str): Path to the sigma coordinate file.
        coordinate_system (str): Coordinate system for the grid ("cartesian" or "geographic").
        epsg_code (Optional[int]): EPSG code for the grid (if applicable).

    Returns:
        Grid: The constructed Grid object.
    """
    # Read mesh data
    mesh_data = read_mesh_file(fvcom_mesh_filename, "fvcom", obc_filename=fvcom_obc_filename)

    # Read sigma data
    sigma_data = read_sigma_file(sigma_filename)

    # Construct and return the Grid object
    grid = Grid(
        mesh_data=mesh_data,
        sigma_data=sigma_data,
        coordinate_system=coordinate_system,
        epsg_code=epsg_code
    )

    return grid
