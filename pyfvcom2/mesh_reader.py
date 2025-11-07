import numpy as np
from typing import List, Optional, NamedTuple

from .exceptions import PyFVCOM2ValueError, PyFVCOM2FileNotFoundError


# Define the public API of this module
__all__ = [
    "MeshData",
    "read_mesh",
    "read_sms_mesh", 
    "read_fvcom_mesh",
    "read_smesh_mesh",
    "read_mike_mesh",
    "read_gmsh_mesh",
    "read_fvcom_obc",
    "parse_obc_sections",
]


class MeshData(NamedTuple):
    """
    Named tuple containing standardized mesh data from all readers
    
    Attrs:
        triangles: Array of triangle vertex indices.
        nodes: Integer identifier for each node.
        x1: Array of node x1-coordinates (x or longitude).
        x2: Array of node x2-coordinates (y or latitude).
        x3: Array of node x3-coordinates (if available).
        types_bdy: Array of boundary types for nodes (if available).
        nodes_bdy: List of lists of open boundary node indices (if available).
    """
    triangle: np.ndarray
    nodes: Optional[np.ndarray]
    x1: np.ndarray
    x2: np.ndarray
    x3: Optional[np.ndarray]
    types_bdy: Optional[np.ndarray]
    nodes_bdy: Optional[List[List[int]]]


def _safe_open(filename: str, mode: str = "r"):
    """
    Safely open a file with PyFVCOM2 exception handling.

    Args:
        filename (str): Path to the file to open.
        mode (str): File opening mode (default 'r').

    Returns:
        File handle.

    Raises:
        PyFVCOM2FileNotFoundError: If the file cannot be found or opened.
    """
    try:
        return open(filename, mode)
    except FileNotFoundError as e:
        raise PyFVCOM2FileNotFoundError(f"Could not find mesh file: {filename}") from e
    except (IOError, OSError) as e:
        raise PyFVCOM2FileNotFoundError(f"Could not open mesh file: {filename}") from e


def read_mesh(filename: str, mesh_type: str, **kwargs) -> MeshData:
    """
    Factory method to read mesh files of various formats.

    Args:
        filename (str): Path to the mesh file.
        mesh_type (str): Type of mesh file. Supported types are:
            - 'sms': SMS unstructured grid format (.2dm)
            - 'fvcom': FVCOM unstructured grid format (.dat)
            - 'smesh': smesh output format
            - 'mike': DHI MIKE21 unstructured grid format (.mesh)
            - 'gmsh': GMSH unstructured grid format (.gmsh)
        **kwargs: Additional keyword arguments passed to the specific reader function.
            For 'fvcom': obc_filename (str) - path to FVCOM OBC file.
            For 'sms': nodestrings (bool) - whether to return nodestring information
            For 'mike': flipZ (bool) - whether to flip Z coordinates (default True)

    Returns:
        MeshData: Named tuple containing standardized mesh data.

    Raises:
        PyFVCOM2ValueError: If mesh_type is not supported.
        PyFVCOM2FileNotFoundError: If the mesh file cannot be found or opened.
    """
    mesh_type = mesh_type.lower()

    if mesh_type == "sms":
        return read_sms_mesh(filename, **kwargs)
    elif mesh_type == "fvcom":
        return read_fvcom_mesh(filename, **kwargs)
    elif mesh_type == "smesh":
        return read_smesh_mesh(filename, **kwargs)
    elif mesh_type == "mike":
        return read_mike_mesh(filename, **kwargs)
    elif mesh_type == "gmsh":
        return read_gmsh_mesh(filename, **kwargs)
    else:
        supported_types = ["sms", "fvcom", "smesh", "mike", "gmsh"]
        raise PyFVCOM2ValueError(
            f"Unsupported mesh_type '{mesh_type}'. Supported types are: {supported_types}"
        )


def read_sms_mesh(mesh: str, nodestrings: Optional[bool] = False) -> MeshData:
    """
    Reads in the SMS unstructured grid format. Also creates IDs for output to
    MIKE unstructured grid format.

    Args:
        mesh (str): Full path to an SMS unstructured grid (.2dm) file.
        nodestrings (bool, optional): Set to True to return the IDs of the node strings. Defaults to False.

    Returns:
        MeshData: Named tuple containing:
            - triangle (np.ndarray): Integer array of shape (nele, 3). Each triangle is composed of
              three points and this contains the three node numbers (stored in
              nodes) which refer to the coordinates in `x' and `y' (see below). Values
              are python-indexed.
            - nodes (np.ndarray): Integer number assigned to each node.
            - X (np.ndarray): X coordinates of each grid node.
            - Y (np.ndarray): Y coordinates of each grid node.
            - Z (np.ndarray): Z coordinates and any associated Z value.
            - types (np.ndarray): Classification for each node string based on the number of node
              strings + 2. This is mainly for use if converting from SMS .2dm
              grid format to DHI MIKE21 .mesh format since the latter requires
              unique IDs for each boundary (with 0 and 1 reserved for land and
              sea nodes).
            - nodestrings (Optional[List]): List of lists containing the node IDs (python-indexed) of the
              node strings in the SMS grid if nodestrings=True, otherwise None.

    """

    fileRead = _safe_open(mesh, "r")
    lines = fileRead.readlines()
    fileRead.close()

    triangles = []
    nodes = []
    types = []
    nodeStrings = []
    nstring = []
    x = []
    y = []
    z = []

    # MIKE unstructured grids allocate their boundaries with a type ID flag.
    # Although this function is not necessarily always the precursor to writing
    # a MIKE unstructured grid, we can create IDs based on the number of node
    # strings in the SMS grid. MIKE starts counting open boundaries from 2 (1
    # and 0 are land and sea nodes, respectively).
    typeCount = 2

    for line in lines:
        line = line.strip()
        if line.startswith("E3T"):
            ttt = line.split()
            t1 = int(ttt[2]) - 1
            t2 = int(ttt[3]) - 1
            t3 = int(ttt[4]) - 1
            triangles.append([t1, t2, t3])
        elif line.startswith("ND "):
            xy = line.split()
            x.append(float(xy[2]))
            y.append(float(xy[3]))
            z.append(float(xy[4]))
            nodes.append(int(xy[1]))
            # Although MIKE keeps zero and one reserved for normal nodes and
            # land nodes, SMS doesn't. This means it's not straightforward
            # to determine this information from the SMS file alone. It would
            # require finding nodes which are edge nodes and assigning their
            # ID to one. All other nodes would be zero until they were
            # overwritten when examining the node strings below.
            types.append(0)
        elif line.startswith("NS "):
            allTypes = line.split(" ")

            for nodeID in allTypes[2:]:
                types[np.abs(int(nodeID)) - 1] = typeCount
                if int(nodeID) > 0:
                    nstring.append(int(nodeID) - 1)
                else:
                    nstring.append(np.abs(int(nodeID)) - 1)
                    nodeStrings.append(nstring)
                    nstring = []

                # Count the number of node strings, and output that to types.
                # Nodes in the node strings are stored in nodeStrings.
                if int(nodeID) < 0:
                    typeCount += 1

    # Convert to numpy arrays.
    triangle = np.asarray(triangles)
    nodes = np.asarray(nodes)
    types = np.asarray(types)
    X = np.asarray(x)
    Y = np.asarray(y)
    Z = np.asarray(z)

    if nodestrings:
        return MeshData(triangle, nodes, X, Y, Z, types, nodeStrings)
    else:
        return MeshData(triangle, nodes, X, Y, Z, types, None)


def read_fvcom_mesh(mesh: str, obc_filename: Optional[str] = None) -> MeshData:
    """
    Reads in the FVCOM unstructured grid format.

    Args:
        mesh (str): Full path to the FVCOM unstructured grid file (.dat usually). The file
            name should also contain '_grd' in the name e.g. 'my_file_grd.dat'.
            The file contains information about the triangle indicies and the
            x, y, z coodinates of the grid nodes.
            The file header should be two lines:
                Node Number = nnn
                Cell Number = eee
            Followed by 'element_index, tri1, tri2, tri3' for eee cells
            Followed by 'node_index, x, y, z' for nnn nodes
        obc_filename (str, optional): Full path to the FVCOM OBC file. This is
            used to read in the open boundary node strings if provided.

    Returns:
        MeshData: Named tuple containing:
            - triangle (np.ndarray): Integer array of shape (ntri, 3). Each triangle is composed of
              three points and this contains the three node numbers (stored in
              nodes) which refer to the coordinates in `x' and `y' (see below).
            - nodes (np.ndarray): Integer number assigned to each node.
            - X (np.ndarray): X coordinates of each grid node.
            - Y (np.ndarray): Y coordinates of each grid node.
            - Z (np.ndarray): Z coordinates and any associated Z value.
            - types (Optional[np.ndarray]): None for FVCOM format (no type information available).
            - nodestrings (Optional[List]): None for FVCOM format (no nodestring information available).

    """

    fileRead = _safe_open(mesh, "r")

    # Read the file and header (two lines)
    lines = fileRead.readlines()
    fileRead.close()

    header = lines[:2]  # two lines of header
    lines = lines[2:]

    node_number = 0
    cell_number = 0

    triangles = []
    nodes = []
    x = []
    y = []
    z = []

    if "Node" in header[0]:
        node_number = int(header[0].split("=")[1])
        cell_number = int(header[1].split("=")[1])
    elif "Cell" in header[0]:
        node_number = int(header[1].split("=")[1])
        cell_number = int(header[0].split("=")[1])

    if (node_number == 0) | (cell_number == 0):
        # if header info not present get grid using column numbers
        for line in lines:
            ttt = line.strip().split()
            if len(ttt) == 5:
                t1 = int(ttt[1]) - 1
                t2 = int(ttt[2]) - 1
                t3 = int(ttt[3]) - 1
                triangles.append([t1, t2, t3])
            elif len(ttt) == 4:
                x.append(float(ttt[1]))
                y.append(float(ttt[2]))
                z.append(float(ttt[3]))
                nodes.append(int(ttt[0]))

    else:
        # Check if triangles or xyz comes first
        tri_first = node_number == int(lines[-1].strip().split()[0])

        # Get index ranges of the two sections
        if tri_first:
            tri_st = 0
            tri_en = cell_number
            xyz_st = cell_number
            xyz_en = cell_number + node_number
        else:
            xyz_st = 0
            xyz_en = node_number
            tri_st = node_number
            tri_en = node_number + cell_number

        # Extract data from the two sections
        for l in range(tri_st, tri_en):
            ttt = lines[l].strip().split()
            t1 = int(ttt[1]) - 1
            t2 = int(ttt[2]) - 1
            t3 = int(ttt[3]) - 1
            triangles.append([t1, t2, t3])

        for l in range(xyz_st, xyz_en):
            ttt = lines[l].strip().split()
            x.append(float(ttt[1]))
            y.append(float(ttt[2]))
            z.append(float(ttt[3]))
            nodes.append(int(ttt[0]))

    # Convert to numpy arrays.
    triangle = np.asarray(triangles)
    nodes = np.asarray(nodes)
    X = np.asarray(x)
    Y = np.asarray(y)
    Z = np.asarray(z)

    # Read FVCOM open boundary file if provided
    if obc_filename is not None:
        obc_node_array, bdy_types, count = read_fvcom_obc(obc_filename)
        open_bdy_node_lists = parse_obc_sections(obc_node_array, triangle)
    else:
        open_bdy_node_lists = None
        bdy_types = None

    return MeshData(triangle, nodes, X, Y, Z, bdy_types, open_bdy_node_lists)


def read_smesh_mesh(mesh: str) -> MeshData:
    """
    Reads output of the smeshing tool. This is (close) to the fort.14 file format.

    Args:
        mesh (str): Full path to the smesh output file.

    Returns:
        MeshData: Named tuple containing:
            - triangle (np.ndarray): Integer array of shape (ntri, 3). Each triangle is composed of three points and this contains the three node
              numbers which refer to the coordinates in `x' and `y' (see below).
            - nodes (Optional[np.ndarray]): None for smesh format (no node information available).
            - X (np.ndarray): X coordinates of each grid node.
            - Y (np.ndarray): Y coordinates of each grid node.
            - Z (Optional[np.ndarray]): None for smesh format (no Z information available).
            - types (Optional[np.ndarray]): None for smesh format (no type information available).
            - nodestrings (Optional[List]): None for smesh format (no nodestring information available).

    """

    with _safe_open(mesh, "r") as file_read:
        # Skip the file header line
        lines = file_read.readlines()[1:]

    triangles = []
    x = []
    y = []

    for line in lines:
        ttt = line.strip().split()
        if len(ttt) == 3:
            t1 = int(ttt[0])
            t2 = int(ttt[1])
            t3 = int(ttt[2])
            triangles.append([t1, t2, t3])
        elif len(ttt) == 2:
            x.append(float(ttt[0]))
            y.append(float(ttt[1]))

    # Convert to numpy arrays.
    triangle = np.asarray(triangles)
    X = np.asarray(x)
    Y = np.asarray(y)

    return MeshData(triangle, None, X, Y, None, None, None)


def read_mike_mesh(mesh: str, flipZ: bool = True) -> MeshData:
    """
    Reads in the MIKE unstructured grid format.

    Depth sign is typically reversed (i.e. z*-1) but can be disabled by
    passing flipZ=False.

    Args:
        mesh (str): Full path to the DHI MIKE21 unstructured grid file (.mesh usually).
        flipZ (bool, optional): DHI MIKE21 unstructured grids store the z value as positive down
            whereas FVCOM wants negative down. The conversion is
            automatically applied unless flipZ is set to False. Defaults to True.

    Returns:
        MeshData: Named tuple containing:
            - triangle (np.ndarray): Integer array of shape (ntri, 3). Each triangle is composed of
              three points and this contains the three node numbers (stored in
              nodes) which refer to the coordinates in `x' and `y' (see below). Given as
              a zero-indexed array.
            - nodes (np.ndarray): Integer number assigned to each node.
            - X (np.ndarray): X coordinates of each grid node.
            - Y (np.ndarray): Y coordinates of each grid node.
            - Z (np.ndarray): Z coordinates and any associated Z value.
            - types (np.ndarray): Classification for each open boundary. DHI MIKE21 .mesh format
              requires unique IDs for each open boundary (with 0 and 1
              reserved for land and sea nodes).
            - nodestrings (Optional[List]): None for MIKE format (no nodestring information available).

    """

    fileRead = _safe_open(mesh, "r")
    # Skip the file header (one line)
    lines = fileRead.readlines()[1:]
    fileRead.close()

    triangles = []
    nodes = []
    types = []
    x = []
    y = []
    z = []

    for line in lines:
        ttt = line.strip().split()
        if len(ttt) == 4:
            t1 = int(ttt[1]) - 1
            t2 = int(ttt[2]) - 1
            t3 = int(ttt[3]) - 1
            triangles.append([t1, t2, t3])
        elif len(ttt) == 5:
            x.append(float(ttt[1]))
            y.append(float(ttt[2]))
            z.append(float(ttt[3]))
            types.append(int(ttt[4]))
            nodes.append(int(ttt[0]))

    # Convert to numpy arrays.
    triangle = np.asarray(triangles)
    nodes = np.asarray(nodes)
    types = np.asarray(types)
    X = np.asarray(x)
    Y = np.asarray(y)
    # N.B. Depths should be negative for FVCOM
    if flipZ:
        Z = -np.asarray(z)
    else:
        Z = np.asarray(z)

    return MeshData(triangle, nodes, X, Y, Z, types, None)


def read_gmsh_mesh(mesh: str) -> MeshData:
    """
    Reads in the GMSH unstructured grid format (version 2.2).

    Args:
        mesh (str): Full path to the GMSH unstructured grid file (.gmsh usually).

    Returns:
        MeshData: Named tuple containing:
            - triangle (np.ndarray): Integer array of shape (ntri, 3). Each triangle is composed of three
              points and this contains the three node numbers (stored in nodes) which
              refer to the coordinates in `x' and `y' (see below).
            - nodes (np.ndarray): Integer number assigned to each node.
            - X (np.ndarray): X coordinates of each grid node.
            - Y (np.ndarray): Y coordinates of each grid node.
            - Z (np.ndarray): Z coordinates and any associated Z value.
            - types (Optional[np.ndarray]): None for GMSH format (no type information available).
            - nodestrings (Optional[List]): None for GMSH format (no nodestring information available).

    """

    fileRead = _safe_open(mesh, "r")
    lines = fileRead.readlines()
    fileRead.close()

    _header = False
    _nodes = False
    _elements = False

    # Counters for the nodes and elements.
    n = 0
    e = 0

    for line in lines:
        line = line.strip()

        # If we've been told we've got to the header, read in the mesh version
        # here.
        if _header:
            _header = False
            continue

        # Grab the number of nodes.
        if _nodes:
            nn = int(line.strip())
            x, y, z, nodes = (
                np.zeros(nn) - 1,
                np.zeros(nn) - 1,
                np.zeros(nn) - 1,
                np.zeros(nn).astype(int) - 1,
            )
            _nodes = False
            continue

        # Grab the number of elements.
        if _elements:
            ne = int(line.strip())
            triangles = np.zeros((ne, 3)).astype(int) - 1
            _elements = False
            continue

        if line == r"$MeshFormat":
            # Header information on the next line
            _header = True
            continue

        elif line == r"$EndMeshFormat":
            continue

        elif line == r"$Nodes":
            _nodes = True
            continue

        elif line == r"$EndNodes":
            continue

        elif line == r"$Elements":
            _elements = True
            continue

        elif line == r"$EndElements":
            continue

        else:
            # Some non-info line, so either nodes or elements. Discern that
            # based on the number of fields.
            s = line.split(" ")
            if len(s) == 4:
                # Nodes
                nodes[n] = int(s[0])
                x[n] = float(s[1])
                y[n] = float(s[2])
                z[n] = float(s[3])
                n += 1

            # Only keep the triangulation for the 2D mesh (ditch the 1D stuff).
            elif len(s) > 4 and int(s[1]) == 2:
                # Offset indices by one for Python indexing.
                triangles[e, :] = [int(i) - 1 for i in s[-3:]]
                e += 1

            else:
                continue

    # Tidy up the triangles array to remove the empty rows due to the number
    # of elements specified in the mesh file including the 1D triangulation.
    # triangles = triangles[triangles[:, 0] != -1, :]
    triangles = triangles[:e, :]

    return MeshData(triangles, nodes, x, y, z, None, None)


def read_fvcom_obc(obc):
    """
    Read in an FVCOM open boundary file.

    Args:
    obc : str
        Path to the casename_obc.dat file from FVCOM.

    Returns:
    nodes : np.ndarray
        Node IDs (zero-indexed) for the open boundary.
    types : np.ndarray
        Open boundary node types (see the FVCOM manual for more information on
        what these values mean).
    count : np.ndarray
        Open boundary node number.

    """

    obcs = np.genfromtxt(obc, skip_header=1).astype(int)
    count = obcs[:, 0]
    nodes = obcs[:, 1] - 1 # -1 for zero indexing
    types = obcs[:, 2]

    return nodes, types, count


def parse_obc_sections(obc_node_array, triangle):
    """
    Separates the open boundary nodes of a mesh into the separate contiguous open boundary segments

    Args:
    obc_node_array : array
        Array of the nodes which are open boundary nodes, as nodes returned by read_fvcom_obc
    triangle : 3xn array
        Triangulation array of nodes, as triangle returned by read_fvcom_mesh

    Returns:
    nodestrings : list of arrays
        A list of arrays, each of which is one contiguous section of open boundary

    """
    all_edges = np.vstack([triangle[:, 0:2], triangle[:, 1:], triangle[:, [0, 2]]])
    boundary_edges = all_edges[np.all(np.isin(all_edges, obc_node_array), axis=1), :]
    u_nodes, bdry_counts = np.unique(boundary_edges, return_counts=True)
    start_end_nodes = list(u_nodes[bdry_counts == 1])
    if len(start_end_nodes) == 0: # This is the case of all one open boundary i.e. an embedded domain
        return [obc_node_array]

    else:

        nodestrings = []

        while len(start_end_nodes) > 0:
            this_obc_section_nodes = [start_end_nodes[0]]
            start_end_nodes.remove(start_end_nodes[0])

            nodes_to_add = True

            while nodes_to_add:
                possible_nodes = np.unique(boundary_edges[np.any(np.isin(boundary_edges, this_obc_section_nodes), axis=1), :])
                nodes_to_add = list(possible_nodes[~np.isin(possible_nodes, this_obc_section_nodes)])
                if nodes_to_add:
                    this_obc_section_nodes.append(nodes_to_add[0])

            nodestrings.append(np.asarray(this_obc_section_nodes))
            start_end_nodes.remove(list(set(start_end_nodes).intersection(this_obc_section_nodes)))

        return nodestrings
