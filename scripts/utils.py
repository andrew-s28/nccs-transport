"""Utilities for running OceanParcels Lagrangian simulations.

Currently holds functions for opening OceanParcels outputs,
computing the land mask of an ocean model (i.e., where land is located),
detecting coastal nodes near land,
and creating an artificial displacement field to prevent stuck particles near land.
"""

from pathlib import Path

import numpy as np
import xarray as xr


def open_parcels_output(
    path: str | Path,
    load: bool = True,  # noqa: FBT001, FBT002
) -> xr.Dataset:
    """Open a Zarr file containing OceanParcels (https://docs.oceanparcels.org/en/latest/index.html) output.

    Automatically detects if the path is a single zarr file or if it
    is a directory containing multiple Zarr files from MPI runs,
    as detailed here: https://docs.oceanparcels.org/en/latest/examples/documentation_MPI.html.

    Args:
        path (str or Path): Path to the Zarr file or directory containing the Zarr files.
        load (bool): If True, load the dataset into memory. Defaults to True.

    Returns:
        ds (xr.Dataset): The dataset containing the OceanParcels output.

    Raises:
        FileNotFoundError: If the specified path does not exist.

    """
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        msg = f"Path {path} does not exist."
        raise FileNotFoundError(msg)
    mpi_files = list(path.glob("proc*"))
    if len(mpi_files) == 0:
        ds = xr.open_zarr(path)
    else:
        ds = xr.concat(
            [xr.open_zarr(f) for f in mpi_files],
            dim="trajectory",
            compat="no_conflicts",
            coords="minimal",
        )
    if load:
        ds.load()  # Load the dataset into memory
    return ds


def make_landmask(grid_file: str | Path) -> np.ndarray:
    """Build a landmask from a grid file.

    Args:
        grid_file (str): Path to the grid file.

    Returns:
        landmask (np.ndarray): 2D array containing the landmask, where land cell = 1
                    and ocean cell = 0.

    """
    landmask = xr.open_dataset(
        grid_file,
    ).mask_rho_no_rivers.to_numpy()
    landmask = np.invert(landmask.astype(bool)).astype(int)

    return landmask.astype(int)


def get_coastal_nodes(landmask: np.ndarray, shift: int = 1) -> np.ndarray:
    """Detect the coastal nodes, i.e., the ocean nodes directly next to land.

    Computes the Laplacian of landmask.

    Args:
        landmask (np.ndarray): The land mask built using `make_landmask`, where land cell = 1
                               and ocean cell = 0.
        shift (int): The shift to apply to the landmask. Default is 1.

    Returns:
        coastal (np.ndarray): 2D array containing the coastal nodes, the coastal nodes are
                    equal to one, and the rest is zero.

    """
    mask_lap = np.roll(landmask, -shift, axis=0) + np.roll(landmask, shift, axis=0)
    mask_lap += np.roll(landmask, -shift, axis=1) + np.roll(landmask, shift, axis=1)
    mask_lap -= 4 * landmask
    coastal = np.ma.masked_array(landmask, mask_lap > 0)

    return coastal.mask.astype("int")


def get_coastal_nodes_diagonal(
    landmask: np.ndarray,
    shift: int = 1,
) -> np.ma.masked_array:
    """Detect the coastal nodes, i.e. the ocean nodes where one of the 8 nearest nodes is land.

    Computes the Laplacian of landmask and the Laplacian of the 45 degree rotated landmask.

    Args:
        landmask (np.ndarray): The land mask built using `make_landmask`, where land cell = 1
                               and ocean cell = 0.
        shift (int): The shift to apply to the landmask. Default is 1.

    Returns:
        coastal_diag (np.ma.masked_array): 2D array containing the diagonal coastal nodes,
                    the diagonal coastal nodes are equal to one, and the rest is zero.

    """
    mask_lap = np.roll(landmask, -shift, axis=0) + np.roll(landmask, shift, axis=0)
    mask_lap += np.roll(landmask, -shift, axis=1) + np.roll(landmask, shift, axis=1)
    mask_lap += np.roll(landmask, (-shift, shift), axis=(0, 1)) + np.roll(
        landmask,
        (shift, shift),
        axis=(0, 1),
    )
    mask_lap += np.roll(landmask, (-shift, -shift), axis=(0, 1)) + np.roll(
        landmask,
        (shift, -shift),
        axis=(0, 1),
    )
    mask_lap -= 8 * landmask
    coastal_diag = np.ma.masked_array(landmask, mask_lap > 0)

    return coastal_diag.mask.astype("int")


def get_shore_nodes(landmask: np.ndarray, shift: int = 1) -> np.ma.masked_array:
    """Detect the shore nodes, i.e. the land nodes directly next to the ocean.

    Computes the Laplacian of landmask.

    Args:
        landmask (np.ndarray): The land mask built using `make_landmask`, where land cell = 1
                               and ocean cell = 0.
        shift (int): The shift to apply to the landmask. Default is 1.

    Returns:
        np.ndarray: 2D array containing the shore nodes, the
            shore nodes are equal to one, and the rest is zero.

    """
    mask_lap = np.roll(landmask, -shift, axis=0) + np.roll(landmask, shift, axis=0)
    mask_lap += np.roll(landmask, -shift, axis=1) + np.roll(landmask, shift, axis=1)
    mask_lap -= 4 * landmask
    shore = np.ma.masked_array(landmask, mask_lap < 0)

    return shore.mask.astype("int")


def get_shore_nodes_diagonal(
    landmask: np.ndarray,
    shift: int = 1,
) -> np.ma.masked_array:
    """Detect the shore nodes, i.e. the land nodes where one of the 8 nearest nodes is ocean.

    Computes the Laplacian of landmask and the Laplacian of the 45 degree rotated landmask.

    Args:
        landmask (np.ndarray): The land mask built using `make_landmask`, where land cell = 1
                               and ocean cell = 0.
        shift (int): The shift to apply to the landmask. Default is 1.

    Returns:
        shore (np.ma.masked_array): 2D array containing the shore nodes, the
            shore nodes are equal to one, and the rest is zero.

    """
    mask_lap = np.roll(landmask, -shift, axis=0) + np.roll(landmask, shift, axis=0)
    mask_lap += np.roll(landmask, -shift, axis=1) + np.roll(landmask, shift, axis=1)
    mask_lap += np.roll(landmask, (-shift, shift), axis=(0, 1)) + np.roll(
        landmask,
        (shift, shift),
        axis=(0, 1),
    )
    mask_lap += np.roll(landmask, (-shift, -shift), axis=(0, 1)) + np.roll(
        landmask,
        (shift, -shift),
        axis=(0, 1),
    )
    mask_lap -= 8 * landmask
    shore = np.ma.masked_array(landmask, mask_lap < 0)

    return shore.mask.astype("int")


def create_displacement_field(
    landmask: np.ndarray,
    shift: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a displacement field 1 m/s away from the shore.

    Args:
        landmask (np.ndarray): The land mask built using `make_landmask`, where land cell = 1
                               and ocean cell = 0.
        shift (int): The shift to apply to the landmask. Default is 1.

    Returns:
        (vx, vy) ((np.ndarray, np.ndarray)): a tuple of 2D arrays,
            one for each component of the displacement velocity field (v_x, v_y).

    """
    shore = get_coastal_nodes(landmask, shift)

    # nodes bordering ocean directly and diagonally
    shore_d = get_coastal_nodes_diagonal(landmask, shift)
    # corner nodes that only border ocean diagonally
    shore_c = shore_d - shore

    # Simple derivative
    ly = np.roll(landmask, -shift, axis=0) - np.roll(landmask, shift, axis=0)
    lx = np.roll(landmask, -shift, axis=1) - np.roll(landmask, shift, axis=1)

    ly_c = np.roll(landmask, -shift, axis=0) - np.roll(landmask, shift, axis=0)
    # Include y-component of diagonal neighbors
    ly_c += np.roll(landmask, (-shift, -shift), axis=(0, 1)) + np.roll(
        landmask,
        (-shift, shift),
        axis=(0, 1),
    )
    ly_c += -np.roll(landmask, (shift, -shift), axis=(0, 1)) - np.roll(
        landmask,
        (shift, shift),
        axis=(0, 1),
    )

    lx_c = np.roll(landmask, -shift, axis=1) - np.roll(landmask, shift, axis=1)
    # Include x-component of diagonal neighbors
    lx_c += np.roll(landmask, (-shift, -shift), axis=(1, 0)) + np.roll(
        landmask,
        (-shift, shift),
        axis=(1, 0),
    )
    lx_c += -np.roll(landmask, (shift, -shift), axis=(1, 0)) - np.roll(
        landmask,
        (shift, shift),
        axis=(1, 0),
    )

    v_x = -lx * (shore)
    v_y = -ly * (shore)

    v_x_c = -lx_c * (shore_c)
    v_y_c = -ly_c * (shore_c)

    v_x += v_x_c
    v_y += v_y_c

    magnitude = np.sqrt(v_y**2 + v_x**2)
    # the coastal nodes between land create a problem. Magnitude there is zero
    # I force it to be 1 to avoid problems when normalizing.
    ny, nx = np.where(magnitude == 0)
    magnitude[ny, nx] = 1

    v_x = v_x.astype(float)
    v_y = v_y.astype(float)

    v_x /= magnitude
    v_y /= magnitude

    return v_x, v_y


def compute_distance_to_shore(
    landmask: np.ndarray,
    dx: float = 1,
    shift: int = 1,
) -> np.ndarray:
    """Compute the distance to the shore, based on the `get_coastal_nodes` algorithm.

    Args:
        landmask (np.ndarray): The land mask built using `make_landmask`, where land cell = 1
                               and ocean cell = 0.
        dx (float): The grid cell dimension. This is a crude approximation of the real
                    distance (be careful). Default is 1.
        shift (int): The shift to apply to the landmask. Default is 1.

    Returns:
        dist (np.ndarray): 2D array containing the distances from shore, where the distance is

    """
    ci = get_coastal_nodes(landmask, shift)  # direct neighbors
    dist = ci * dx  # 1 dx away

    ci_d = get_coastal_nodes_diagonal(landmask, shift)  # diagonal neighbors
    dist_d = (ci_d - ci) * np.sqrt(2 * dx**2)  # sqrt(2) dx away

    return dist + dist_d
