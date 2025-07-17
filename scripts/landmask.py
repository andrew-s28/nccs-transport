import numpy as np
import xarray as xr


def make_landmask(grid_file: str) -> np.ndarray:
    """Function that builds a landmask from a grid file.

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
    landmask = landmask.astype(int)
    return landmask


def get_coastal_nodes(landmask: np.ndarray, shift: int = 1) -> np.ndarray:
    """Function that detects the coastal nodes, i.e. the ocean nodes directly next to land. Computes the Laplacian of landmask.

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
    coastal = coastal.mask.astype("int")

    return coastal


def get_coastal_nodes_diagonal(landmask: np.ndarray, shift: int = 1) -> np.ma.masked_array:
    """Function that detects the coastal nodes, i.e. the ocean nodes where one of the 8 nearest nodes is land. Computes the Laplacian of landmask and the Laplacian of the 45 degree rotated landmask.

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
    mask_lap += np.roll(landmask, (-shift, shift), axis=(0, 1)) + np.roll(landmask, (shift, shift), axis=(0, 1))
    mask_lap += np.roll(landmask, (-shift, -shift), axis=(0, 1)) + np.roll(landmask, (shift, -shift), axis=(0, 1))
    mask_lap -= 8 * landmask
    coastal_diag = np.ma.masked_array(landmask, mask_lap > 0)
    coastal_diag = coastal_diag.mask.astype("int")

    return coastal_diag


def get_shore_nodes(landmask: np.ndarray, shift: int = 1) -> np.ma.masked_array:
    """Function that detects the shore nodes, i.e. the land nodes directly next to the ocean. Computes the Laplacian of landmask.

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
    shore = shore.mask.astype("int")

    return shore


def get_shore_nodes_diagonal(landmask: np.ndarray, shift: int = 1) -> np.ma.masked_array:
    """Function that detects the shore nodes, i.e. the land nodes where one of the 8 nearest nodes is ocean. Computes the Laplacian of landmask and the Laplacian of the 45 degree rotated landmask.

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
    mask_lap += np.roll(landmask, (-shift, shift), axis=(0, 1)) + np.roll(landmask, (shift, shift), axis=(0, 1))
    mask_lap += np.roll(landmask, (-shift, -shift), axis=(0, 1)) + np.roll(landmask, (shift, -shift), axis=(0, 1))
    mask_lap -= 8 * landmask
    shore = np.ma.masked_array(landmask, mask_lap < 0)
    shore = shore.mask.astype("int")

    return shore


def create_displacement_field(landmask: np.ndarray, shift: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Function that creates a displacement field 1 m/s away from the shore.

    Args:
        landmask (np.ndarray): The land mask built using `make_landmask`, where land cell = 1
                               and ocean cell = 0.
        shift (int): The shift to apply to the landmask. Default is 1.

    Returns:
        (vx, vy) ((np.ndarray, np.ndarray)): a tuple of 2D arrays, one for each component of the displacement velocity field (v_x, v_y).

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
    ly_c += np.roll(landmask, (-shift, -shift), axis=(0, 1)) + np.roll(landmask, (-shift, shift), axis=(0, 1))
    ly_c += -np.roll(landmask, (shift, -shift), axis=(0, 1)) - np.roll(landmask, (shift, shift), axis=(0, 1))

    lx_c = np.roll(landmask, -shift, axis=1) - np.roll(landmask, shift, axis=1)
    # Include x-component of diagonal neighbors
    lx_c += np.roll(landmask, (-shift, -shift), axis=(1, 0)) + np.roll(landmask, (-shift, shift), axis=(1, 0))
    lx_c += -np.roll(landmask, (shift, -shift), axis=(1, 0)) - np.roll(landmask, (shift, shift), axis=(1, 0))

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

    v_x /= magnitude
    v_y /= magnitude

    return v_x, v_y


def distance_to_shore(landmask: np.ndarray, dx: float = 1, shift: int = 1) -> np.ndarray:
    """Function that computes the distance to the shore. It is based in the the `get_coastal_nodes` algorithm.

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
