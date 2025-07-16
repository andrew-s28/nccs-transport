# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cartopy",
#     "cmocean",
#     "imageio-ffmpeg",
#     "matplotlib",
#     "numpy",
#     "tqdm",
#     "xarray[accel,io,parallel]",
# ]
# ///
import calendar
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean.cm as cmo
import imageio_ffmpeg
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection
from numpy.typing import NDArray
from tqdm import tqdm

from overwrite_cli import parse_args

if TYPE_CHECKING:
    from cartopy.mpl.geoaxes import GeoAxes
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def open_parcels_output(
    path: str | Path,
    load: bool = True,  # noqa: FBT001,FBT002
) -> xr.Dataset:
    """Open a Zarr file containing OceanParcels (https://docs.oceanparcels.org/en/latest/index.html) output.

    Automatically detects if the path is a single zarr file or if it is a directory containing multiple Zarr files from MPI runs,
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


def convert_to_time_index(ds: xr.Dataset, save_path: Path | None = None) -> xr.Dataset:
    """Convert the 'obs' dimension to a time index.

    Args:
        ds (xr.Dataset): The dataset containing the OceanParcels output.
        save_path (Path | None): Optional path to save the converted dataset as a Zarr file.

    Returns:
        ds (xr.Dataset): The dataset with the 'obs' dimension converted to a time index.

    """
    ds = ds.dropna(dim="obs", how="all")  # Drop observations that are all NaN
    ds["time"] = ds["time"].astype("datetime64[D]").astype("datetime64[ns]")  # Round to nearest day
    ds_list = [
        ds.isel(trajectory=i).dropna("obs").swap_dims({"obs": "time"}).drop_duplicates("time")
        for i in tqdm(ds["trajectory"], desc="Processing trajectory")
    ]
    ds = xr.concat(ds_list, dim="trajectory")
    ds["obs"] = ds["obs"].astype(np.float32)
    ds.chunk({"trajectory": -1, "time": 1})
    if save_path is not None:
        ds.to_zarr(save_path, mode="w")
        print(f"Dataset saved to {save_path}")
    return ds


def make_animation(ds: xr.Dataset, save_path: Path | str, trajectory_colors: list | NDArray) -> None:
    """Create an animation of the particle trajectories for a given year of particle releases.

    Args:
        ds (xr.Dataset): The dataset containing the particle trajectories.
        save_path (Path | str): The path where the animation will be saved.
        trajectory_colors (list | NDArray): An array of RGBA colors for each particle given by particle release month.

    """
    if isinstance(save_path, str):
        save_path = Path(save_path)

    fig: Figure
    ax: GeoAxes | Axes
    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": ccrs.PlateCarree()})
    fig.tight_layout(pad=2.0)
    # Need to explicitly cast to GeoAxes for type checking, since plt.subplots doesn't return different types based on subplot_kw
    ax = cast("GeoAxes", ax)

    sca_data = np.array([ds["lon"].isel(time=0), ds["lat"].isel(time=0)])
    sca = ax.scatter(
        sca_data[0],
        sca_data[1],
        transform=ccrs.PlateCarree(),
        c=trajectory_colors,
        rasterized=True,
    )
    ax.coastlines()
    ax.set_extent([-160, -120, 30, 60], crs=ccrs.PlateCarree())

    gl = ax.gridlines(ls="--", color="gray", alpha=0.5, linewidth=0.5)
    gl.bottom_labels = True
    gl.left_labels = True
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle="-", linewidth=1.5)
    ax.add_feature(cfeature.STATES, linestyle=":")

    cmap = mcolors.LinearSegmentedColormap.from_list("colors", trajectory_colors)
    norm = mcolors.BoundaryNorm(
        np.arange(0.5, 13, 1),
        cmap.N,  # type: ignore[reportAttributeAccessIssue]
        clip=True,
    )
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)  # type: ignore[reportAttributeAccessIssue]
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        orientation="vertical",
        label="Month of Release",
        ticks=np.arange(1, 13),
        pad=0.01,
    )
    cbar.ax.set_yticklabels([calendar.month_abbr[i] for i in range(1, 13)])
    cbar.ax.minorticks_off()

    def init() -> tuple[PathCollection]:
        return (sca,)

    def update(frame: int, sca: PathCollection) -> tuple[PathCollection]:
        sca_data = np.array([ds["lon"].isel(time=frame), ds["lat"].isel(time=frame)])
        sca.set_offsets(sca_data.T)
        time = ds["time"].to_numpy()[frame].astype("datetime64[D]").astype(str)
        ax.set_title(time)
        return (sca,)

    ani = FuncAnimation(fig, partial(update, sca=sca), frames=ds["time"].size, init_func=init, blit=True, interval=50)

    ani.save(str(save_path), writer="ffmpeg")


def month_colors(year: int, n_particles_per_day: int) -> NDArray:
    """Generate a list of colors for each month in the given year.

    Args:
        year (int): The year for which to generate month colors.
        n_particles_per_day (int): Number of particles released per day.

    Returns:
        trajectory_colors (NDArray): An array of RGBA colors for each particle given by particle release month.

    """
    month_lengths = [calendar.monthrange(year, i)[1] for i in range(1, 13)]
    month_colors = [cmo.phase(i / 12) for i in range(1, 13)]  # type: ignore[reportAttributeAccessIssue]
    trajectory_colors = np.concatenate(
        [np.full((ml * n_particles_per_day, 4), color) for color, ml in zip(month_colors, month_lengths, strict=True)],
    )
    return trajectory_colors


if __name__ == "__main__":
    # Set the data directory and years to process
    DATA_DIR = Path("D:/nccs-transport/forward_releases/")
    years = np.arange(1997, 2020)
    # Ensure the data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    for year in tqdm(years, desc="Processing years"):
        file = DATA_DIR / f"fwd_release_timeidx_140W_45-55N_{year}.zarr"
        if not file.exists():
            print(f"File {file} does not exist. Skipping year {year}.")
            continue
        ds_time = open_parcels_output(file)
        n_particles_per_day = int(ds_time.attrs["n_particles"])

        # Calculate trajectory colors for each month
        trajectory_colors = month_colors(year, n_particles_per_day)

        # Create the animation
        save_dir = DATA_DIR / "animations"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"fwd_release_animation_{year}.mp4"
        if save_path.exists():
            if args.prompt:
                response = input(f"Animation for {year} already exists. Overwrite? (y/n): ")
                if response.lower() != "y":
                    print("Skipping year:", year)
                    continue
            elif not args.force:
                print(f"Animation for {year} already exists. Use --force to overwrite.")
                continue
        make_animation(ds_time, save_path, trajectory_colors)
