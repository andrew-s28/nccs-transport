import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

from forward_altimeter_tracking import ParcelsConfig
from overwrite_cli import parse_args


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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")
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
        ds.sel(trajectory=i).dropna("obs").swap_dims({"obs": "time"}).drop_duplicates("time")
        for i in tqdm(ds["trajectory"], desc="Processing trajectory", leave=False)
    ]
    ds = xr.concat(ds_list, dim="trajectory")
    ds["obs"] = ds["obs"].astype(np.float32)
    ds.chunk({"trajectory": -1, "time": 1})
    if save_path is not None:
        ds.to_zarr(save_path, mode="w")
    return ds


if __name__ == "__main__":
    DATA_DIR = Path("D:/nccs-transport/forward_releases/")
    years = np.arange(1997, 2020)
    config = ParcelsConfig(
        lon_release=-140,
        min_lat_release=45,
        max_lat_release=55,
        n_particles=50,
        runtime=timedelta(days=366 * 3),
        velocity_file=Path("D:/nccs-transport/combined_velocity_dataset.nc"),
        output_path=Path("D:/nccs-transport/forward_releases"),
    )  # Just using ParcelsConfig to find output paths
    args = parse_args()
    for year in tqdm(years, desc="Processing years"):
        config.set_start_end_times(year)
        config.set_output_file(year)
        output_file_name = config.output_file.name.replace("fwd_release", "fwd_release_timeidx")
        save_path = config.output_path / output_file_name
        if save_path.exists():
            if args.prompt:
                response = input(f"Time-indexed output for {year} already exists. Overwrite? (y/n): ")
                if response.lower() != "y":
                    print("Skipping year:", year)
                    continue
            elif not args.force:
                print(f"Time-indexed output for {year} already exists. Use --force to overwrite.")
                continue
        ds = open_parcels_output(config.output_file)
        ds = convert_to_time_index(ds, save_path=save_path)
