"""Convert Parcels output to time-indexed format."""

from datetime import timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

from altimeter_tracking import ParcelsConfig
from overwrite_cli import parse_args
from utils import open_parcels_output


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
    years = np.arange(1997, 2020, dtype=int)
    config = ParcelsConfig(
        min_lon_release=-127,
        max_lon_release=-125,
        d_lon_release=0.1,
        min_lat_release=45,
        max_lat_release=55,
        d_lat_release=0.1,
        runtime=timedelta(days=366 * 3),
        velocity_file=Path("D:/nccs-transport/combined_velocity_dataset.nc"),
        output_dir=Path("D:/nccs-transport/forward_releases"),
        year_release=years[0],
    )  # Just using ParcelsConfig to find output paths
    args = parse_args("Converting Parcels output to time-indexed format.")
    for year in tqdm(years, desc="Processing years"):
        config.update_year_release(year)
        config.set_start_end_times()
        config.set_output_file()
        output_file_name = config.output_file.name.replace("fwd_release", "fwd_release_timeidx")
        save_path = config.output_dir / output_file_name
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
