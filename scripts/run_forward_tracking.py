"""Wrapper for running forward tracking OceanParcels simulations using altimeter velocity fields."""

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "parcels",
#     "xarray",
#     "zarr<3",
# ]
# ///

from datetime import timedelta
from pathlib import Path

import numpy as np

from scripts.forward_altimeter_tracking import ParcelsConfig as ParcelsAltimeterConfig
from scripts.forward_altimeter_tracking import execute_forward_release, write_metadata
from scripts.overwrite_cli import parse_args

if __name__ == "__main__":
    years = np.arange(1997, 2020)
    config = ParcelsAltimeterConfig(
        lon_release=-140,
        min_lat_release=40,
        max_lat_release=50,
        n_particles=100,
        velocity_file=Path("D:/nccs-transport/combined_velocity_dataset.nc"),
        output_path=Path("D:/nccs-transport/forward_releases"),
        runtime=timedelta(days=366 * 3),
    )
    args = parse_args("Run OceanParcels plankton tracking simulations.")
    for year in years:
        config.set_start_end_times(year)
        config.set_output_file(year)
        if config.output_file.exists():
            if args.prompt:
                response = input(f"Output file for year {year} already exists. Overwrite? (y/n): ")
                if response.lower() != "y":
                    print("Skipping year:", year)
                    continue
            elif not args.force:
                print(f"Output file for year {year} already exists. Use --force to overwrite.")
                continue
        execute_forward_release(config)
        write_metadata(config)
        print(f"Completed forward release for year {year}.")
