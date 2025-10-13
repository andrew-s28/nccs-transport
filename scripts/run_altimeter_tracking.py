"""Wrapper for running OceanParcels simulations."""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "parcels",
#     "xarray[accel,io,parallel]",
#     "zarr<3",
# ]
# ///
from datetime import datetime, timedelta
from pathlib import Path

from altimeter_tracking import ParcelsConfig as ParcelsAltimeterConfig
from altimeter_tracking import execute_backward_release, write_metadata
from overwrite_cli import parse_args

local_tz = datetime.now().astimezone().tzinfo

if __name__ == "__main__":
    years = range(1994, 1999)
    config = ParcelsAltimeterConfig(
        d_lon_release=0.025,
        d_lat_release=0.025,
        n_particles=64,
        velocity_file=Path("/mnt/d/nccs-transport/combined_velocity_dataset.nc"),
        input_dir=Path("/mnt/d/plankton_particle_tracks"),
        output_dir=Path("/mnt/d/plankton_particle_tracks/test"),
        runtime=timedelta(days=210),
        year_release=1997,
        max_age=timedelta(days=180),
    )
    args = parse_args("Run OceanParcels plankton tracking simulations.")
    for year in years:
        config.update_year_release(year)
        if config.output_file.exists():
            if args.prompt:
                response = input(
                    f"Output file for year {year} already exists. Overwrite? (y/n): ",
                )
                if response.lower() != "y":
                    print("Skipping year:", year)
                    continue
            elif not args.force:
                print(
                    f"Output file for year {year} already exists. Use --force to overwrite.",
                )
                continue
        t_start = datetime.now(tz=local_tz)
        print(
            f"{t_start.strftime('%Y-%m-%d %H:%M:%S')}: starting backward release for year {year}.",
        )
        execute_backward_release(config)
        write_metadata(config)
        t_finish = datetime.now(tz=local_tz)
        print(
            f"{t_finish.strftime('%Y-%m-%d %H:%M:%S')}: "
            f"completed backward release for year {year}"
            f"in {timedelta(seconds=(t_finish - t_start).seconds)!s}.",
        )
