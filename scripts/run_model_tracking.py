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

from model_tracking import ParcelsConfig as ParcelsModelConfig
from model_tracking import execute_backward_release, write_metadata
from overwrite_cli import parse_args

local_tz = datetime.now().astimezone().tzinfo

if __name__ == "__main__":
    years = range(1999, 2018)
    config = ParcelsModelConfig(
        min_lon_release=-127,
        max_lon_release=-125,
        d_lon_release=0.1,  # a bit coarse for test releases
        min_lat_release=44.6,
        max_lat_release=44.6,  # only want one latitude for test releases
        d_lat_release=0.1,
        model_dir=Path("/mnt/d/avg/"),
        output_dir=Path("/mnt/d/nccs-transport/tracks/model/"),
        runtime=timedelta(days=365 * 5),
        year_release=years[0],
        max_age=timedelta(days=365 * 3),
        depth_release=150,
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
