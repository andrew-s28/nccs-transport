"""Forward particle tracking using altimeter-derived geostrophic velocities with Parcels and MPI."""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "dask[complete]",
#     "mpi4py",
#     "mpich",
#     "numpy",
#     "parcels",
#     "scikit-learn",
#     "xarray",
#     "zarr<3",
# ]
# ///
import platform
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mpi4py import MPI
from parcels import FileWarning

from scripts.forward_altimeter_tracking import ParcelsConfig, execute_forward_release, write_metadata

##############################################
# User-defined parameters for the simulation #
##############################################

LON_RELEASE = -140
MIN_LAT_RELEASE = 45
MAX_LAT_RELEASE = 55
YEAR_RELEASE = 2017
N_PARTICLES = 50

VELOCITY_FILE_PATH = Path("/mnt/d/nccs-transport/combined_velocity_dataset.nc")
OUTPUT_PATH = Path("/mnt/d/nccs-transport/forward_releases")

START_TIME = datetime(YEAR_RELEASE, 1, 1, tzinfo=UTC)  # Start time of the release
END_TIME = datetime(YEAR_RELEASE, 12, 31, tzinfo=UTC)  # End time of the release
# Create output directory if it does not exist
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_PATH / f"fwd_release_140W_45-55N_{YEAR_RELEASE!s}.zarr"  # Output file name

OUTPUT_CHUNKS = (int(1e4), 100)  # Output chunk size
OUTPUT_TIMESTEP = timedelta(days=1)
RUNTIME = timedelta(days=30)  # Total length of the run
ADVECTION_TIMESTEP = timedelta(minutes=10)  # Timestep of the advection simulation


def finalize_mpi_communication(config: ParcelsConfig) -> None:
    """Synchronize MPI runners and then create consistent metadata in the output files.

    Args:
        config (ParcelsConfig): Configuration object containing simulation parameters.

    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm.barrier()

    try:
        if rank == 0:
            files = Path(VELOCITY_FILE_PATH).glob("proc*")
            for _f in files:
                write_metadata(config)
    except Exception as e:  # noqa: BLE001
        print(e)


if __name__ == "__main__":
    config = ParcelsConfig(
        lon_release=LON_RELEASE,
        min_lat_release=MIN_LAT_RELEASE,
        max_lat_release=MAX_LAT_RELEASE,
        year_release=YEAR_RELEASE,
        n_particles=N_PARTICLES,
        velocity_file=VELOCITY_FILE_PATH,
        output_path=OUTPUT_PATH,
        start_time=START_TIME,
        end_time=END_TIME,
        output_chunks=OUTPUT_CHUNKS,
        output_timestep=OUTPUT_TIMESTEP,
        advection_timestep=ADVECTION_TIMESTEP,
        runtime=RUNTIME,
    )

    # Execute the forward release simulation
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The ParticleFile name contains .zarr extension", FileWarning)
        execute_forward_release(config)

    if platform.system() == "Linux":
        # Finalize MPI communication if running on Linux
        finalize_mpi_communication(config)
