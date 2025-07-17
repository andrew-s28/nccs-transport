# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "dask[complete]",
#     "numpy",
#     "parcels",
#     "xarray",
#     "zarr<3",
# ]
# ///
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import parcels
import xarray as xr
import zarr

##############################################
# User-defined parameters for the simulation #
##############################################

LON_RELEASE = -140
MIN_LAT_RELEASE = 45
MAX_LAT_RELEASE = 55
YEAR_RELEASE = 2017
N_PARTICLES = 50

VELOCITY_FILE = Path("D:/nccs-transport/combined_velocity_dataset.nc")
OUTPUT_PATH = Path("D:/nccs-transport/forward_releases")

# Create output directory if it does not exist
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_PATH / f"fwd_release_140W_45-55N_{YEAR_RELEASE!s}.zarr"  # Output file name

OUTPUT_CHUNKS = (int(1e4), 100)  # Output chunk size
OUTPUT_TIMESTEP = timedelta(days=1)
RUNTIME = timedelta(days=30)  # Total length of the run
ADVECTION_TIMESTEP = timedelta(minutes=10)  # Timestep of the advection simulation


@dataclass
class ParcelsConfig:
    """Dataclass to hold the configuration for Parcels simulations."""

    lon_release: float
    min_lat_release: float
    max_lat_release: float
    n_particles: int
    velocity_file: Path
    output_path: Path
    runtime: timedelta
    year_release: int = YEAR_RELEASE
    output_chunks: tuple = OUTPUT_CHUNKS
    output_timestep: timedelta = OUTPUT_TIMESTEP
    advection_timestep: timedelta = ADVECTION_TIMESTEP

    def __post_init__(self) -> None:
        """Initialize the ParcelsConfig dataclass."""
        self.set_start_end_times(self.year_release)
        self.set_output_file(self.year_release)

    def set_start_end_times(self, year_release: int) -> None:
        """Set the start and end times for the simulation."""
        self.year_release = year_release
        self.start_time = datetime(year_release, 1, 1, tzinfo=UTC)
        self.end_time = datetime(year_release, 12, 31, tzinfo=UTC)

    def set_output_file(self, year_release: int) -> None:
        """Set the output file path based on the year."""
        self.year_release = year_release
        east_west = "W" if self.lon_release < 0 else "E"
        if self.start_time > self.end_time:
            self.output_file = (
                self.output_path
                / f"bwd_release_{abs(self.lon_release)}{east_west}_{self.min_lat_release}-{self.max_lat_release}N_{self.year_release!s}.zarr"
            )
        else:
            self.output_file = (
                self.output_path
                / f"fwd_release_{abs(self.lon_release)}{east_west}_{self.min_lat_release}-{self.max_lat_release}N_{self.year_release!s}.zarr"
            )


def execute_forward_release(
    config: ParcelsConfig,
) -> None:
    """Function that executes a forward release simulation using the Parcels library."""
    variables = {
        "U": "u",
        "V": "v",
    }
    dimensions = {
        "U": {"time": "time", "lon": "longitude", "lat": "latitude"},
        "V": {"time": "time", "lon": "longitude", "lat": "latitude"},
    }

    fieldset = parcels.FieldSet.from_netcdf(
        config.velocity_file,
        variables=variables,
        dimensions=dimensions,
        chunksize="auto",
    )

    # Determine if the release is backwards or forwards in time
    if config.end_time < config.start_time:
        release_times = xr.date_range(config.start_time, config.end_time, freq="-1D").to_numpy(dtype="datetime64[ns]")
    else:
        release_times = xr.date_range(config.start_time, config.end_time, freq="1D").to_numpy(dtype="datetime64[ns]")

    shape = (release_times.size, 1, config.n_particles)
    release_lats = np.linspace(config.min_lat_release, config.max_lat_release, config.n_particles)
    release_lons = np.array([config.lon_release])
    release_lats, release_lons = np.meshgrid(release_lats, release_lons, indexing="xy")
    release_lats = np.tile(release_lats, shape[0]).reshape(shape)
    release_lons = np.tile(release_lons, shape[0]).reshape(shape)
    release_times = np.tile(release_times[:, None, None], (1, release_lons.shape[-1]))

    class Particle(parcels.JITParticle):
        age = parcels.Variable("age", initial=0)

    def particle_age(particle, fieldset, time) -> None:  # noqa: ARG001
        if particle.time > 0:
            particle.age += particle.dt / 86400

    def delete_old_particle(particle, fieldset, time) -> None:  # noqa: ARG001
        if particle.age > 365:  # noqa: PLR2004; can't use any global variables in this function
            particle.delete()
        if particle.age < -365:  # noqa: PLR2004; can't use any global variables in this function
            particle.delete()

    def handle_error_particle(particle, fieldset, time) -> None:  # noqa: ARG001
        if particle.state == StatusCode.ErrorOutOfBounds and particle.age > -1:  # type: ignore[reportUndefinedVariable] # noqa: F821
            particle.delete()

    kernels = [particle_age, parcels.AdvectionRK4, handle_error_particle]

    pset = parcels.ParticleSet(
        fieldset=fieldset,
        pclass=Particle,
        lon=release_lons,
        lat=release_lats,
        time=release_times,
    )

    output_particle_file = pset.ParticleFile(
        name=config.output_file,
        outputdt=config.output_timestep,
        chunks=config.output_chunks,
    )

    pset.execute(
        kernels,
        runtime=config.runtime,
        dt=config.advection_timestep,
        output_file=output_particle_file,
    )


def write_metadata(config: ParcelsConfig) -> None:
    """Function to write metadata to the output file."""
    metadata = {
        "min_latitude": str(config.min_lat_release),
        "max_latitude": str(config.max_lat_release),
        "longitude": str(config.lon_release),
        "min_time": str(config.start_time),
        "max_time": str(config.end_time),
        "n_particles": str(config.n_particles),
        "input_velocity_fields": str(config.velocity_file),
        "runtime": str(config.runtime),
        "advection_timestep": str(config.advection_timestep),
        "output_timestep": str(config.output_timestep),
        "output_chunks": str(config.output_chunks),
    }

    group = zarr.open_group(config.output_file)
    group.attrs.update(
        metadata,
    )
    zarr.consolidate_metadata(config.output_file)  # type: ignore[reportArgumentType]


if __name__ == "__main__":
    config = ParcelsConfig(
        lon_release=LON_RELEASE,
        min_lat_release=MIN_LAT_RELEASE,
        max_lat_release=MAX_LAT_RELEASE,
        n_particles=N_PARTICLES,
        velocity_file=VELOCITY_FILE,
        output_path=OUTPUT_PATH,
        runtime=RUNTIME,
    )
    # Execute the forward release simulation
    execute_forward_release(config)
    # Write metadata to the output file
    write_metadata(config)
