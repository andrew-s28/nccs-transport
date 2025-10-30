"""Primary code for setting up and executing an OceanParcels Lagrangian simulation using altimeter-derived velocity fields."""  # noqa: E501

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

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import parcels
import xarray as xr
import zarr


@dataclass
class ParcelsConfig:
    """Dataclass to hold the configuration for Parcels simulations using altimeter-derived velocity fields."""

    min_lon_release: float  # Minimum longitude for particle release
    max_lon_release: float  # Maximum longitude for particle release
    d_lon_release: float  # Longitude between release points
    min_lat_release: float  # Minimum latitude for particle release
    max_lat_release: float  # Maximum latitude for particle release
    d_lat_release: float  # Latitude between release points
    velocity_file: Path
    output_dir: Path
    runtime: timedelta
    year_release: int
    distance_to_shore_limit: float = 0.1  # Distance to shore limit for offshore displacement, between 0 and 1
    output_chunks: tuple[int, int] = (10_000, 100)  # Output chunk size
    output_timestep: timedelta = timedelta(
        days=1,
    )  # Output timestep for the particle file
    advection_timestep: timedelta = timedelta(
        minutes=-10,
    )  # Timestep of the advection simulation
    max_age: timedelta = timedelta(days=365)  # Maximum age of particles before deletion

    def __post_init__(self) -> None:
        """Initialize the ParcelsConfig dataclass."""
        self.set_start_end_times()
        self.set_output_file()
        self.set_output_file()

    def update_year_release(self, year_release: int) -> None:
        """Update the year of release and reconfigure the model files, start and end times, and output file."""
        self.year_release = year_release
        self.set_start_end_times()
        self.set_output_file()

    def set_start_end_times(self) -> None:
        """Set the start and end times for the simulation."""
        self.start_time = datetime(self.year_release, 12, 31, tzinfo=UTC)
        self.end_time = datetime(self.year_release, 1, 1, tzinfo=UTC)
        if self.start_time > self.end_time:
            self.advection_timestep = -abs(self.advection_timestep)
        else:
            self.advection_timestep = abs(self.advection_timestep)

    def set_output_file(self) -> None:
        """Set the output file path based on the year."""
        if self.start_time > self.end_time:
            self.output_file = self.output_dir / f"bwd_release_altimeter_nhl_{self.year_release!s}.zarr"
        else:
            self.output_file = self.output_dir / f"fwd_release_altimeter_nhl_{self.year_release!s}.zarr"


def fp_arange(
    start: float,
    stop: float,
    step: float,
) -> np.ndarray:
    """Create an array with a floating point start, stop, and step.

    Ensures that the array creation will behave as intended by decimal arithmetic.
    Try `np.arange(15, 15.8, 0.1)` vs `fp_arange(15, 15.8, 0.1)`.
    Tries to avoid floating point weirdness.
    Barely tested, use at your own risk.

    Args:
        start (float): Start value.
        stop (float): Stop value.
        step (float): Step value.

    Returns:
        np.ndarray: Array of values from start to stop with the given step.

    """
    decimal_start = Decimal(str(start))
    decimal_stop = Decimal(str(stop))
    decimal_step = Decimal(str(step))

    return np.arange(decimal_start, decimal_stop, decimal_step).astype(float)


def setup_fieldset(config: ParcelsConfig) -> parcels.FieldSet:
    """Create the fieldset for particle tracking.

    Args:
        config (ParcelsConfig): Configuration object containing model and grid information.

    Returns:
        fieldset (parcels.FieldSet): FieldSet object containing the necessary fields for particle tracking.

    """
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

    fieldset.add_constant(
        "max_age",
        config.runtime.total_seconds() if config.max_age is None else config.max_age.total_seconds(),
    )

    return fieldset


def setup_kernels() -> list[Callable[..., None]]:
    """Create the kernels for particle tracking.

    Returns:
        kernels (list): List of kernel functions to be used in the particle tracking simulation.

    """

    def particle_age(particle, fieldset, time) -> None:  # noqa: ARG001, ANN001
        particle.age += particle.dt

    def delete_old_particle(particle, fieldset, time) -> None:  # noqa: ARG001, ANN001
        if particle.age > fieldset.max_age:
            particle.delete()
            particle.delete_by_age = 1
        # Handle backward releases
        if particle.age < -fieldset.max_age:
            particle.delete()
            particle.delete_by_age = 1
        if particle.age >= -fieldset.max_age and particle.age <= fieldset.max_age:
            particle.delete_by_age = 0

    def handle_error_particle(particle, fieldset, time) -> None:  # noqa: ARG001, ANN001
        if particle.state == StatusCode.ErrorOutOfBounds:  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
            particle.delete()
            particle.delete_by_out_of_bound = 1
        else:
            particle.delete_by_out_of_bound = 0

    kernels = [particle_age, parcels.AdvectionRK4, handle_error_particle, delete_old_particle]

    return kernels


def get_release_positions(  # noqa: PLR0913
    start_time: datetime,
    end_time: datetime,
    min_lon_release: float,
    max_lon_release: float,
    d_lon_release: float,
    min_lat_release: float,
    max_lat_release: float,
    d_lat_release: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the release positions for the particles based on the station file.

    Args:
        start_time (datetime): Start time of the simulation.
        end_time (datetime): End time of the simulation.
        min_lon_release (float): Minimum longitude for particle release.
        max_lon_release (float): Maximum longitude for particle release.
        d_lon_release (float): Longitude between release points.
        min_lat_release (float): Minimum latitude for particle release.
        max_lat_release (float): Maximum latitude for particle release.
        d_lat_release (float): Latitude between release points.
        depth_release (float): Depth at which particles are released.

    Returns:
        (release_lons, release_lats, release_depths, release_times) (tuple):
            A tuple containing arrays of release longitudes, latitudes, depths, and times.

    """
    # Setup release locations and times
    release_times = xr.date_range(start_time, end_time, freq="-1D").to_numpy(
        dtype="datetime64[ns]",
    )

    lons = fp_arange(min_lon_release, max_lon_release + d_lon_release, d_lon_release)
    lats = fp_arange(min_lat_release, max_lat_release + d_lat_release, d_lat_release)
    release_lons, release_lats = np.meshgrid(lons, lats, indexing="ij")
    release_lons = np.tile(release_lons[None, :, :], (release_times.shape[0], 1, 1))
    release_lats = np.tile(release_lats[None, :, :], (release_times.shape[0], 1, 1))
    release_times = np.tile(release_times[:, None, None], (1, release_lons.shape[1], release_lons.shape[2]))

    return release_lons, release_lats, release_times


def execute_backward_release(
    config: ParcelsConfig,
) -> None:
    """Execute the backward release simulation using Parcels.

    Args:
        config (ParcelsConfig): Configuration object containing simulation parameters.

    """
    fieldset = setup_fieldset(config)

    kernels = setup_kernels()

    release_lons, release_lats, release_times = get_release_positions(
        config.start_time,
        config.end_time,
        config.min_lon_release,
        config.max_lon_release,
        config.d_lon_release,
        config.min_lat_release,
        config.max_lat_release,
        config.d_lat_release,
    )

    class Particle(parcels.JITParticle):
        age = parcels.Variable(
            "age",
            initial=int(-config.advection_timestep.total_seconds()),
        )
        delete_by_age = parcels.Variable("delete_by_age", initial=0)
        delete_by_out_of_bound = parcels.Variable("delete_by_out_of_bound", initial=0)

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
    """Write metadata to the output file.

    Args:
        config (ParcelsConfig): Configuration object containing simulation parameters.

    """
    metadata = {
        "min_lon_release": str(config.min_lon_release),
        "max_lon_release": str(config.max_lon_release),
        "d_lon": str(config.d_lon_release),
        "min_lat_release": str(config.min_lat_release),
        "max_lat_release": str(config.max_lat_release),
        "d_lat": str(config.d_lat_release),
        "min_time": str(config.start_time),
        "max_time": str(config.end_time),
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
