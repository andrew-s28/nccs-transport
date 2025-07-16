# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "xarray[accel,io,parallel]",
# ]
# ///
"""Calculates geostrophic velocities from altimeter data with mean dynamic topography correction and Ekman velocities from OSCAR."""

from datetime import UTC, datetime
from pathlib import Path

import xarray as xr


def _convert_dict_to_string(d: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in d.items() if v is not None)


AVISO_PATH = Path("D:/nccs-transport/aviso/aviso.nc")
MDT_PATH = Path("D:/nccs-transport/mdt/mdt_cnes_cls22_global.nc")
OSCAR_PATH = Path("D:/nccs-transport/oscar/oscar.nc")
SAVE_PATH = Path("D:/nccs-transport/combined_velocity_dataset.nc")

aviso = xr.open_mfdataset(AVISO_PATH, chunks={"time": 1})
mdt = xr.open_mfdataset(MDT_PATH)
oscar = xr.open_mfdataset(OSCAR_PATH, chunks={"time": 1}, decode_times={"time": True})

# rename OSCAR to match AVISO and MDT
oscar = oscar.swap_dims({"latitude": "lat", "longitude": "lon"}).rename({"lat": "latitude", "lon": "longitude"})
# Convert AVISO time to standard calendar
oscar = oscar.convert_calendar("standard", use_cftime=False)
# Convert longitude to -180 to 180 range for OSCAR
oscar["longitude"] = oscar["longitude"] - 360  # noqa: PLR6104; xarray indices are immutable so -= 360 does not work

# Interpolate MDT and OSCAR data to AVISO grid
mdt = mdt.interp(
    longitude=aviso.longitude,
    latitude=aviso.latitude,
    method="linear",
).squeeze()
oscar = oscar.interp(
    longitude=aviso.longitude,
    latitude=aviso.latitude,
    method="linear",
).squeeze()

# Align datasets in time and space
aviso_al, oscar_al, mdt_al = xr.align(aviso, oscar, mdt)

# Create combined dataset with geostrophic and Ekman velocities
combined_ds = xr.Dataset(
    {
        "ug": aviso_al["ugosa"] + mdt_al["u"],  # aviso geostrophic u-component with long-term mean MDT added
        "vg": aviso_al["vgosa"] + mdt_al["v"],  # aviso geostrophic v-component with long-term mean MDT added
        "uek": oscar_al["u"] - oscar_al["ug"],  # OSCAR Ekman u-component
        "vek": oscar_al["v"] - oscar_al["vg"],  # OSCAR Ekman v-component
        "u": aviso_al["ugosa"]
        + mdt_al["u"]
        + oscar_al["u"]
        - oscar_al["ug"],  # total u-component combining geostrophic and Ekman components
        "v": aviso_al["vgosa"]
        + mdt_al["v"]
        + oscar_al["v"]
        - oscar_al["vg"],  # total v-component combining geostrophic and Ekman components
    },
    coords={
        "time": aviso_al.time,
        "latitude": aviso_al.latitude,
        "longitude": aviso_al.longitude,
    },
    attrs={
        "title": "Combined dataset of AVISO geostrophic current anomalies with long-term mean dynamic topography geostrophic currents added and OSCAR Ekman currents",
        "institution": _convert_dict_to_string(
            {
                "aviso": aviso.attrs.get("institution"),
                "mdt": mdt.attrs.get("institution"),
                "oscar": oscar.attrs.get("institution"),
            },
        ),
        "history": _convert_dict_to_string(
            {
                datetime.now(tz=UTC).isoformat(): "Created combined dataset from AVISO, MDT, and OSCAR datasets",
                "aviso": aviso.attrs.get("history"),
                "mdt": mdt.attrs.get("history"),
                "oscar": oscar.attrs.get("history"),
            },
        ),
        "source": _convert_dict_to_string(
            {
                "aviso": aviso.attrs.get("source"),
                "mdt": mdt.attrs.get("source"),
                "oscar": oscar.attrs.get("source"),
            },
        ),
        "references": _convert_dict_to_string(
            {
                "aviso": "https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/description",
                "mdt": "https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/mdt/mdt-global-cnes-cls.html",
                "oscar": "https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_FINAL_V2.0",
            },
        ),
    },
)

combined_ds = combined_ds.chunk({"time": 1, "latitude": -1, "longitude": -1})

combined_ds["ug"].attrs = {
    "units": "m/s",
    "long_name": "zonal geostrophic current",
    "standard_name ": "surface_geostrophic_eastward_sea_water_velocity",
    "description": "Geostrophic current anomaly from AVISO with long-term mean dynamic topography (MDT) added",
}
combined_ds["vg"].attrs = {
    "units": "m/s",
    "long_name": "meridional geostrophic current",
    "standard_name ": "surface_geostrophic_northward_sea_water_velocity",
    "description": "Geostrophic current anomaly from AVISO with long-term mean dynamic topography (MDT) added",
}
combined_ds["uek"].attrs = {
    "units": "m/s",
    "long_name": "zonal Ekman current",
    "standard_name ": "eastward_sea_water_velocity_due_to_ekman_drift",
    "description": "Ekman current from OSCAR calculated as the difference between the total OSCAR u-component and the geostrophic OSCAR u-component",
}
combined_ds["vek"].attrs = {
    "units": "m/s",
    "long_name": "meridional Ekman current",
    "standard_name ": "northward_sea_water_velocity_due_to_ekman_drift",
    "description": "Ekman current from OSCAR calculated as the difference between the total OSCAR v-component and the geostrophic OSCAR v-component",
}
combined_ds["u"].attrs = {
    "units": "m/s",
    "long_name": "total eastward velocity",
    "standard_name ": "eastward_sea_water_velocity",
    "description": "Total eastward velocity combining geostrophic and Ekman components from AVISO, MDT, and OSCAR",
}
combined_ds["v"].attrs = {
    "units": "m/s",
    "long_name": "total northward velocity",
    "standard_name ": "northward_sea_water_velocity",
    "description": "Total northward velocity combining geostrophic and Ekman components from AVISO, MDT, and OSCAR",
}

# Convert all data variables to float32 to save memory
for var in combined_ds.data_vars:
    combined_ds[var] = combined_ds[var].astype("float32")

# Save the combined dataset to a NetCDF file
combined_ds.to_netcdf(SAVE_PATH, mode="w", format="NETCDF4", engine="h5netcdf")
