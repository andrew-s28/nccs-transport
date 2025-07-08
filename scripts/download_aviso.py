# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "copernicusmarine",
#     "python-dotenv",
# ]
# ///
import os

import copernicusmarine
from dotenv import load_dotenv

load_dotenv()

copernicusmarine.login(
    username=os.getenv("COPERNICUSMARINE_SERVICE_USERNAME"),
    password=os.getenv("COPERNICUSMARINE_SERVICE_PASSWORD"),
)

copernicusmarine.subset(
    dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
    variables=["sla", "ugosa", "vgosa"],
    minimum_longitude=-160,
    maximum_longitude=-80,
    minimum_latitude=10,
    maximum_latitude=60,
    start_datetime="1993-01-19T00:00:00",
    end_datetime="2024-11-19T00:00:00",
    output_directory="D:/nccs-transport/aviso",
    output_filename="aviso.nc",
)
