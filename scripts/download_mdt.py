# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "paramiko",
#     "python-dotenv",
# ]
# ///
import os

import paramiko
from dotenv import load_dotenv

load_dotenv()
username = os.getenv("AVISO_USERNAME")
password = os.getenv("AVISO_PASSWORD")
if username is None or password is None:
    msg = "AVISO_USERNAME and AVISO_PASSWORD must be set in the environment."
    raise ValueError(msg)

# Connect to the SFTP server and download the MDT file
# Implementation based on example from https://stackoverflow.com/questions/432385/sftp-in-python-platform-independent
host = "ftp-access.aviso.altimetry.fr"
port = 2221
transport = paramiko.Transport((host, port))
transport.connect(username=username, password=password)
sftp = paramiko.SFTPClient.from_transport(transport)
if sftp is None:
    msg = "Failed to connect to SFTP server."
    raise ValueError(msg)
sftp.get(
    "auxiliary/mdt/mdt_cnes_cls2022_global/mdt_cnes_cls22_global.nc",
    "D:/nccs-transport/mdt/mdt_cnes_cls22_global.nc",
)
sftp.close()
transport.close()
