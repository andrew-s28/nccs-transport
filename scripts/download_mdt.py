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

# urllib.request.urlretrieve("ftp://ftp-access.aviso.altimetry.fr/auxiliary/mdt", "D:/nccs-transport/mdt/mdt.nc")

load_dotenv()
username = os.getenv("AVISO_USERNAME")
password = os.getenv("AVISO_PASSWORD")
if username is None or password is None:
    raise ValueError("Please set AVISO_USERNAME and AVISO_PASSWORD in your .env file.")

host = "ftp-access.aviso.altimetry.fr"  # hard-coded
port = 2221
transport = paramiko.Transport((host, port))
transport.connect(username=username, password=password)
sftp = paramiko.SFTPClient.from_transport(transport)
if sftp is None:
    raise ValueError("Failed to connect to SFTP server.")
sftp.get(
    "auxiliary/mdt/mdt_cnes_cls2022_global/mdt_cnes_cls22_global.nc",
    "D:/nccs-transport/mdt/mdt_cnes_cls22_global.nc",
)
sftp.close()
transport.close()
