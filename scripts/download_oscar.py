# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "requests",
#     "python-dotenv",
#     "xarray[io,accel,parallel]",
#     "tqdm",
# ]
# ///

# This script is an absolute pain to run.
# I'm not sure if it's NASA servers rate limiting or OSU wifi,
# but it requires several restarts to complete the download.
# Tread carefully.


import concurrent.futures
import os
from http import HTTPStatus
from pathlib import Path
from urllib import parse

import requests
import xarray as xr
from dotenv import load_dotenv
from tqdm import tqdm

# Constants for the lat/lon range to subset the dataset
MIN_LONGITUDE = 180
MAX_LONGITUDE = 280
MIN_LATITUDE = 10
MAX_LATITUDE = 60

load_dotenv()


def access_earthdata_url(url: str) -> requests.Response:
    """Make an authenticated request to an Earthdata URL.

    Args:
        url (str): The Earthdata URL to access.

    Returns:
        requests.Response: The response object containing the data from the URL.

    Raises:
        ValueError: If EARTHDATA_USERNAME or EARTHDATA_PASSWORD is not set in the environment.
        requests.HTTPError: If the request to the URL fails or is not authenticated.

    """
    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")
    if username is None or password is None:
        msg = "EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set in the environment."
        raise ValueError(msg)

    # bit of a hack to handle Earthdata authentication but the only one I could get working
    # from the last example shown at https://urs.earthdata.nasa.gov/documentation/for_users/data_access/python
    with requests.Session() as session:
        # this will redirect to the login page if not authenticated
        auth_response = session.get(url)
        if auth_response.status_code != HTTPStatus.OK:
            msg = f"Failed to access {url}: {auth_response.status_code}"
            raise requests.HTTPError(msg)
        # use response URL with auth to get the final URL after login redirection
        response = session.get(auth_response.url, stream=True, auth=(username, password))
    return response


def write_stream_to_file(file_path: Path, response: requests.Response) -> None:
    """Write a binary or string response stream to a file, ensuring the directory exists.

    Args:
        file_path (Path): The path to the file where the response will be written.
        response (requests.Response): The response object containing the data to write.

    Raises:
        RuntimeError: If the file cannot be written or the directory cannot be created.

    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with Path(file_path).open("wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                encoded_chunk = chunk.encode("utf-8") if isinstance(chunk, str) else chunk
                file.write(encoded_chunk)
    except Exception as e:
        msg = f"Failed to write to {file_path}"
        raise RuntimeError(msg) from e


def subset_and_save_dataset(file_path: Path) -> None:
    """Subset the dataset to the desired lat/lon range and save it as a NetCDF file.

    Args:
        file_path (Path): The path to the dataset file to subset and save.

    Raises:
        RuntimeError: If the dataset cannot be opened, subsetted, or saved.

    """
    try:
        with xr.open_dataset(file_path) as ds:
            ds_subset = ds.where(
                (ds["lat"] >= MIN_LATITUDE) & (ds["lat"] <= MAX_LATITUDE),
                drop=True,
            ).where(
                (ds["lon"] >= MIN_LONGITUDE) & (ds["lon"] <= MAX_LONGITUDE),
                drop=True,
            )
            ds_subset.to_netcdf(file_path.with_suffix(".nc"), mode="w", format="NETCDF4")
    except Exception as e:
        msg = f"Failed to subset and save dataset {file_path}"
        raise RuntimeError(msg) from e


def remove_temp_file(file_path: Path) -> None:
    """Remove the temporary file created during the download process.

    Args:
        file_path (Path): The path to the temporary file to remove.

    Raises:
        RuntimeError: If the file cannot be removed.

    """
    try:
        file_path.unlink()
    except Exception as e:
        msg = f"Failed to remove temporary file {file_path}"
        raise RuntimeError(msg) from e


def download_file(url: str, local_path: Path, local_filename: Path) -> None:
    """Download a file from a URL and save it to a local path."""
    if Path.joinpath(local_path, local_filename.with_suffix(".nc")).exists():
        # File already exists, skip download
        return
    if not url_validator(url):
        print(f"Invalid URI: {url}")
        return
    response = access_earthdata_url(url)
    if response.ok:
        try:
            write_stream_to_file(local_path / local_filename.with_suffix(".temp"), response)
            subset_and_save_dataset(local_path / local_filename.with_suffix(".temp"))
            remove_temp_file(local_path / local_filename.with_suffix(".temp"))
        except Exception as e:  # noqa: BLE001
            print(f"Error processing {local_filename}: {e}")
    else:
        print(f"Failed to download {url}: {response.status_code}")


def url_validator(url) -> bool:
    """Validate if the given URL is a well-formed URI.

    Args:
        url (str): The URL to validate.

    Returns:
        bool: True if the URL is valid, False otherwise.

    """
    try:
        parsed = parse.urlparse(url)
        # Check if the URL has a valid scheme and netloc
        if parsed.scheme.lower() not in {"http", "https"}:
            print(f"Invalid scheme in URL: {url}")
            return False
        if not parsed.netloc:
            print(f"Invalid netloc in URL: {url}")
            return False
        return all([parsed.scheme, parsed.netloc])
    except Exception as e:  # noqa: BLE001
        print(f"Failed to validate URI {url}: {e}")
        return False


def concatenate_datasets(file_path: Path) -> None:
    """Concatenate multiple NetCDF files into a single dataset.

    Args:
        file_path (Path): The path to the directory containing the NetCDF files to concatenate.

    Raises:
        RuntimeError: If the concatenation fails.

    """
    try:
        ds = xr.open_mfdataset(list(file_path.glob("*.nc")), combine="by_coords")
        ds.to_netcdf(file_path / "oscar.nc", mode="w", format="NETCDF4")
    except Exception as e:
        msg = f"Failed to concatenate datasets in {file_path}"
        raise RuntimeError(msg) from e


# Read the list of URLs from the text file and download each file
with Path("D:/nccs-transport/oscar/4185382644-Harmony.txt").open(encoding="utf-8") as f:
    lines = f.readlines()
    # Remove empty lines and comments
    lines = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    # Extract file names from the URLs
    file_names = [Path(line.split("/")[-1]) for line in lines if line.startswith("https://")]
    # Use a thread pool to download files concurrently for IO-bound workload
    with tqdm(total=len(lines), desc="Downloading files") as pbar, concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit download tasks to the executor
        futures = {
            executor.submit(download_file, line, Path("D:/nccs-transport/oscar"), name): name
            for line, name in zip(lines, file_names, strict=True)
        }
        results = {}
        # Process the results as they complete
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                future.result()  # Wait for the download to complete
                results[name] = "Success"
            except Exception as e:  # noqa: BLE001
                results[name] = f"Failed: {e}"
            # Manually update the progress bar
            pbar.update(1)

# Concatenate the downloaded NetCDF files into a single dataset
concatenate_datasets(Path("D:/nccs-transport/oscar"))
