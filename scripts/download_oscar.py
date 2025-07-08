# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "requests",
#     "python-dotenv",
#     "xarray[io,accel,parallel]",
#     "tqdm",
# ]
# ///


import concurrent.futures
import os
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
    """
    Set up the request library so that it authenticates against the given Earthdata
    url and is able to track cookies between requests.  This uses the .netrc file.
    """
    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")
    if username is None or password is None:
        raise ValueError(
            "Please set EARTHDATA_USERNAME and EARTHDATA_PASSWORD in your .env file."
        )

    # bit of a hack to handle Earthdata authentication but the only one I could get working
    # from the last example shown at https://urs.earthdata.nasa.gov/documentation/for_users/data_access/python
    with requests.Session() as session:
        # this will redirect to the login page if not authenticated
        auth_response = session.get(url)
        if auth_response.status_code != 200:
            raise requests.HTTPError(
                f"Failed to access {url}: {auth_response.status_code}"
            )
        # use response URL with auth to get the final URL after login redirection
        response = session.get(
            auth_response.url, stream=True, auth=(username, password)
        )
    return response


def write_stream_to_file(file_path: Path, response: requests.Response) -> None:
    """
    Write a binary or string response stream to a file, ensuring the directory exists.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if isinstance(chunk, str):
                    encoded_chunk = chunk.encode("utf-8")
                else:
                    encoded_chunk = chunk
                file.write(encoded_chunk)
    except Exception as e:
        raise RuntimeError(f"Failed to write to {file_path}: {e}")


def subset_and_save_dataset(file_path: Path) -> None:
    """
    Subset the dataset to the desired lat/lon range and save it as a NetCDF file.
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
            ds_subset.to_netcdf(
                file_path.with_suffix(".nc"), mode="w", format="NETCDF4"
            )
    except Exception as e:
        raise RuntimeError(f"Failed to subset and save dataset {file_path}: {e}")


def remove_temp_file(file_path: Path) -> None:
    """
    Remove the temporary file created during the download process.
    """
    try:
        file_path.unlink()
    except Exception as e:
        raise RuntimeError(f"Failed to remove temporary file {file_path}: {e}")


def download_file(url: str, local_path: Path, local_filename: Path) -> None:
    if not url_validator(url):
        print(f"Invalid URI: {url}")
        return
    response = access_earthdata_url(url)
    if response.ok:
        try:
            write_stream_to_file(
                local_path / local_filename.with_suffix(".temp"), response
            )
            subset_and_save_dataset(local_path / local_filename.with_suffix(".temp"))
            remove_temp_file(local_path / local_filename.with_suffix(".temp"))
        except Exception as e:
            print(f"Error processing {local_filename}: {e}")
    else:
        print(f"Failed to download {url}: {response.status_code}")


def url_validator(url) -> bool:
    """
    Validate if the given URL is a well-formed URI.
    """
    try:
        parsed = parse.urlparse(url)
        # Check if the URL has a valid scheme and netloc
        if parsed.scheme.lower() not in ["http", "https"]:
            print(f"Invalid scheme in URL: {url}")
            return False
        if not parsed.netloc:
            print(f"Invalid netloc in URL: {url}")
            return False
        return all([parsed.scheme, parsed.netloc])
    except Exception as e:
        print(f"Failed to validate URI {url}: {e}")
        return False


# Read the list of URLs from the text file and download each file
with open("D:/nccs-transport/oscar/4185382644-Harmony.txt") as f:
    lines = f.readlines()
    # Remove empty lines and comments
    lines = [
        line.strip() for line in lines if line.strip() and not line.startswith("#")
    ]
    # Extract file names from the URLs
    file_names = [
        Path(line.split("/")[-1]) for line in lines if line.startswith("https://")
    ]
    # Use a thread pool to download files concurrently for IO-bound workload
    with tqdm(total=len(lines), desc="Downloading files") as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit download tasks to the executor
            futures = {
                executor.submit(
                    download_file, line, Path("D:/nccs-transport/oscar"), name
                ): name
                for line, name in zip(lines, file_names)
            }
            results = {}
            # Process the results as they complete
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    future.result()  # Wait for the download to complete
                    results[name] = "Success"
                except Exception as e:
                    results[name] = f"Failed: {e}"
                # Manually update the progress bar
                pbar.update(1)
