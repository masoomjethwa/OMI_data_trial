#!/usr/bin/env python3
# @author: mp10
# @coding assistant: [TGC-DD26092025]
# pip install h5py numpy matplotlib basemap

"""
Module: read_and_map_omi_no2_so2.py

Purpose:
    Extract NO2 or SO2 data from OMI HDF5 files listed in a text file and optionally create a map of the data.

Disclaimer:
    The code is for demonstration purposes only. Users are responsible for checking accuracy and adapting as needed.

Author: Justin Roberts-Pierel, 2015 (original)
Refactored by: mp10, 2025
"""

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


def read_file_list(file_list_path: Path) -> list[Path]:
    """
    Read file names from a text file.

    Args:
        file_list_path (Path): Path to the fileList.txt containing HDF5 filenames.

    Returns:
        list[Path]: List of file paths.
    """
    if not file_list_path.exists():
        print(f"File list not found: {file_list_path}")
        exit(1)

    with file_list_path.open(encoding="utf-8") as f:
        return [Path(line.strip()) for line in f if line.strip()]


def extract_omi_data(file_path: Path) -> Optional[dict]:
    """
    Extract NO2 or SO2 data, geolocation, and metadata from an OMI HDF5 file.

    Args:
        file_path (Path): Path to the HDF5 file.

    Returns:
        Optional[dict]: Dictionary containing data arrays and metadata, or None if invalid file.
    """
    try:
        with h5py.File(file_path, "r") as file:
            file_name = file_path.name
            if "NO2" in file_name:
                print("This is an OMI NO2 file. Here is some information:")
                data_fields = file["HDFEOS"]["SWATHS"]["ColumnAmountNO2"]["Data Fields"]
                geolocation = file["HDFEOS"]["SWATHS"]["ColumnAmountNO2"]["Geolocation Fields"]
                sds_name = "ColumnAmountNO2"
                data = data_fields[sds_name]
                map_label = data.attrs["Units"].decode()
                valid_min, valid_max = None, None
            elif "SO2" in file_name:
                print("This is an OMI SO2 file. Here is some information:")
                data_fields = file["HDFEOS"]["SWATHS"]["OMI Total Column Amount SO2"]["Data Fields"]
                geolocation = file["HDFEOS"]["SWATHS"]["OMI Total Column Amount SO2"]["Geolocation Fields"]
                sds_name = "ColumnAmountSO2_PBL"
                data = data_fields[sds_name]
                valid_min, valid_max = data.attrs["ValidRange"]
                map_label = data.attrs["Units"].decode()
                print(f"Valid Range is: {valid_min} to {valid_max}")
            else:
                print(f"The file named: {file_name} is not a valid OMI file.\n")
                return None

            # Retrieve fill and missing values, offset and scale factors
            fill_value = data.attrs.get("_FillValue", None)
            missing_value = data.attrs.get("MissingValue", None)
            offset = data.attrs.get("Offset", 0.0)
            scale_factor = data.attrs.get("ScaleFactor", 1.0)

            # Latitude and longitude arrays
            latitudes = geolocation["Latitude"][:]
            longitudes = geolocation["Longitude"][:]

            # Raw data array with fill/missing values masked as NaN
            data_array = data[:].astype(float)
            if fill_value is not None:
                data_array[data_array == fill_value] = np.nan
            if missing_value is not None:
                data_array[data_array == missing_value] = np.nan

            # Apply scale and offset
            data_array = scale_factor * (data_array - offset)

            return {
                "data": data_array,
                "latitudes": latitudes,
                "longitudes": longitudes,
                "map_label": map_label,
                "sds_name": sds_name,
                "file_name": file_name,
                "valid_range": (valid_min, valid_max),
            }

    except (OSError, KeyError) as e:
        print(f"Error reading file {file_path.name}: {e}")
        return None


def print_statistics(data_array: np.ndarray) -> None:
    """
    Compute and print basic statistics of the data array.

    Args:
        data_array (np.ndarray): Data array with NaNs representing missing data.
    """
    average = np.nanmean(data_array)
    stdev = np.nanstd(data_array)
    median = np.nanmedian(data_array)
    print(
        f"The average of this data is: {average:.3f}\n"
        f"The standard deviation is: {stdev:.3f}\n"
        f"The median is: {median:.3f}"
    )


def plot_omi_data(
    data_array: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    map_label: str,
    file_name: str,
    sds_name: str,
) -> None:
    """
    Plot OMI data on a map using Basemap.

    Args:
        data_array (np.ndarray): The data values (masked array).
        latitudes (np.ndarray): Latitude values.
        longitudes (np.ndarray): Longitude values.
        map_label (str): Label for the colorbar (e.g., units).
        file_name (str): The name of the file for the plot title.
        sds_name (str): The SDS dataset name for the plot title.
    """
    masked_data = np.ma.masked_invalid(data_array)

    m = Basemap(
        projection="cyl",
        resolution="l",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=-180,
        urcrnrlon=180,
    )
    m.drawcoastlines(linewidth=0.5)
    m.drawparallels(np.arange(-90, 120, 30), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(-180, 180, 45), labels=[0, 0, 0, 1])

    cmap = plt.cm.get_cmap("gist_stern_r")
    cmap.set_under("w")

    vmin = 0
    vmax = np.nanmax(data_array) * 0.35 if np.nanmax(data_array) > 0 else 1

    mesh = m.pcolormesh(
        longitudes, latitudes, masked_data, latlon=True, vmin=vmin, vmax=vmax, cmap=cmap
    )
    cb = m.colorbar(mesh, location="right", pad="5%")
    cb.set_label(map_label)

    plt.title(f"{file_name}\n{sds_name}")
    plt.autoscale()

    plt.show()


def main() -> None:
    """
    Main function to read file list, process files, show stats, and optionally plot maps.
    """
    file_list_path = Path("path/to/fileList.txt")
    file_paths = read_file_list(file_list_path)

    for file_path in file_paths:
        user_choice = input(f"\nWould you like to process\n{file_path}\n\n(Y/N): ").strip().lower()
        if user_choice == "n":
            print("Skipping...")
            continue

        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        result = extract_omi_data(file_path)
        if result is None:
            continue

        latitudes = result["latitudes"]
        longitudes = result["longitudes"]
        data_array = result["data"]
        map_label = result["map_label"]
        sds_name = result["sds_name"]
        file_name = result["file_name"]

        min_lat, max_lat = np.min(latitudes), np.max(latitudes)
        min_lon, max_lon = np.min(longitudes), np.max(longitudes)

        print_statistics(data_array)

        print(
            f"The range of latitude in this file is: {min_lat} to {max_lat} degrees\n"
            f"The range of longitude in this file is: {min_lon} to {max_lon} degrees"
        )

        create_map = input("\nWould you like to create a map of this data? (Y/N): ").strip().lower()
        if create_map == "y":
            plot_omi_data(data_array, latitudes, longitudes, map_label, file_name, sds_name)

            save_map = input("\nWould you like to save this map? (Y/N): ").strip().lower()
            if save_map == "y":
                output_filename = file_path.with_suffix(".png")
                plt.savefig(output_filename)
                print(f"Map saved as {output_filename}")


if __name__ == "__main__":
    main()
