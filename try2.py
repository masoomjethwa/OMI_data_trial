# -*- coding: utf-8 -*-
"""
Example script to access and visualize GES DISC OMI v3 Swath HDF-EOS5 ozone profile data.

Author: mp10

Usage:
    Place the HDF5 file in your working directory and run:
        python omi_omo3pr_plot.py

Tested with: Python 3.7+, h5py, numpy, matplotlib, Basemap
"""

import os
import datetime
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.ticker import FormatStrFormatter

FILE_NAME = 'OMI-Aura_L2-OMO3PR_2017m1018t0103-o70523_v003-2017m1019t111518.he5'
PATH = '/HDFEOS/SWATHS/O3Profile'

def plot_omi_ozone_profile(file_name: str, path: str, tdim: int = 26, track: int = 0) -> None:
    """
    Load and plot OMI ozone vertical profile data at a given time and track.

    Args:
        file_name (str): Path to the HDF5 file.
        path (str): Group path inside the HDF5 file to O3Profile.
        tdim (int): Time dimension index (default: 26).
        track (int): Track index along cross track dimension (default: 0).
    """
    with h5py.File(file_name, mode='r') as f:
        # Read ozone profile data at the specified time and track
        ozone_path = f"{path}/Data Fields/O3"
        ozone_data = f[ozone_path][tdim, track, :]
        attrs = f[ozone_path].attrs
        missing_value = attrs['MissingValue']
        fill_value = attrs['_FillValue']
        title = attrs['Title'].decode()
        units = attrs['Units'].decode()

        # Read pressure data (bounds), slice to match ozone layers (18)
        pressure_path = f"{path}/Geolocation Fields/Pressure"
        pressure_all = f[pressure_path][tdim, track, :]
        # Take only the 18 layers (skip surface pressure for plotting)
        pressure = pressure_all[1:]
        pres_units = f[pressure_path].attrs['Units'].decode()
        pres_fill_value = f[pressure_path].attrs['_FillValue']

        # Read time array for timestamp
        time_path = f"{path}/Geolocation Fields/Time"
        time_arr = f[time_path][:]

        # Replace fill/missing values with nan for plotting
        ozone_data = np.where((ozone_data == missing_value) | (ozone_data == fill_value), np.nan, ozone_data)
        pressure = np.where(pressure == pres_fill_value, np.nan, pressure)

        # Mask nan values for cleaner plotting
        ozone_masked = np.ma.masked_invalid(ozone_data)
        pressure_masked = np.ma.masked_invalid(pressure)

        # Plot ozone profile vs pressure
        plt.figure(figsize=(8, 6))
        plt.plot(ozone_masked, pressure_masked, marker='o')
        plt.xlabel(f"{title} ({units})")
        plt.ylabel(f"Pressure ({pres_units})")
        plt.title(f"{os.path.basename(file_name)}\n"
                  f"{title} at Time = {get_time_string(time_arr[tdim])} (track={track})")
        plt.gca().invert_yaxis()  # High pressure at bottom
        plt.yscale('log')         # Log scale for pressure axis
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d'))
        plt.grid(True)
        plt.tight_layout()

        # Save figure as PNG in the current directory
        png_filename = f"{os.path.basename(file_name)}.png"
        plt.savefig(png_filename)
        plt.show()


def get_time_string(seconds_since_1993: int) -> str:
    """
    Convert seconds since Jan 1, 1993 UTC to formatted datetime string.

    Args:
        seconds_since_1993 (int): Seconds offset from 1993-01-01 00:00:00 UTC.

    Returns:
        str: Formatted datetime string.
    """
    base_time = datetime.datetime(1993, 1, 1, 0, 0, 0)
    time_value = base_time + datetime.timedelta(seconds=seconds_since_1993)
    return time_value.strftime('%Y-%m-%d %H:%M:%S')


if __name__ == "__main__":
    plot_omi_ozone_profile(FILE_NAME, PATH)
