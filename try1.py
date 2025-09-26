# -*- coding: utf-8 -*-
"""
@author: masoom

This script reads ozone profile data from an OMI HDF-EOS5 file and plots the
ozone vertical profile for a specific time and cross-track position.
"""

import os
import h5py
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

FILE_NAME = 'OMI-Aura_L2-OMTO3_2021m0401t0533-o88889_v003-2021m0401t105813.he5'
PATH = '/HDFEOS/SWATHS/O3Profile'

# Time dimension size = 329
# Lat/Lon dimensions = 329x30
# O3 dimension = 329x30x18
# Pressure dimension = 329x30x19
# 30 is the cross-track dimension
# Pressure has 19 points because it includes layer bounds (18 layers + bounds)

# Parameters to subset data
tdim = 26      # Time index (0 to 328)
track = 0      # Cross-track index (0 to 29)

with h5py.File(FILE_NAME, 'r') as f:
    # Access ozone data for given time and track
    varname_o3 = PATH + '/Data Fields/O3'
    ozone_data = f[varname_o3][tdim, track, :]
    
    # Read attributes for metadata
    attrs = f[varname_o3].attrs
    missing_value = attrs['MissingValue']
    fill_value = attrs['_FillValue']
    title = attrs['Title'].decode('utf-8')
    units = attrs['Units'].decode('utf-8')
    
    # Access pressure data (bounds)
    varname_pressure = PATH + '/Geolocation Fields/Pressure'
    pressure_all = f[varname_pressure][tdim, track, :]
    
    # Pressure units and fill value
    pres_units = f[varname_pressure].attrs['Units'].decode('utf-8')
    pres_fill_value = f[varname_pressure].attrs['_FillValue']
    
    # Select pressure corresponding to the ozone layers (skip surface pressure)
    pressure = pressure_all[1:]
    
    # Access time data
    varname_time = PATH + '/Geolocation Fields/Time'
    time = f[varname_time][:]
    
    # Replace fill and missing values with np.nan for ozone
    ozone_data = np.where((ozone_data == missing_value) | (ozone_data == fill_value), np.nan, ozone_data)
    ozone_masked = np.ma.masked_invalid(ozone_data)
    
    # Replace fill values with np.nan for pressure
    pressure = np.where(pressure == pres_fill_value, np.nan, pressure)
    pressure_masked = np.ma.masked_invalid(pressure)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(ozone_masked, pressure_masked, marker='o')
    plt.xlabel(f'{title} ({units})')
    plt.ylabel(f'Pressure ({pres_units})')
    
    # Convert time to human-readable format
    base_time = datetime.datetime(1993, 1, 1)
    timedatum = (base_time + datetime.timedelta(seconds=time[tdim])).strftime('%Y-%m-%d %H:%M:%S')
    
    # Title including file basename, variable, time, and track info
    basename = os.path.basename(FILE_NAME)
    plt.title(f'{basename}\n{title} at Time = {timedatum} (track={track})', fontsize=11)
    
    # Invert y-axis so pressure decreases upward
    plt.gca().invert_yaxis()
    # Use log scale for pressure
    plt.gca().set_yscale('log')
    # Format y-axis ticks as integers
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.grid(True)
    plt.tight_layout()
    
    # Save figure
    pngfile = f"{basename}.py.png"
    plt.savefig(pngfile)
    plt.show()
