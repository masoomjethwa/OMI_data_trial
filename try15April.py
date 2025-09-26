#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: masoom
@coding assistant: TGC-DD26092025
# pip install h5py cartopy matplotlib numpy
"""

from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature


def plot_omi_o3_data(file_path: Path) -> None:
    """
    Reads OMI Total Column O3 data from an HDF5 file and plots it on a Mercator projection map.

    Args:
        file_path (Path): Path to the OMI .he5 file
    """

    with h5py.File(file_path, 'r') as f:
        # Extract geolocation and O3 data arrays
        lat = f['HDFEOS']['GRIDS']['OMI Total Column O3']['Geolocation Fields']['Latitude'][:]
        lon = f['HDFEOS']['GRIDS']['OMI Total Column O3']['Geolocation Fields']['Longitude'][:]
        o3 = f['HDFEOS']['GRIDS']['OMI Total Column O3']['Data Fields']['ColumnAmountO3'][:]

    # Mask invalid or fill values (assuming NaNs represent invalid)
    o3_masked = np.ma.masked_invalid(o3)

    # Set up plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=crs.Mercator())

    # Set extent to data bounds with some padding
    min_lon, max_lon = np.nanmin(lon), np.nanmax(lon)
    min_lat, max_lat = np.nanmin(lat), np.nanmax(lat)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=crs.PlateCarree())

    # Add map features
    ax.coastlines(resolution='50m', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Plot O3 data
    pcm = ax.pcolormesh(
        lon,
        lat,
        o3_masked,
        cmap='viridis',
        shading='auto',
        transform=crs.PlateCarree()
    )

    # Add colorbar with label
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label('OMI Total Column O3 (Dobson Units)')

    # Title
    ax.set_title(f'OMI Total Column O3\n{file_path.name}')

    plt.show()


def main():
    # Define file path placeholder - update this to your local file location
    omi_file = Path('OMI-Aura_L2-OMTO3_2021m0401t0712-o88890_v003-2021m0401t142502.he5')

    if not omi_file.is_file():
        print(f"ERROR: File not found: {omi_file}")
        return

    plot_omi_o3_data(omi_file)


if __name__ == '__main__':
    main()
