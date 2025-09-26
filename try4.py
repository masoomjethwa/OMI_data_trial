#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: mp10
@coding assistant: TGC-DD26092025
# pip install numpy matplotlib cartopy
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def plot_random_points_basic() -> None:
    """
    Plot random points on a basic scatter plot without geographic projection.
    """
    np.random.seed(1)
    x = 360 * np.random.rand(100)
    y = 180 * np.random.rand(100) - 90

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Random Points (No Projection)')
    plt.show()


def plot_random_points_azimuthal_equidistant(to_lon: float, to_lat: float) -> None:
    """
    Plot random points on a map with Azimuthal Equidistant projection centered at (to_lon, to_lat).

    Args:
        to_lon (float): Central longitude for projection
        to_lat (float): Central latitude for projection
    """
    np.random.seed(1)
    lon = 360 * np.random.rand(100)
    lat = 180 * np.random.rand(100) - 90

    # Plot without transform argument (points not projected correctly)
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(1, 1, 1,
                           projection=ccrs.AzimuthalEquidistant(central_longitude=to_lon,
                                                                central_latitude=to_lat))
    ax1.scatter(lon, lat)
    ax1.stock_img()
    ax1.coastlines()
    ax1.gridlines(draw_labels=True)
    ax1.set_title('Azimuthal Equidistant Projection (No Transform)')
    plt.show()

    # Plot with Geodetic transform - points correctly placed on map
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(1, 1, 1,
                           projection=ccrs.AzimuthalEquidistant(central_longitude=to_lon,
                                                                central_latitude=to_lat))
    ax2.scatter(lon, lat, transform=ccrs.Geodetic())
    ax2.stock_img()
    ax2.coastlines()
    ax2.gridlines(draw_labels=True)
    ax2.set_title('Azimuthal Equidistant Projection (With Geodetic Transform)')
    plt.show()


def main() -> None:
    """
    Run example plots.
    """
    plot_random_points_basic()

    # Coordinates for Toronto, Canada
    toronto_lon = -79.398329
    toronto_lat = 43.660924
    plot_random_points_azimuthal_equidistant(toronto_lon, toronto_lat)


if __name__ == "__main__":
    main()
