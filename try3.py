    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: masoom
@coding assistant: TGC-DD26092025
# pip install numpy matplotlib cartopy
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def plot_random_points_basic_scatter(x: np.ndarray, y: np.ndarray) -> None:
    """
    Plot random points on a simple scatter plot without geographic projection.

    Args:
        x (np.ndarray): Array of x coordinates (longitude).
        y (np.ndarray): Array of y coordinates (latitude).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Random Points (No Projection)")
    plt.show()


def plot_random_points_plate_carree(lon: np.ndarray, lat: np.ndarray) -> None:
    """
    Plot random points on a Plate Carrée map projection in increasing detail.

    Args:
        lon (np.ndarray): Array of longitude values.
        lat (np.ndarray): Array of latitude values.
    """
    # Basic scatter on PlateCarree projection
    fig1, ax1 = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(8, 6))
    ax1.scatter(lon, lat)
    ax1.set_title("Scatter on PlateCarree Projection (No coastlines)")
    plt.show()

    # Add coastlines
    fig2, ax2 = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(8, 6))
    ax2.scatter(lon, lat)
    ax2.coastlines()
    ax2.set_title("Scatter on PlateCarree Projection with Coastlines")
    plt.show()

    # Add stock image and coastlines
    fig3, ax3 = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(8, 6))
    ax3.scatter(lon, lat)
    ax3.stock_img()
    ax3.coastlines()
    ax3.set_title("Scatter on PlateCarree Projection with Stock Image and Coastlines")
    plt.show()


def main() -> None:
    """
    Main function to generate random points and plot them.
    """
    np.random.seed(1)
    x = 360 * np.random.rand(100)
    y = 180 * np.random.rand(100) - 90

    plot_random_points_basic_scatter(x, y)
    plot_random_points_plate_carree(x, y)


if __name__ == "__main__":
    main()
