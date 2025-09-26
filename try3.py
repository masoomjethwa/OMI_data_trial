#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract SO₂, Latitude, and Longitude from OMI HDF5 files and aggregate into a DataFrame.

@author: mp10
@coding assistant: [TGC-26092025]
"""

# pip install numpy pandas h5py

from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import h5py

# Constants
#DATA_DIR = Path("path/to/SO2/files")  # 🔒 Replace with actual directory path
DATA_DIR = Path("D:/python learning/Satellite Data/SO2")
FILE_PATTERN = "OMI*.he5"

# HDF5 Keys
SO2_KEY = "HDFEOS/GRIDS/OMI Total Column Amount SO2/Data Fields/ColumnAmountSO2"
LAT_KEY = "HDFEOS/GRIDS/OMI Total Column Amount SO2/Data Fields/Latitude"
LON_KEY = "HDFEOS/GRIDS/OMI Total Column Amount SO2/Data Fields/Longitude"


def extract_values(file_path: Path) -> pd.DataFrame:
    """
    Extract SO2, latitude, and longitude data from an HDF5 file.

    Args:
        file_path (Path): Path to the .he5 file.

    Returns:
        pd.DataFrame: Flattened DataFrame with columns ['Lat', 'Long', 'SO2']
    """
    try:
        with h5py.File(file_path, "r") as file:
            so2_data = file[SO2_KEY][:]
            lat_data = file[LAT_KEY][:]
            lon_data = file[LON_KEY][:]

            # Flatten arrays (assuming 3D arrays, take middle time slice if needed)
            if so2_data.ndim == 3:
                so2_data = so2_data[1, :, :]  # use the second time slice
            so2_data = np.squeeze(so2_data)
            lat_data = np.squeeze(lat_data)
            lon_data = np.squeeze(lon_data)

            # Filter invalid values
            so2_data = np.where(so2_data < 0, np.nan, so2_data)

            # Ensure shape compatibility
            if so2_data.shape != lat_data.shape or so2_data.shape != lon_data.shape:
                raise ValueError("Shape mismatch between SO2, latitude, and longitude arrays.")

            flat_df = pd.DataFrame({
                "Lat": lat_data.flatten(),
                "Long": lon_data.flatten(),
                "SO2": so2_data.flatten()
            })

            return flat_df

    except Exception as e:
        print(f"Error processing file {file_path.name}: {e}")
        return pd.DataFrame(columns=["Lat", "Long", "SO2"])


def main() -> None:
    """
    Main function to aggregate all SO2 datasets into a single DataFrame.
    """
    print("Scanning for OMI HE5 files...")

    all_files = list(DATA_DIR.glob(FILE_PATTERN))
    if not all_files:
        print(f"No files found in: {DATA_DIR}")
        return

    combined_df_list: List[pd.DataFrame] = []

    for file_path in all_files:
        print(f"Processing: {file_path.name}")
        df = extract_values(file_path)
        if not df.empty:
            combined_df_list.append(df)

    if combined_df_list:
        full_df = pd.concat(combined_df_list, ignore_index=True)
        print("Final DataFrame preview:")
        print(full_df.head())

        # Optionally save the result
        output_file = DATA_DIR / "combined_so2_data.csv"
        full_df.to_csv(output_file, index=False)
        print(f"Saved combined data to: {output_file.resolve()}")
    else:
        print("No valid data was extracted.")


if __name__ == "__main__":
    main()
