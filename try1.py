#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize SO₂ with NASA OMI Data

Data Source: https://towardsdatascience.com/visualize-the-invisible-so2-with-nasa-data-and-python-2619f8ed4ea1

@author: mp10
@coding assistant: [TGC-DD26092025]
"""

# pip install numpy pandas matplotlib seaborn h5py folium pillow

from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import h5py
import folium
from folium.plugins import HeatMap

# === Constants ===
DATA_DIR = Path(".")  # 🔁 Replace with your actual path, e.g., Path("/Users/user/SO2_Data")
SO2_KEY = 'HDFEOS/GRIDS/OMI Total Column Amount SO2/Data Fields/ColumnAmountSO2'
LAT_KEY = 'HDFEOS/GRIDS/OMI Total Column Amount SO2/Data Fields/Latitude'
LON_KEY = 'HDFEOS/GRIDS/OMI Total Column Amount SO2/Data Fields/Longitude'
FILL_VALUE_THRESHOLD = -1e30  # Used by NASA as missing data flag


def load_hdf5_files(directory: Path, extension: str = "*.he5") -> List[h5py.File]:
    """
    Load all readable HDF5 files from the specified directory.
    """
    datasets: List[h5py.File] = []

    for file in directory.glob(extension):
        try:
            datasets.append(h5py.File(file, "r"))
        except (OSError, IOError) as e:
            print(f"[⚠️] Could not read file: {file.name} — {e}")

    return datasets


def extract_dataset_field(dataset: List[h5py.File], key: str) -> List[float]:
    """
    Extract and flatten data from a specific dataset key across all HDF5 files.
    """
    extracted: List[float] = []

    for h5_file in dataset:
        try:
            data = h5_file[key][()]  # Load full numpy array
            if data.ndim == 3:
                data = data[1, :, :]  # Use second time slice if present
            flat_data = data.flatten().tolist()

            # Replace fill values with NaN
            cleaned = [np.nan if val < FILL_VALUE_THRESHOLD or np.isnan(val) else float(val)
                       for val in flat_data]

            extracted.extend(cleaned)

        except KeyError:
            print(f"[⚠️] Key not found in file: {key}")
        except Exception as e:
            print(f"[⚠️] Error reading key '{key}': {e}")

    return extracted


def create_so2_dataframe(lat: List[float], lon: List[float], so2: List[float]) -> pd.DataFrame:
    """
    Create a DataFrame from latitude, longitude, and SO2 values.
    """
    df = pd.DataFrame(zip(lat, lon, so2), columns=["Lat", "Long", "SO2"])
    return df.dropna()  # Remove invalid rows


def generate_folium_heatmap(df: pd.DataFrame, min_opacity: float = 0.05) -> folium.Map:
    """
    Generate a folium heatmap from a DataFrame of Lat, Long, SO2 values.
    """
    heat_data = [[row.Lat, row.Long, row.SO2] for row in df.itertuples(index=False)]

    fmap = folium.Map()
    HeatMap(heat_data, min_opacity=min_opacity).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    return fmap


def main() -> None:
    """
    Execute the SO2 data loading, processing, and visualization pipeline.
    """
    print("📦 Loading SO2 satellite data...")
    dataset = load_hdf5_files(DATA_DIR)

    if not dataset:
        print("❌ No valid HDF5 files found. Exiting.")
        return

    print("📊 Extracting fields: Latitude, Longitude, SO2...")
    latitudes = extract_dataset_field(dataset, LAT_KEY)
    longitudes = extract_dataset_field(dataset, LON_KEY)
    so2_values = extract_dataset_field(dataset, SO2_KEY)

    print("🧮 Creating DataFrame...")
    df = create_so2_dataframe(latitudes, longitudes, so2_values)
    print(df.head())

    print("🗺️ Generating heatmap...")
    heatmap = generate_folium_heatmap(df)

    output_path = Path(".") / "so2_heatmap.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap.save(output_path)
    print(f"✅ Heatmap saved to: {output_path.resolve()}")

    # Close opened files
    for file in dataset:
        file.close()


if __name__ == "__main__":
    main()
