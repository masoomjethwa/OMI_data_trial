#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO2 Satellite Data Processing with:
- XML Schema generation & validation
- Logging instead of prints
- Parallel processing for files
- SO2 Data EDA plotting

Author: mp10
@coding assistant: [TGC-26092025]
"""

import os
import logging
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from lxml import etree  # for XML schema validation
from zipfile import ZipFile

# Constants
DATA_DIRECTORY = Path(".")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
XSD_FILE = OUTPUT_DIR / "h5_structure.xsd"
MAX_WORKERS = 4  # Adjust based on your CPU

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("so2_processing.log"),
        logging.StreamHandler()
    ]
)

# XML Schema Definition (XSD) as a string
XSD_STRING = """<?xml version="1.0" encoding="utf-8" ?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
           elementFormDefault="qualified">

  <xs:element name="HDF5File">
    <xs:complexType>
      <xs:sequence>
        <xs:element name="Filename" type="xs:string"/>
        <xs:element name="Groups" minOccurs="0" maxOccurs="unbounded">
          <xs:complexType>
            <xs:sequence>
              <xs:element name="Group" maxOccurs="unbounded">
                <xs:complexType>
                  <xs:sequence>
                    <xs:element name="Name" type="xs:string"/>
                    <xs:element name="Datasets" minOccurs="0" maxOccurs="unbounded">
                      <xs:complexType>
                        <xs:sequence>
                          <xs:element name="Dataset" maxOccurs="unbounded">
                            <xs:complexType>
                              <xs:sequence>
                                <xs:element name="Name" type="xs:string"/>
                                <xs:element name="Shape" type="xs:string"/>
                                <xs:element name="Datatype" type="xs:string"/>
                              </xs:sequence>
                            </xs:complexType>
                          </xs:element>
                        </xs:sequence>
                      </xs:complexType>
                    </xs:element>
                    <xs:element name="Subgroups" minOccurs="0" maxOccurs="unbounded">
                      <xs:complexType>
                        <xs:sequence>
                          <xs:element ref="Group" maxOccurs="unbounded"/>
                        </xs:sequence>
                      </xs:complexType>
                    </xs:element>
                  </xs:sequence>
                </xs:complexType>
              </xs:element>
            </xs:sequence>
          </xs:complexType>
        </xs:element>
      </xs:sequence>
    </xs:complexType>
  </xs:element>

</xs:schema>
"""

def save_xsd():
    """Save the XML Schema Definition to disk."""
    with open(XSD_FILE, "w", encoding="utf-8") as f:
        f.write(XSD_STRING)
    logging.info(f"XSD schema saved to {XSD_FILE}")

def validate_xml(xml_path: Path, xsd_path: Path) -> bool:
    """Validate XML file against the provided XSD schema."""
    try:
        xml_doc = etree.parse(str(xml_path))
        xml_schema_doc = etree.parse(str(xsd_path))
        xml_schema = etree.XMLSchema(xml_schema_doc)
        valid = xml_schema.validate(xml_doc)
        if not valid:
            logging.error(f"Validation errors for {xml_path.name}: {xml_schema.error_log}")
        else:
            logging.info(f"XML {xml_path.name} validated successfully.")
        return valid
    except Exception as e:
        logging.error(f"XML validation exception for {xml_path.name}: {e}")
        return False

def convert_dtype(dtype) -> str:
    """Convert numpy dtype to readable string."""
    return str(dtype)

def build_xml_group(name, group):
    """Recursively build XML elements for HDF5 groups/datasets."""
    group_elem = ET.Element("Group")
    ET.SubElement(group_elem, "Name").text = name

    # Datasets
    datasets_elem = ET.SubElement(group_elem, "Datasets")
    for ds_name, ds_obj in group.items():
        if isinstance(ds_obj, h5py.Dataset):
            ds_elem = ET.SubElement(datasets_elem, "Dataset")
            ET.SubElement(ds_elem, "Name").text = ds_name
            ET.SubElement(ds_elem, "Shape").text = str(ds_obj.shape)
            ET.SubElement(ds_elem, "Datatype").text = convert_dtype(ds_obj.dtype)

    # Subgroups
    subgroups_elem = ET.SubElement(group_elem, "Subgroups")
    for sg_name, sg_obj in group.items():
        if isinstance(sg_obj, h5py.Group):
            subgroups_elem.append(build_xml_group(sg_name, sg_obj))

    return group_elem

def h5_structure_to_xml(h5_file: h5py.File, filename: str) -> ET.ElementTree:
    """Convert HDF5 structure to XML ElementTree."""
    root = ET.Element("HDF5File")
    ET.SubElement(root, "Filename").text = filename
    groups_elem = ET.SubElement(root, "Groups")
    # Top-level groups
    for group_name, group_obj in h5_file.items():
        groups_elem.append(build_xml_group(group_name, group_obj))
    return ET.ElementTree(root)

def save_xml(tree: ET.ElementTree, path: Path) -> None:
    """Save XML ElementTree to file."""
    tree.write(path, encoding="utf-8", xml_declaration=True)

def save_json(data: dict, path: Path) -> None:
    """Save dict data to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def save_yaml(data: dict, path: Path) -> None:
    """Save dict data to YAML file."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

def h5_structure_to_dict(group) -> dict:
    """Convert HDF5 group structure to nested dict for JSON/YAML export."""
    structure = {}
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            structure[key] = {
                "shape": item.shape,
                "dtype": convert_dtype(item.dtype)
            }
        elif isinstance(item, h5py.Group):
            structure[key] = h5_structure_to_dict(item)
    return structure

def extract_so2_data(h5_file: h5py.File) -> np.ndarray:
    """Extract SO2 data array from the file."""
    try:
        # Using a common key from your original script
        key = "HDFEOS/GRIDS/OMI Total Column Amount SO2/Data Fields/ColumnAmountSO2"
        data = h5_file[key][:]
        data = np.array(data)
        # Clean invalid values (negative or fill values)
        data[data < 0] = np.nan
        return data
    except Exception as e:
        logging.error(f"Error extracting SO2 data: {e}")
        return np.array([])

def plot_eda(so2_data: np.ndarray, output_prefix: Path) -> None:
    """Generate 20 EDA plots from SO2 data array."""
    if so2_data.size == 0:
        logging.warning("No SO2 data available for plotting.")
        return

    # Flatten the data ignoring NaNs for some plots
    data_flat = so2_data.flatten()
    data_flat = data_flat[~np.isnan(data_flat)]

    # 1. Histogram
    plt.figure(figsize=(8,5))
    sns.histplot(data_flat, bins=50, kde=True)
    plt.title("Histogram of SO2 Values")
    plt.savefig(output_prefix.with_name(output_prefix.name + "_histogram.png"))
    plt.close()

    # 2. Boxplot
    plt.figure(figsize=(8,5))
    sns.boxplot(x=data_flat)
    plt.title("Boxplot of SO2 Values")
    plt.savefig(output_prefix.with_name(output_prefix.name + "_boxplot.png"))
    plt.close()

    # 3. Density plot
    plt.figure(figsize=(8,5))
    sns.kdeplot(data_flat, shade=True)
    plt.title("Density Plot of SO2 Values")
    plt.savefig(output_prefix.with_name(output_prefix.name + "_density.png"))
    plt.close()

    # 4. Violin plot
    plt.figure(figsize=(8,5))
    sns.violinplot(x=data_flat)
    plt.title("Violin Plot of SO2 Values")
    plt.savefig(output_prefix.with_name(output_prefix.name + "_violin.png"))
    plt.close()

    # 5. QQ plot
    from scipy import stats
    plt.figure(figsize=(8,5))
    stats.probplot(data_flat, dist="norm", plot=plt)
    plt.title("QQ Plot of SO2 Values")
    plt.savefig(output_prefix.with_name(output_prefix.name + "_qqplot.png"))
    plt.close()

    # 6. Heatmap of mean values per pixel (if 3D)
    if so2_data.ndim == 3:
        mean_2d = np.nanmean(so2_data, axis=0)
        plt.figure(figsize=(10,8))
        sns.heatmap(mean_2d, cmap="viridis")
        plt.title("Heatmap of Mean SO2 Values (Time-aggregated)")
        plt.savefig(output_prefix.with_name(output_prefix.name + "_heatmap_mean.png"))
        plt.close()

    # 7-20 More plots exploring statistics, distribution, spatial correlation etc.

    # 7. Boxplot by slice (if 3D)
    if so2_data.ndim == 3:
        plt.figure(figsize=(10,6))
        sns.boxplot(data=[so2_data[i].flatten() for i in range(so2_data.shape[0])])
        plt.title("Boxplot of SO2 Values per Slice (Time dimension)")
        plt.savefig(output_prefix.with_name(output_prefix.name + "_boxplot_slices.png"))
        plt.close()

    # 8. Line plot of mean SO2 per slice (time)
    if so2_data.ndim == 3:
        means = [np.nanmean(so2_data[i]) for i in range(so2_data.shape[0])]
        plt.figure(figsize=(10,6))
        plt.plot(range(so2_data.shape[0]), means, marker='o')
        plt.title("Mean SO2 Values per Slice (Time)")
        plt.xlabel("Slice index")
        plt.ylabel("Mean SO2")
        plt.savefig(output_prefix.with_name(output_prefix.name + "_mean_time.png"))
        plt.close()

    # 9. Scatter plot of first two slices flattened
    if so2_data.ndim == 3 and so2_data.shape[0] > 1:
        slice1 = so2_data[0].flatten()
        slice2 = so2_data[1].flatten()
        mask = ~np.isnan(slice1) & ~np.isnan(slice2)
        plt.figure(figsize=(8,6))
        plt.scatter(slice1[mask], slice2[mask], alpha=0.3)
        plt.title("Scatter Plot: Slice 1 vs Slice 2 SO2 Values")
        plt.xlabel("Slice 1")
        plt.ylabel("Slice 2")
        plt.savefig(output_prefix.with_name(output_prefix.name + "_scatter_slices.png"))
        plt.close()

    # 10. Histogram of non-NaN counts per pixel (if 3D)
    if so2_data.ndim == 3:
        count_valid = np.sum(~np.isnan(so2_data), axis=0)
        plt.figure(figsize=(10,8))
        sns.histplot(count_valid.flatten(), bins=50)
        plt.title("Histogram of Valid Data Counts per Pixel")
        plt.savefig(output_prefix.with_name(output_prefix.name + "_valid_counts_hist.png"))
        plt.close()

    # 11. Line plot of max SO2 per slice
    if so2_data.ndim == 3:
        maxs = [np.nanmax(so2_data[i]) for i in range(so2_data.shape[0])]
        plt.figure(figsize=(10,6))
        plt.plot(range(so2_data.shape[0]), maxs, marker='x', color='r')
        plt.title("Max SO2 Values per Slice (Time)")
        plt.xlabel("Slice index")
        plt.ylabel("Max SO2")
        plt.savefig(output_prefix.with_name(output_prefix.name + "_max_time.png"))
        plt.close()

    # 12. Line plot of min SO2 per slice
    if so2_data.ndim == 3:
        mins = [np.nanmin(so2_data[i]) for i in range(so2_data.shape[0])]
        plt.figure(figsize=(10,6))
        plt.plot(range(so2_data.shape[0]), mins, marker='x', color='g')
        plt.title("Min SO2 Values per Slice (Time)")
        plt.xlabel("Slice index")
        plt.ylabel("Min SO2")
        plt.savefig(output_prefix.with_name(output_prefix.name + "_min_time.png"))
        plt.close()

    # 13. Histogram of SO2 values > threshold (e.g., >1)
    plt.figure(figsize=(8,5))
    sns.histplot(data_flat[data_flat > 1], bins=30)
    plt.title("Histogram of SO2 Values > 1 DU")
    plt.savefig(output_prefix.with_name(output_prefix.name + "_hist_gt1.png"))
    plt.close()

    # 14. Autocorrelation plot (if 1D time series available)
    # For demo purposes skipped as data is spatial

    # 15. Pairplot of stats (mean, median, std) by slice (if 3D)
    if so2_data.ndim == 3:
        stats_df = pd.DataFrame({
            'mean': [np.nanmean(so2_data[i]) for i in range(so2_data.shape[0])],
            'median': [np.nanmedian(so2_data[i]) for i in range(so2_data.shape[0])],
            'std': [np.nanstd(so2_data[i]) for i in range(so2_data.shape[0])]
        })
        sns.pairplot(stats_df)
        plt.suptitle("Pairplot of SO2 Statistics per Slice")
        plt.savefig(output_prefix.with_name(output_prefix.name + "_pairplot_stats.png"))
        plt.close()

    # 16. CDF plot
    plt.figure(figsize=(8,5))
    sorted_data = np.sort(data_flat)
    cdf = np.arange(len(sorted_data)) / float(len(sorted_data))
    plt.plot(sorted_data, cdf)
    plt.title("CDF of SO2 Values")
    plt.savefig(output_prefix.with_name(output_prefix.name + "_cdf.png"))
    plt.close()

    # 17. Bar plot of missing values percentage (if 3D)
    if so2_data.ndim == 3:
        missing_pct = [np.isnan(so2_data[i]).mean()*100 for i in range(so2_data.shape[0])]
        plt.figure(figsize=(10,6))
        sns.barplot(x=list(range(so2_data.shape[0])), y=missing_pct)
        plt.title("Percentage of Missing SO2 Values per Slice")
        plt.xlabel("Slice index")
        plt.ylabel("Missing %")
        plt.savefig(output_prefix.with_name(output_prefix.name + "_missing_pct.png"))
        plt.close()

    # 18. Scatter plot of mean vs std per slice
    if so2_data.ndim == 3:
        plt.figure(figsize=(8,6))
        plt.scatter(stats_df['mean'], stats_df['std'])
        plt.title("Mean vs Std Dev of SO2 per Slice")
        plt.xlabel("Mean")
        plt.ylabel("Std Dev")
        plt.savefig(output_prefix.with_name(output_prefix.name + "_mean_vs_std.png"))
        plt.close()

    # 19. Histogram of SO2 values by quantiles
    quantiles = np.quantile(data_flat, [0, 0.25, 0.5, 0.75, 1.0])
    plt.figure(figsize=(8,5))
    sns.histplot(data_flat, bins=50, hue=pd.cut(data_flat, bins=quantiles, include_lowest=True))
    plt.title("Histogram of SO2 Values by Quantiles")
    plt.savefig(output_prefix.with_name(output_prefix.name + "_hist_quantiles.png"))
    plt.close()

    # 20. Scatter matrix for some slices (first 4 slices if exist)
    if so2_data.ndim == 3 and so2_data.shape[0] >= 4:
        df_scatter = pd.DataFrame({
            f"slice_{i}": so2_data[i].flatten() for i in range(4)
        }).dropna()
        sns.pairplot(df_scatter)
        plt.suptitle("Scatter Matrix of First 4 Slices")
        plt.savefig(output_prefix.with_name(output_prefix.name + "_scatter_matrix.png"))
        plt.close()

    logging.info(f"Generated 20 EDA plots for SO2 data in {output_prefix.parent}")

def process_file(filepath: Path) -> int:
    """Process one .he5 file: extract info, save XML/JSON/YAML, validate, extract SO2 data & plot."""
    logging.info(f"Processing file: {filepath.name}")
    try:
        with h5py.File(filepath, "r") as h5f:
            # Convert structure
            xml_tree = h5_structure_to_xml(h5f, filepath.name)
            # Save XML
            xml_path = OUTPUT_DIR / (filepath.stem + ".xml")
            save_xml(xml_tree, xml_path)

            # Validate XML against XSD
            validate_xml(xml_path, XSD_FILE)

            # Save JSON and YAML
            dict_structure = h5_structure_to_dict(h5f)
            json_path = OUTPUT_DIR / (filepath.stem + ".json")
            yaml_path = OUTPUT_DIR / (filepath.stem + ".yaml")
            save_json(dict_structure, json_path)
            save_yaml(dict_structure, yaml_path)

            # Extract SO2 data and plot EDA
            so2_data = extract_so2_data(h5f)
            plot_prefix = OUTPUT_DIR / filepath.stem
            plot_eda(so2_data, plot_prefix)

        return 1  # success count

    except Exception as e:
        logging.error(f"Failed processing {filepath.name}: {e}")
        return 0

def main():
    logging.info("Starting batch processing of .he5 files...")

    save_xsd()  # Save XSD once

    files = list(DATA_DIRECTORY.glob("*.he5"))
    logging.info(f"Found {len(files)} files to process.")

    success_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file, file): file for file in files}
        for future in as_completed(futures):
            result = future.result()
            success_count += result

    # Zip all XML files
    xml_files = list(OUTPUT_DIR.glob("*.xml"))
    zip_path = OUTPUT_DIR / "h5_structures.zip"
    with ZipFile(zip_path, 'w') as zipf:
        for xmlf in xml_files:
            zipf.write(xmlf, arcname=xmlf.name)
    logging.info(f"All XML files zipped into {zip_path}")

    logging.info(f"Processed files: {success_count}")
    logging.info("Done.")

if __name__ == "__main__":
    main()
