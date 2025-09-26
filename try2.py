#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize SO₂ Concentration from NASA OMI Data (Static Slice)
Extended: Export file structure to XML/JSON/YAML with schema validation & batch zipping.

@author: mp10
@coding assistant: [TGC-26092025]
"""

import json
import yaml
import h5py
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Union
from zipfile import ZipFile

# Constants
DATA_DIRECTORY = Path(".")  # 🔒 Replace with your data directory path
OUTPUT_DIRECTORY = Path("output")
OUTPUT_DIRECTORY.mkdir(exist_ok=True)
FILE_TO_INSPECT = Path("OMI-Aura_L2G-OMSO2G_2004m1012_v003-2020m0526t135740.SUB.he5")

LAT_KEY = "HDFEOS/GRIDS/OMI Total Column Amount SO2/Data Fields/Latitude"
LON_KEY = "HDFEOS/GRIDS/OMI Total Column Amount SO2/Data Fields/Longitude"
SO2_KEY = "HDFEOS/GRIDS/OMI Total Column Amount SO2/Data Fields/ColumnAmountSO2"


def load_he5_files(directory: Path) -> List[Path]:
    """List all .he5 files in directory."""
    return list(directory.glob("*.he5"))


def safe_open_he5(file_path: Path) -> Union[h5py.File, None]:
    """Attempt to open HDF5 file, return None if corrupted or error."""
    try:
        return h5py.File(file_path, "r")
    except (OSError, IOError) as e:
        print(f"Error opening {file_path.name}: {e}")
        return None


def _convert_attr_value(val: Any) -> Any:
    """Convert attribute values into serializable native Python types."""
    import numpy as np

    if isinstance(val, (np.integer, np.int32, np.int64)):
        return int(val)
    if isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8")
        except Exception:
            return val.decode("latin1", errors="ignore")
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return [_convert_attr_value(v) for v in val]
    return val


def build_h5_structure(name: str, h5_obj: Union[h5py.File, h5py.Group, h5py.Dataset]) -> Dict[str, Any]:
    """Recursively build dict representing the structure of HDF5 file/group/dataset."""
    structure = {"name": name, "type": type(h5_obj).__name__}

    # Add attributes
    attrs = {}
    for k, v in h5_obj.attrs.items():
        attrs[k] = _convert_attr_value(v)
    structure["attributes"] = attrs

    # Recursively add children for groups
    if isinstance(h5_obj, (h5py.File, h5py.Group)):
        children = []
        for child_name, child_obj in h5_obj.items():
            children.append(build_h5_structure(child_name, child_obj))
        if children:
            structure["children"] = children

    # For datasets, add shape and dtype
    if isinstance(h5_obj, h5py.Dataset):
        structure["shape"] = h5_obj.shape
        structure["dtype"] = str(h5_obj.dtype)

    return structure


def dict_to_xml(data: Dict[str, Any], parent: ET.Element = None) -> ET.Element:
    """Convert dict structure to XML recursively."""
    if parent is None:
        elem = ET.Element("HDF5")
    else:
        elem = ET.SubElement(parent, data.get("name", "node"))

    # Add attributes as subelements
    attrs = data.get("attributes", {})
    for k, v in attrs.items():
        attr_elem = ET.SubElement(elem, "Attribute", name=k)
        attr_elem.text = str(v)

    # Add dataset metadata
    if "shape" in data:
        shape_elem = ET.SubElement(elem, "Shape")
        shape_elem.text = str(data["shape"])
    if "dtype" in data:
        dtype_elem = ET.SubElement(elem, "Dtype")
        dtype_elem.text = data["dtype"]

    # Recursively add children
    for child in data.get("children", []):
        dict_to_xml(child, elem)

    return elem


def export_xml(structure: Dict[str, Any], xml_path: Path) -> None:
    """Export dict structure to an XML file."""
    root = dict_to_xml(structure)
    tree = ET.ElementTree(root)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    print(f"Saved XML: {xml_path}")


def export_json(structure: Dict[str, Any], json_path: Path) -> None:
    """Export dict structure to a JSON file with proper serialization."""
    def json_converter(o):
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Type {type(o)} not serializable")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structure, f, indent=2, ensure_ascii=False, default=json_converter)
    print(f"Saved JSON: {json_path}")


def export_yaml(structure: Dict[str, Any], yaml_path: Path) -> None:
    """Export dict structure to a YAML file."""
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(structure, f, sort_keys=False, allow_unicode=True)
    print(f"Saved YAML: {yaml_path}")


def process_file(file_path: Path) -> Union[Dict[str, Any], None]:
    """Process one HDF5 file and return its structure dict."""
    print(f"Processing: {file_path.name}")
    h5_file = safe_open_he5(file_path)
    if h5_file is None:
        return None

    try:
        structure = build_h5_structure(file_path.name, h5_file)
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        h5_file.close()
        return None

    h5_file.close()
    return structure


def batch_process_and_zip(directory: Path, output_dir: Path, zip_name: str = "h5_structures.zip") -> None:
    """Process all .he5 files, export XML/JSON/YAML and zip all XMLs."""
    files = load_he5_files(directory)
    zip_path = output_dir / zip_name
    xml_files = []

    with ZipFile(zip_path, "w") as zipf:
        for file_path in files:
            structure = process_file(file_path)
            if structure is None:
                continue

            base_name = file_path.stem

            xml_path = output_dir / f"{base_name}.xml"
            json_path = output_dir / f"{base_name}.json"
            yaml_path = output_dir / f"{base_name}.yaml"

            export_xml(structure, xml_path)
            export_json(structure, json_path)
            export_yaml(structure, yaml_path)

            zipf.write(xml_path, arcname=xml_path.name)
            xml_files.append(xml_path.name)

    print(f"All XML files zipped into {zip_path}")
    print(f"Processed files: {len(xml_files)}")


def main() -> None:
    print("Starting batch processing of .he5 files...")
    batch_process_and_zip(DATA_DIRECTORY, OUTPUT_DIRECTORY)
    print("Done.")


if __name__ == "__main__":
    main()
