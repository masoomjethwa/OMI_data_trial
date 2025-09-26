#!/usr/bin/env python3
# @author: mp10
# @coding assistant: [TGC-DD26092025]
# pip install h5py

"""
Purpose: Read and dump all OMI NO2 and SO2 HDF5 files listed in a text file.
"""

from pathlib import Path
from typing import Iterator
import sys
import h5py


def read_file_list(file_path: Path) -> Iterator[str]:
    """
    Reads a list of file names from a text file, yielding each filename stripped of whitespace.

    Args:
        file_path (Path): Path to the text file containing the list of filenames.

    Yields:
        str: Filename stripped of leading/trailing whitespace.
    """
    try:
        with file_path.open('r', encoding='utf-8') as file_list:
            for line in file_list:
                yield line.strip()
    except FileNotFoundError:
        print(f"File list not found: {file_path}")
        sys.exit(1)


def process_file(file_path: Path) -> None:
    """
    Process an individual HDF5 file, displaying SDS datasets based on file type (NO2 or SO2).

    Args:
        file_path (Path): Path to the HDF5 file to process.
    """
    try:
        with h5py.File(file_path, 'r') as hdf_file:
            if 'NO2' in file_path.name:
                print(f"\nThis is an OMI NO2 file. Here is a list of SDS in your file:\n")
                data_fields = hdf_file['HDFEOS']['SWATHS']['ColumnAmountNO2']['Data Fields']
            elif 'SO2' in file_path.name:
                print(f"\nThis is an OMI SO2 file. Here is a list of SDS in your file:\n")
                data_fields = hdf_file['HDFEOS']['SWATHS']['OMI Total Column Amount SO2']['Data Fields']
            else:
                print(f"The file named: {file_path.name} is not a valid OMI file.\n")
                return

            # Print all SDS datasets with their dimensions
            for dataset_name in data_fields:
                shape = data_fields[dataset_name].shape
                print(f"  {dataset_name}, dim={shape}\n")

    except (OSError, KeyError) as error:
        print(f"Error processing file {file_path.name}: {error}")


def main() -> None:
    """
    Main program logic: reads the file list and processes each file with user confirmation.
    """
    file_list_path = Path('path/to/fileList.txt')

    for file_name in read_file_list(file_list_path):
        file_path = Path(file_name)

        user_response = input(f"\nWould you like to process\n{file_path}\n\n(Y/N): ").strip().lower()
        if user_response == 'n':
            print("Skipping...")
            continue

        if not file_path.exists():
            print(f"File does not exist: {file_path}")
            continue

        process_file(file_path)


if __name__ == '__main__':
    main()
