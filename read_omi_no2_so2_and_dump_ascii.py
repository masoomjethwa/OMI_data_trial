#!/usr/bin/env python3

import h5py
import numpy as np
import sys
import time
import calendar
from pathlib import Path


def convert_scan_time_to_datetime(scan_time: np.ndarray) -> dict[str, np.ndarray]:
    """
    Convert OMI scan time (seconds since Dec 31, 1992 23:59:59 UTC) to datetime components.

    Args:
        scan_time (np.ndarray): 1D array of scan times.

    Returns:
        dict: Dictionary with keys 'year', 'month', 'day', 'hour', 'minute', 'second', each an np.ndarray.
    """
    # Base timestamp: Dec 31, 1992 @ 23:59:59 UTC in seconds since epoch
    base_timestamp = calendar.timegm(time.strptime('Dec 31, 1992 @ 23:59:59 UTC', '%b %d, %Y @ %H:%M:%S UTC'))

    years = np.zeros(scan_time.shape[0], dtype=int)
    months = np.zeros(scan_time.shape[0], dtype=int)
    days = np.zeros(scan_time.shape[0], dtype=int)
    hours = np.zeros(scan_time.shape[0], dtype=int)
    minutes = np.zeros(scan_time.shape[0], dtype=int)
    seconds = np.zeros(scan_time.shape[0], dtype=int)

    for i, t in enumerate(scan_time):
        # Convert scan_time seconds to absolute timestamp
        absolute_time = t + base_timestamp
        tm = time.gmtime(absolute_time)
        years[i] = tm.tm_year
        months[i] = tm.tm_mon
        days[i] = tm.tm_mday
        hours[i] = tm.tm_hour
        minutes[i] = tm.tm_min
        seconds[i] = tm.tm_sec

    return {
        "year": years,
        "month": months,
        "day": days,
        "hour": hours,
        "minute": minutes,
        "second": seconds,
    }


def main():
    # Try to open the file list
    file_list_path = Path("fileList.txt")
    if not file_list_path.is_file():
        print('Did not find a text file containing file names (perhaps name does not match)')
        sys.exit(1)

    with file_list_path.open('r') as file_list:
        for file_name_raw in file_list:
            file_name = file_name_raw.strip()
            if not file_name:
                continue

            user_input = input(f'\nWould you like to process\n{file_name}\n\n(Y/N): ').strip().lower()
            if user_input == 'n':
                print('Skipping...')
                continue

            hdf5_path = Path(file_name)
            if not hdf5_path.is_file():
                print(f"File not found: {file_name}")
                continue

            with h5py.File(file_name, 'r') as file:
                # Determine if NO2 or SO2 file
                if 'NO2' in file_name:
                    print('This is an OMI NO2 file. Saving...')
                    sds_dict = {1: 'ColumnAmountNO2', 2: 'ColumnAmountNO2Std', 3: 'VcdQualityFlags'}
                    data_fields = file['HDFEOS']['SWATHS']['ColumnAmountNO2']['Data Fields']
                    geolocation = file['HDFEOS']['SWATHS']['ColumnAmountNO2']['Geolocation Fields']

                elif 'SO2' in file_name:
                    print('This is an OMI SO2 file. Saving...')
                    sds_dict = {1: 'ColumnAmountSO2_PBL', 2: 'ColumnAmountO3', 3: 'QualityFlags_PBL'}
                    data_fields = file['HDFEOS']['SWATHS']['OMI Total Column Amount SO2']['Data Fields']
                    geolocation = file['HDFEOS']['SWATHS']['OMI Total Column Amount SO2']['Geolocation Fields']

                else:
                    print(f'The file named: {file_name} is not a valid OMI file.\n')
                    continue

                # Get lat, lon, and scan time as 1D arrays
                lat = geolocation['Latitude'][:].ravel()
                lon = geolocation['Longitude'][:].ravel()
                scan_time = geolocation['Time'][:].ravel()

                # Convert scan_time to datetime components
                dt_components = convert_scan_time_to_datetime(scan_time)
                year = dt_components['year']
                month = dt_components['month']
                day = dt_components['day']
                hour = dt_components['hour']
                minute = dt_components['minute']
                second = dt_components['second']

                # Number of columns: 8 fixed + number of SDS to save
                n_cols = 8 + len(sds_dict)
                n_rows = year.size

                # Prepare output array (float) to store data + datetime + lat/lon
                output = np.zeros((n_rows, n_cols), dtype=float)

                # Fill fixed columns: year, month, day, hour, minute, second, lat, lon
                output[:, 0] = year
                output[:, 1] = month
                output[:, 2] = day
                output[:, 3] = hour
                output[:, 4] = minute
                output[:, 5] = second
                output[:, 6] = lat
                output[:, 7] = lon

                # Prepare headers for CSV columns
                headers = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'Latitude', 'Longitude']

                # Extract SDS datasets and add to output
                for col_idx, sds_key in enumerate(range(1, n_cols - 7), start=8):
                    sds_name = sds_dict[sds_key]
                    try:
                        sds = data_fields[sds_name]
                    except KeyError:
                        print(f'SDS "{sds_name}" not found in the file. Exiting.')
                        sys.exit(1)

                    scale = sds.attrs.get('ScaleFactor', 1.0)
                    fill_value = sds.attrs.get('_FillValue', np.nan)
                    missing_value = sds.attrs.get('MissingValue', np.nan)
                    offset = sds.attrs.get('Offset', 0.0)

                    data = sds[:].ravel().astype(float)

                    # Convert fill and missing values to nan for processing
                    data[data == fill_value] = np.nan
                    data[data == missing_value] = np.nan

                    # Apply scale and offset
                    data = (data - offset) * scale

                    # Replace NaNs with fill_value for saving (optional, could save as NaN)
                    data[np.isnan(data)] = fill_value

                    output[:, col_idx] = data
                    headers.append(sds_name)

                # Save output to CSV
                output_filename = hdf5_path.with_suffix('.csv')
                print(f"Saving to {output_filename}")

                # Write CSV with headers, formatting numbers as floats or ints as appropriate
                with output_filename.open('w') as outfile:
                    # Write header
                    outfile.write(','.join(headers) + '\n')

                    # Write data rows (convert floats to string)
                    for row in output:
                        row_str = ','.join(f"{val:.6g}" if isinstance(val, float) else str(val) for val in row)
                        outfile.write(row_str + '\n')

    print('\nAll files have been checked successfully. Truth Alone Triumphs!')


if __name__ == "__main__":
    main()
