#!/usr/bin/python
'''
Module: read_omi_no2_so2_at_a_location.py
==========================================================================================
Disclaimer: The code is for demonstration purposes only. Users are responsible to check for accuracy and revise to fit their objective.

Author: Justin Roberts-Pierel, 2015 
Organization: NASA ARSET
Purpose: To view info about a variety of SDS from an OMI he5 file both generally and at a specific lat/lon

See the README associated with this module for more information.
==========================================================================================
'''

import h5py
import numpy as np
import sys

def main():
    try:
        fileList = open('fileList.txt','r')
    except FileNotFoundError:
        print('Did not find a text file containing file names (perhaps name does not match)')
        sys.exit()

    for FILE_NAME in fileList:
        FILE_NAME = FILE_NAME.strip()
        user_input = input(f'\nWould you like to process\n{FILE_NAME}\n\n(Y/N): ').lower()
        if user_input == 'n':
            print('Skipping...')
            continue

        try:
            file = h5py.File(FILE_NAME, 'r')   # Open file read-only
        except Exception as e:
            print(f"Error opening file {FILE_NAME}: {e}")
            continue

        if 'NO2' in FILE_NAME:
            print('This is an OMI NO2 file. Here is some information:')
            dataFields = file['HDFEOS']['SWATHS']['ColumnAmountNO2']['Data Fields']
            geolocation = file['HDFEOS']['SWATHS']['ColumnAmountNO2']['Geolocation Fields']
            SDS_NAME = 'ColumnAmountNO2'
            data = dataFields[SDS_NAME]
            map_label = data.attrs['Units'].decode()
        elif 'SO2' in FILE_NAME:
            print('This is an OMI SO2 file. Here is some information:')
            dataFields = file['HDFEOS']['SWATHS']['OMI Total Column Amount SO2']['Data Fields']
            geolocation = file['HDFEOS']['SWATHS']['OMI Total Column Amount SO2']['Geolocation Fields']
            SDS_NAME = 'ColumnAmountSO2_PBL'
            data = dataFields[SDS_NAME]
            valid_min = data.attrs['ValidRange'][0]
            valid_max = data.attrs['ValidRange'][1]
            map_label = data.attrs['Units'].decode()
            print('Valid Range is:', valid_min, valid_max)
        else:
            print(f'The file named: {FILE_NAME} is not a valid OMI NO2 or SO2 file.\n')
            file.close()
            continue

        lat = geolocation['Latitude'][:]
        lon = geolocation['Longitude'][:]
        min_lat, max_lat = np.min(lat), np.max(lat)
        min_lon, max_lon = np.min(lon), np.max(lon)

        try:
            sds = dataFields[SDS_NAME]
        except KeyError:
            print(f'Sorry, your OMI file does not contain the SDS: {SDS_NAME}. Please try again with the correct file type.')
            file.close()
            continue

        scale = sds.attrs['ScaleFactor']
        fv = sds.attrs['_FillValue']
        mv = sds.attrs['MissingValue']
        offset = sds.attrs['Offset']

        dataArray = sds[:].astype(float)
        dataArray[dataArray == float(fv)] = np.nan
        dataArray[dataArray == float(mv)] = np.nan
        dataArray = scale * (dataArray - offset)

        print(f'The range of latitude in this file is: {min_lat} to {max_lat} degrees')
        print(f'The range of longitude in this file is: {min_lon} to {max_lon} degrees')

        # Get valid user lat/lon
        while True:
            try:
                user_lat = float(input('\nPlease enter the latitude you would like to analyze (Deg. N): '))
                if min_lat <= user_lat <= max_lat:
                    break
                print('Latitude out of range. Please enter a valid latitude.')
            except ValueError:
                print('Invalid input. Please enter a numeric value.')

        while True:
            try:
                user_lon = float(input('Please enter the longitude you would like to analyze (Deg. E): '))
                if min_lon <= user_lon <= max_lon:
                    break
                print('Longitude out of range. Please enter a valid longitude.')
            except ValueError:
                print('Invalid input. Please enter a numeric value.')

        # Haversine formula to find nearest point in meters
        R = 6371000  # Earth radius in meters
        lat1 = np.radians(user_lat)
        lat2 = np.radians(lat)
        delta_lat = np.radians(lat - user_lat)
        delta_lon = np.radians(lon - user_lon)

        a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        d = R * c

        x, y = np.unravel_index(d.argmin(), d.shape)
        print(f'\nThe nearest pixel to your entered location is at:\nLatitude: {lat[x, y]} Longitude: {lon[x, y]}')

        if np.isnan(dataArray[x, y]):
            print(f'The value of {SDS_NAME} at this pixel is {fv} (No Value)\n')
        else:
            print(f'The value of {SDS_NAME} at this pixel is {round(dataArray[x, y], 3)}')

        # Helper function to safely get slices
        def safe_slice(idx, max_idx, radius):
            start = max(idx - radius, 0)
            end = min(idx + radius + 1, max_idx)
            return start, end

        # 3x3 grid stats
        x_start, x_end = safe_slice(x, dataArray.shape[0], 1)
        y_start, y_end = safe_slice(y, dataArray.shape[1], 1)
        three_by_three = dataArray[x_start:x_end, y_start:y_end].astype(float)
        nnan = np.count_nonzero(~np.isnan(three_by_three))
        if nnan == 0:
            print('There are no valid pixels in a 3x3 grid centered at your entered location.')
        else:
            avg_3x3 = np.nanmean(three_by_three)
            std_3x3 = np.nanstd(three_by_three)
            med_3x3 = np.nanmedian(three_by_three)
            pixel_word = 'pixel' if nnan == 1 else 'pixels'
            print(f'There are {nnan} valid {pixel_word} in a 3x3 grid centered at your entered location.')
            print(f'Average: {round(avg_3x3,3)}\nMedian: {round(med_3x3,3)}\nStandard deviation: {round(std_3x3,3)}')

        # 5x5 grid stats
        x_start, x_end = safe_slice(x, dataArray.shape[0], 2)
        y_start, y_end = safe_slice(y, dataArray.shape[1], 2)
        five_by_five = dataArray[x_start:x_end, y_start:y_end].astype(float)
        nnan = np.count_nonzero(~np.isnan(five_by_five))
        if nnan == 0:
            print('\nThere are no valid pixels in a 5x5 grid centered at your entered location.\n')
        else:
            avg_5x5 = np.nanmean(five_by_five)
            std_5x5 = np.nanstd(five_by_five)
            med_5x5 = np.nanmedian(five_by_five)
            pixel_word = 'pixel' if nnan == 1 else 'pixels'
            print(f'\nThere are {nnan} valid {pixel_word} in a 5x5 grid centered at your entered location.')
            print(f'Average: {round(avg_5x5,3)}\nMedian: {round(med_5x5,3)}\nStandard deviation: {round(std_5x5,3)}')

        file.close()

    fileList.close()

if __name__ == "__main__":
    main()
