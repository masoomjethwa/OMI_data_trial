#!/usr/bin/python
'''
Module: read_omi_no2_so2_and_list_sds.py
==========================================================================================
Disclaimer: The code is for demonstration purposes only. Users are responsible to check for accuracy and revise to fit their objective.

Author: Justin Roberts-Pierel, 2015 
Organization: NASA ARSET
Purpose: To print all SDS from an OMI hdf5 file

See the README associated with this module for more information.
==========================================================================================
'''

import h5py
import sys

def main():
    try:
        fileList = open('fileList.txt', 'r')
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
            file = h5py.File(FILE_NAME, 'r')  # open file read-only
        except Exception as e:
            print(f'Error opening file {FILE_NAME}: {e}')
            continue

        if 'NO2' in FILE_NAME:
            print('\nThis is an OMI NO2 file. Here is a list of SDS in your file:\n')
            # dataFields = file['HDFEOS']['SWATHS']['ColumnAmountNO2']['Data Fields']
            geolocation = file['HDFEOS']['SWATHS']['ColumnAmountNO2']['Geolocation Fields']
        elif 'SO2' in FILE_NAME:
            print('\nThis is an OMI SO2 file. Here is a list of SDS in your file:\n')
            # dataFields = file['HDFEOS']['SWATHS']['OMI Total Column Amount SO2']['Data Fields']
            geolocation = file['HDFEOS']['SWATHS']['OMI Total Column Amount SO2']['Geolocation Fields']
        else:
            print(f'The file named: {FILE_NAME} is not a valid OMI file.\n')
            file.close()
            continue

        for i in geolocation:
            print(f' {i}, dim= {geolocation[i].shape}\n')

        file.close()

    fileList.close()

if __name__ == "__main__":
    main()
