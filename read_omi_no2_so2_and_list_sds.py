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

#import necessary modules
import h5py
import sys

#This finds the user's current path so that all hdf4 files can be found
try:
    fileList = open('fileList.txt', 'r')
except:
    print('Did not find a text file containing file names (perhaps name does not match)')
    sys.exit()

#loops through all files listed in the text file
for FILE_NAME in fileList:
    FILE_NAME = FILE_NAME.strip()
    user_input = input('\nWould you like to process\n' + FILE_NAME + '\n\n(Y/N) ').strip().lower()
    if user_input == 'n':
        print('Skipping...')
        continue
    else:
        try:
            file = h5py.File(FILE_NAME, 'r')   # Open file read-only
        except Exception as e:
            print(f'Error opening file {FILE_NAME}: {e}')
            continue

        #checks if the file contains NO2 or SO2 data, and reacts accordingly
        if 'NO2' in FILE_NAME:
            print('\nThis is an OMI NO2 file. Here is a list of SDS in your file:\n')
            dataFields = file['HDFEOS']['SWATHS']['ColumnAmountNO2']['Data Fields']            
        elif 'SO2' in FILE_NAME:
            print('\nThis is an OMI SO2 file. Here is a list of SDS in your file:\n')
            dataFields = file['HDFEOS']['SWATHS']['OMI Total Column Amount SO2']['Data Fields']
        else:
            print('The file named:', FILE_NAME, ' is not a valid OMI file.\n')
            file.close()
            continue

        #Print the list of SDS found in the file 
        for sds_name in dataFields:
            print(f' {sds_name}, dim={dataFields[sds_name].shape} \n')

        #close the hdf5 file 
        file.close()

fileList.close()
