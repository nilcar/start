

import datetime
import math
import pandas
import numpy
from os import listdir
from os.path import isfile, join
import sys





def build_frame(directory_data, directory_labels):
	
	CSV_COLUMN_NAMES_DATA = ['VEHICL_ID', 'T_CHASSIS','PARAMETER_CODE', 'SEND_DATETIME']
	CSV_COLUMN_NAMES_LABELS = ['VEHICL_ID', 'T_CHASSIS', 'PART_CODE', 'Record_count', 'RepairDate']
	
	for y in range(1, 21):
		for x in range(1, 21):
			column = str(y) + '_' + str(x)
			CSV_COLUMN_NAMES_DATA.append(column)
	
	datafiles = []
	for item in listdir(directory_data): 
		if isfile(join(directory_data, item)):
			datafiles.append(directory_data + item)
	
	dataframe = pandas.DataFrame()
	for datafile in datafiles:
		csv_data = pandas.read_csv(datafile, sep=";", names=CSV_COLUMN_NAMES_DATA, header=0, index_col=False)
		
		dataframe = dataframe.append(csv_data, ignore_index=True)
	
	dataframe['valid'] = 0
	dataframe['repaired'] = 0
	
	print(dataframe.head())
	
	labelfiles = []
	for item in listdir(directory_labels): 
		if isfile(join(directory_labels, item)):
			labelfiles.append(directory_labels + item)
	
	labelframe = pandas.DataFrame()
	for datafile in labelfiles:
		csv_data = pandas.read_csv(datafile, sep=";", names=CSV_COLUMN_NAMES_LABELS, header=0, index_col=False)
		labelframe = labelframe.append(csv_data, ignore_index=True)
	
	print(labelframe.head())
	
	repairdatesmapping = {}
	for index, row in labelframe.iterrows():
		datelist = []
		try:
			datelist = repairdatesmapping[row['T_CHASSIS']]
			datelist.append(row['RepairDate'])
			repairdatesmapping[row['T_CHASSIS']] = datelist
		except KeyError:
			datelist.append(row['RepairDate'])
			repairdatesmapping[row['T_CHASSIS']] = datelist
	
	found_chassis = 0
	found_repaired = 0
	nr_of_rows = 0
	
	for index, row in dataframe.iterrows():
		nr_of_rows += 1
		try:
			datelist = repairdatesmapping[row['T_CHASSIS']]
			dataframe.loc[index, 'valid'] = 1
			found_chassis += 1
			print('Found chassis: ' + str(found_chassis))
			
			for reapairdate in datelist:
				datedelta = pandas.to_datetime(reapairdate) - pandas.to_datetime(row['SEND_DATETIME'])
				days = datedelta.days
				if days >= 0 and days < 90:
					dataframe.loc[index, 'repaired'] = 1
					found_repaired += 1
					print('Found repaired: ' + str(found_repaired))
		except KeyError:
			None
	
	"""
	for index, row in dataframe.iterrows():
		#if dataframe.loc[index, :].at['repaired'] == 1:
		if dataframe.loc[index,'repaired'] == 1:
			print('Repaired inserted')
	"""
	
	print('Chassis total: ' + str(nr_of_rows))
	print('Chassis found: ' + str(found_chassis))
	print('Chassis repaired: ' + str(found_repaired))
	datestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '--')
	datestring = datestring.replace(':', '-')
	dataframe.to_csv('data_frame--' + datestring + '.csv', sep=';', index = False, index_label = False)
	
	
	

def check_chassis(directory):

	
	CSV_COLUMN_NAMES = ['VEHICL_ID', 'T_CHASSIS','PART_CODE', 'Record_count', 'RepairDate']
	
	chassis_numbers = {}
	
	datafiles = []
	for item in listdir(directory): 
		if isfile(join(directory, item)):
			datafiles.append(directory + item)
	
	for datafile in datafiles:
		csv_data = pandas.read_csv(datafile, sep=";", names=CSV_COLUMN_NAMES, header=0, index_col=False)
		for index, row in csv_data.iterrows():
			# Only insert data where truck_id is found in label_mapping
			chassi = row['T_CHASSIS']
			try:
				nr_of_chassis = chassis_numbers[chassi]
				nr_of_chassis += 1
				chassis_numbers[chassi] = nr_of_chassis
			except KeyError:
				chassis_numbers[chassi] = 1
				
	doublesframe = pandas.DataFrame()
	doublesframe['T_CHASSIS'] = chassis_numbers.keys()
	doublesframe['number_of'] = chassis_numbers.values()
	doublesframe = doublesframe.sort_values(['number_of'], ascending = [False])
	
	
	datestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '--')
	datestring = datestring.replace(':', '-')
	doublesframe.to_csv('frame_chassis' + datestring + '.csv', sep=';', index = False, index_label = False)	
	

def analyse_frame(directory):	
	
	datafiles = []
	for item in listdir(directory): 
		if isfile(join(directory, item)):
			datafiles.append(directory + item)
	
	for datafile in datafiles:
		csv_data = pandas.read_csv(datafile, sep=";", index_col=False)
	
	nr_of_rows = 0
	nr_of_repaired = 0
	nr_of_valid = 0
	for index, row in csv_data.iterrows():
		nr_of_rows += 1
		if row['valid'] == 1:
			nr_of_valid += 1
		if row['repaired'] == 1:
			nr_of_repaired += 1
	
	print('Rows: ' + str(nr_of_rows))
	print('Valid: ' + str(nr_of_valid))
	print('Repaired: ' + str(nr_of_repaired))
	

analyse_frame('Data/Compressed/')	

#build_frame('Data/Flatten/', 'Data/Labels/')	
	
#check_chassis('Data/...') #








