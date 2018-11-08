

import datetime
import math
import pandas
import numpy
from os import listdir
from os.path import isfile, join
import sys


def build_frameUpsample(directory):

	datafiles = []
	for item in listdir(directory): 
		if isfile(join(directory, item)):
			datafiles.append(directory + item)
	
	for datafile in datafiles:
		dataframe = pandas.read_csv(datafile, sep=";", index_col=False)
	
	dataframe = dataframe.reindex(numpy.random.permutation(dataframe.index))
	dataframe = dataframe.sort_values(['repaired'], ascending = [False])
	
	delete_rows = []
	nr_valid = 0
	row_nr = 1
	doubleframe = pandas.DataFrame()
	df_columns = dataframe.columns
	
	for index, row in dataframe.iterrows():
	
		print('Row: ' + str(row_nr))
		if row['repaired'] == 1:
			tempframe = pandas.DataFrame([row.values], columns=df_columns)
			doubleframe = doubleframe.append(tempframe, ignore_index=True)
			doubleframe = doubleframe.append(tempframe, ignore_index=True)
			nr_valid += 1
		if row['repaired'] == 0 and nr_valid > 0:
			nr_valid -= 1
		if row['repaired'] == 0 and nr_valid <= 0:
			delete_rows.append(index)
		row_nr += 1
	
	dataframe = dataframe.drop(delete_rows)
	dataframe = dataframe.append(doubleframe, ignore_index=True)
	
	print('Deleted rows:' + str(len(delete_rows)))
	datestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '--')
	datestring = datestring.replace(':', '-')
	dataframe.to_csv('data_frameUpsample--' + datestring + '.csv', sep=';', index = False, index_label = False)	

def build_frameV1(directory):

	datafiles = []
	for item in listdir(directory): 
		if isfile(join(directory, item)):
			datafiles.append(directory + item)
	
	for datafile in datafiles:
		dataframe = pandas.read_csv(datafile, sep=";", index_col=False)
	
	dataframe = dataframe.reindex(numpy.random.permutation(dataframe.index))
	dataframe = dataframe.sort_values(['repaired'], ascending = [False])
	
	delete_rows = []
	nr_valid = 0
	row_nr = 1
	for index, row in dataframe.iterrows():
	
		print('Row: ' + str(row_nr))
		if row['repaired'] == 1:
			nr_valid += 1
		if row['repaired'] == 0 and nr_valid > 0:
			nr_valid -= 1
		if row['repaired'] == 0 and nr_valid <= 0:
			delete_rows.append(index)
		row_nr += 1
	
	dataframe = dataframe.drop(delete_rows)
	print('Deleted rows:' + str(len(delete_rows)))
	datestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '--')
	datestring = datestring.replace(':', '-')
	dataframe.to_csv('data_frameV1--' + datestring + '.csv', sep=';', index = False, index_label = False)	
	
def build_frameV3(directory_data):
	
	CSV_COLUMN_NAMES_DATA = ['VEHICL_ID', 'T_CHASSIS','PARAMETER_CODE']
	
	for y in range(1, 21):
		for x in range(1, 21):
			column = str(y) + '_' + str(x)
			CSV_COLUMN_NAMES_DATA.append(column)
	
	CSV_COLUMN_NAMES_DATA.append('Send_Date')
	CSV_COLUMN_NAMES_DATA.append('All_Fault')
	CSV_COLUMN_NAMES_DATA.append('All_Fault_in_3_months')
	CSV_COLUMN_NAMES_DATA.append('PARTITIONNING')
	
	print(len(CSV_COLUMN_NAMES_DATA))
	
	
	
	datafiles = []
	for item in listdir(directory_data): 
		if isfile(join(directory_data, item)):
			datafiles.append(directory_data + item)
	
	dataframe = pandas.DataFrame()
	for datafile in datafiles:
		csv_data = pandas.read_csv(datafile, sep=";", names=CSV_COLUMN_NAMES_DATA, header=0, index_col=False)

	datestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '--')
	datestring = datestring.replace(':', '-')
	csv_data.to_csv('data_frameV3--' + datestring + '.csv', sep=';', index = False, index_label = False)


def build_frame(directory_data, directory_labels):
	
	CSV_COLUMN_NAMES_DATA = ['VEHICL_ID', 'T_CHASSIS','PARAMETER_CODE', 'truck_date']
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
	exclude_number = 0
	delete_rows = []
	
	for index, row in dataframe.iterrows():
		nr_of_rows += 1
		try:
			datelist = repairdatesmapping[row['T_CHASSIS']]
			dataframe.loc[index, 'valid'] = 1
			found_chassis += 1
			print('Found chassis: ' + str(found_chassis))
			
			for reapairdate in datelist:
				datedelta = pandas.to_datetime(reapairdate) - pandas.to_datetime(row['truck_date'])
				days = datedelta.days
				if days >= 0 and days < 90:
					dataframe.loc[index, 'repaired'] = 1
					found_repaired += 1
					print('Found repaired: ' + str(found_repaired))
		except KeyError:
			exclude_number += 1
			if exclude_number != 1:
				delete_rows.append(index)
				print('deleting index: ' + str(index))
			if exclude_number > 9:
				exclude_number = 0
	
	"""
	for index, row in dataframe.iterrows():
		#if dataframe.loc[index, :].at['repaired'] == 1:
		if dataframe.loc[index,'repaired'] == 1:
			print('Repaired inserted')
	"""
	print('Dateframe before deletion: ' + str(dataframe.size))
	print('Deleting rows: ' + str(len(delete_rows)))
	#dataframe = dataframe.drop(delete_rows)
	print('Dateframe after deletion: ' + str(dataframe.size))
	
	print('Chassis total: ' + str(nr_of_rows))
	print('Chassis found: ' + str(found_chassis))
	print('Chassis repaired: ' + str(found_repaired))
	datestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '--')
	datestring = datestring.replace(':', '-')
	dataframe.to_csv('data_frame--' + datestring + '.csv', sep=';', index = False, index_label = False)
	

def addOneToFrame(directory):

	datafiles = []
	for item in listdir(directory): 
		if isfile(join(directory, item)):
			datafiles.append(directory + item)

	for datafile in datafiles:
		csv_data = pandas.read_csv(datafile, sep=";", index_col=False)

	for index, row in csv_data.iterrows():
		for y in range(1, 21):
			for x in range(1, 21):
				column = str(x) + '_' + str(y)
				csv_data.loc[index, column] = row[column] + 1
	
	datestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '--')
	datestring = datestring.replace(':', '-')
	csv_data.to_csv('data_frame_plus_one--' + datestring + '.csv', sep=';', index = False, index_label = False)
	
def addMultiFourToFrame(directory):

	datafiles = []
	for item in listdir(directory): 
		if isfile(join(directory, item)):
			datafiles.append(directory + item)

	for datafile in datafiles:
		csv_data = pandas.read_csv(datafile, sep=";", index_col=False)

	for index, row in csv_data.iterrows():
		if row['repaired'] == 1:
			for y in range(1, 21):
				for x in range(1, 21):
					column = str(x) + '_' + str(y)
					csv_data.loc[index, column] = row[column] * 8
	
	datestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '--')
	datestring = datestring.replace(':', '-')
	csv_data.to_csv('data_frame_times_eight--' + datestring + '.csv', sep=';', index = False, index_label = False)
	
	
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
		if str(row['repaired']) == '1':
			nr_of_repaired += 1
	
	print('Rows: ' + str(nr_of_rows))
	print('Valid: ' + str(nr_of_valid))
	print('Repaired: ' + str(nr_of_repaired))
	

#build_frameUpsample('Data2/V1/Org/')	
	
#build_frameV1('Data2/V1/Org/')
	
#analyse_frame('Data2/V1/Upsample3x/')	

#build_frame('Data2/Flatten/', 'Data/Labels/')	

#build_frameV3('Data2/V3/Org/')
	
#check_chassis('Data/...') #

#addOneToFrame('Data2/V1/Half_repaired/')

addMultiFourToFrame('Data2/V1/Half_repaired/')





