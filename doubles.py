

import datetime
import math
import pandas
import numpy
from os import listdir
from os.path import isfile, join
import sys



def get_doubles(directory):

	truck_type = 'truck'
	truck_id = 'T_CHASSIS'
	truck_date = 'truck_date'
	index_tuple = (truck_type, truck_id, truck_date)
	
	CSV_COLUMN_NAMES = ['A', 'B','truck', 'T_CHASSIS', 'E', 'truck_date', 'G', 'H', 'I', 'J', 'x_index', 'L', 'M', 'N', 'y_index', 'value', 'Q', 'R', 'S']
	
	doubles_dict = {}
	found_doubles = {}
	
	datafiles = []
	for item in listdir(directory): 
		if isfile(join(directory, item)):
			datafiles.append(directory + item)
	
	for datafile in datafiles:
		row_nr = 1
		csv_data = pandas.read_csv(datafile, sep=";", names=CSV_COLUMN_NAMES, header=None, index_col=False)
		for index, row in csv_data.iterrows():
			# Only insert data where truck_id is found in label_mapping
			index2 = row[truck_id]
			index1 = row[truck_type]
			index3 = row[truck_date]
			column = str(row['x_index']) + '_' + str(row['y_index'])
			value = row['value']
	
			# Look for double entries in the source data
			key = index1 + ':' +  index2 + ':' +  index3 + '#' +  column
			try:
				valuefirst = doubles_dict[key]
				org_value, org_row, orgfile = valuefirst.split(':')
				if org_value != str(value):
					found_doubles[key] = valuefirst + '#' + str(value) + ':' + str(row_nr) + ':' + datafile
			except KeyError:
				doubles_dict[key] = str(value) + ':' + str(row_nr) + ':' + datafile
			row_nr += 1
				
	doublesframe = pandas.DataFrame()
	doublesframe['Key'] = found_doubles.keys()
	doublesframe['first#double'] = found_doubles.values()
	datestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '--')
	datestring = datestring.replace(':', '-')
	doublesframe.to_csv('frame_doubles_different_values' + datestring + '.csv', sep=';', index = False, index_label = False)	


get_doubles('Data_original/') # 'Testdata/' 'Data_original/'


