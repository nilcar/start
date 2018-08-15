

import datetime
import math
import pandas
import numpy
from os import listdir
from os.path import isfile, join
import sys
import collections



def getDoubleFrequency(directory):

	truck_type = 'truck'
	truck_id = 'T_CHASSIS'
	truck_date = 'truck_date'
	index_tuple = (truck_type, truck_id, truck_date)
	
	doubles_dict = {}
	found_doubles = {}
	I_doubles = {}
	M_doubles = {}
	
	CSV_COLUMN_NAMES = ['A', 'B','truck', 'T_CHASSIS', 'E', 'truck_date', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'value', 'Q', 'R', 'S']
	
	#print(len(CSV_COLUMN_NAMES))
	
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
			#column = str(row['x_index']) + '_' + str(row['y_index'])
			#value = row['value']
	
			# Look for double entries in the source data
			key = index1 + ':' +  index2 + ':' +  index3 + '#' +  str(row['I']) + ':' +  str(row['J'])  + ':' +  str(row['K'])  + ':' +  str(row['M'])  + ':' +  str(row['N'])  + ':' +  str(row['O'])
			try:
				valuefirst = doubles_dict[key]
				valuefirst += 1
				found_doubles[key] = valuefirst
				I_doubles[key] = str(row['I'])
				M_doubles[key] = str(row['M'])
				
			except KeyError:
				doubles_dict[key] = 1
			row_nr += 1
				
				
	doublesframe = pandas.DataFrame()
	doublesframe['Key'] = found_doubles.keys()
	doublesframe['Frequency'] = found_doubles.values()
	doublesframe['I'] = I_doubles.values()
	doublesframe['M'] = M_doubles.values()
	
	#print(doublesframe.head())
	#index_tuple = ('Frequency', 'I', 'M')
	#doublesframe = doublesframe.set_index(list(index_tuple))
	
	doublesframe = doublesframe.sort_values(['Frequency', 'I', 'M'], ascending = [False,False,False])
	#doublesframe = doublesframe.sort_index(level=['Frequency', 'I', 'M'], ascending = [False,False,False])
	#doublesframe = doublesframe.sort_values(by=['Frequency', 'I', 'M'], ascending = [False,False,False])
	
	
	#doublesframe = doublesframe.reset_index()
	
	#print(doublesframe.head())
	
	datestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '--')
	datestring = datestring.replace(':', '-')
	doublesframe.to_csv('volvo_doubles_different_columnvalues' + datestring + '.csv', sep=';', index = False, index_label = False)	

	
	
getDoubleFrequency('Testdata/') # 'Data_original/' 'Testdata/'
	
	
	
	
	

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	