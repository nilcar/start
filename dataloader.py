

import pandas
import numpy
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split


def loadData(directory):

	truck_type = 'truck_type'
	truck_id = 'truck_id'
	truck_date = 'truck_date'
	index_tuple = (truck_type, truck_id, truck_date)
	dataframe = pandas.DataFrame()
	dataframe[truck_type] = []
	dataframe[truck_id] = []
	dataframe[truck_date] = []
	
	CSV_COLUMN_NAMES = ['A', 'B','truck_type', 'truck_id', 'E', 'truck_date', 'G', 'H', 'I', 'J', 'x_index', 'L', 'M', 'N', 'y_index', 'value', 'Q', 'R', 'S']
	
	datafiles = []
	for item in listdir(directory): 
		if isfile(join(directory, item)):
			datafiles.append(directory + item)
	
	print(datafiles)
	
	csv_data = pandas.DataFrame()
	for datafile in datafiles:
		print(datafile)
		csv_data = pandas.read_csv(datafile, sep=";", names=CSV_COLUMN_NAMES, header=None)
		csv_data.append(csv_data)
	
	print(csv_data.head())
	
	csv_data = csv_data.reindex(numpy.random.permutation(csv_data.index))
	
	# Build the target dataframe structure
	for x in range(20):
		for y in range(20):
			dataframe[str(x) + '_' + str(y)] = []
			
	dataframe = dataframe.set_index(list(index_tuple))
	
	# fill the dataframe with data
	for index, row in csv_data.iterrows():
		index1 = row[truck_type]
		index2 = row[truck_id]
		index3 = row[truck_date]
		column = str(row['x_index']) + '_' + str(row['y_index'])
		value = row['value']
		
		try: 
			# insert value if indexed row exist
			dataframe.loc[(index1, index2, index3), :].at[column] = value
		
		except KeyError:
			dataframe.loc[(index1, index2, index3), :] = 0.0 # Inserts a row with default 0.0 in each column
			dataframe.loc[(index1, index2, index3), :].at[column] = value

	print(dataframe.head())
	
	
	# DEBUG...
	"""
	for index, row in dataframe.iterrows():
		for x in range(20):
			for y in range(20):
				column = str(x) + '_' + str(y)
				if row[column] != 0.0:
					print(row[column])
	"""
	
	
	
	trainset, testset = train_test_split(dataframe, test_size=0.2)
	testset, validationset = train_test_split(testset, test_size=0.5)
	print(trainset.head())
	print(trainset.size)
	print(testset.head())
	print(testset.size)
	print(validationset.head())
	print(validationset.size)
	#DataFrame.to_csv(path)
	
	
	
	
	
	return trainset, testset, validationset

	
loadData('Test/')	
	
#print(loadData('Test/'))


#print(df.values) # as np array without indexes
#print(pandas.DataFrame(df.values)) #pandas ndarray to dataframe with integer index, columns as integers.
















