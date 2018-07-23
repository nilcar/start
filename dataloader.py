
import datetime
import math
import pandas
import numpy
from os import listdir
from os.path import isfile, join
import sys

from sklearn.model_selection import train_test_split
import tensorflow
from collections import OrderedDict
#from tensorflow.python.data import Dataset



"""
Returns three pandas dataframes randomously shuffled data with standard index
If compressed is True; data will be read from a csv file with already structured data.
If compressed is False; data will be read from csv source files and then structured
labelmapping should been done before outside this function and come as an filled dictionary
"""

def loadData(directory, compressed_data=False, label_mapping = []):

	truck_type = 'truck'
	truck_id = 'T_CHASSIS'
	truck_date = 'truck_date'
	index_tuple = (truck_type, truck_id, truck_date)
	
	resultfile = open("Results/model_statistics.txt", "a")
	
	if compressed_data:
		directory = 'Compressed/'
	
	CSV_COLUMN_NAMES = ['A', 'B','truck', 'T_CHASSIS', 'E', 'truck_date', 'G', 'H', 'I', 'J', 'x_index', 'L', 'M', 'N', 'y_index', 'value', 'Q', 'R', 'S']
	
	#print(len(CSV_COLUMN_NAMES))
	
	datafiles = []
	for item in listdir(directory): 
		if isfile(join(directory, item)):
			datafiles.append(directory + item)
	
	#print(datafiles)
	
	#print('Reading and merging cvs files, merging only if "compressed" is false')
	csv_data = pandas.DataFrame()
	nr = 1
	if not(compressed_data):		
		for datafile in datafiles:
			#print(datafile)
			csv_data_file = pandas.read_csv(datafile, sep=";", names=CSV_COLUMN_NAMES, header=None, index_col=False)
			#print(csv_data_file.head())
			csv_data = csv_data.append(csv_data_file, ignore_index=True)
			del csv_data_file
			print('appended cvsfile nr: ' + str(nr))
			nr += 1
	else:
		# Read already structured data
		csv_data = pandas.read_csv(datafiles[0], sep=";", index_col=False)
		# Remove data where truck_id isn't in label_mapping
		
	print('Csv data from file')
	print(csv_data.head(10))
	print('Shuffeling around the data randomously')
	csv_data = csv_data.reindex(numpy.random.permutation(csv_data.index))
	print('Size of read csv data:' + str(csv_data.size))
	
	print('Csv data after shuffeling')
	print(csv_data.head(10))
	
	if not(compressed_data):
		# Build the target dataframe structure
		dataframe = pandas.DataFrame()
		dataframe[truck_type] = []
		dataframe[truck_id] = []
		dataframe[truck_date] = []
		print('Building struture and fill it with data\n')
		for x in range(1, 21):
			for y in range(1, 21):
				dataframe[str(x) + '_' + str(y)] = []
			
		dataframe = dataframe.set_index(list(index_tuple))
	
		# fill the dataframe with data
		print('Inserting data...\n')
		accepted_labels = 0
		invalid_labels = 0
		for index, row in csv_data.iterrows():
			# Only insert data where truck_id is found in label_mapping
			index2 = row[truck_id]
			try:
				intlabel = label_mapping[index2] # Only to see if the label is possible to map
				index1 = row[truck_type]
				index3 = row[truck_date]
				column = str(row['x_index']) + '_' + str(row['y_index'])
				value = row['value']
				
				try: 
					# insert value if indexed row exist
					dataframe.loc[(index1, index2, index3), :].at[column] = value
					# Add engine and country here...
					
				except KeyError:
					dataframe.loc[(index1, index2, index3), :] = numpy.nan # Inserts a row with default NaN in each x,y column
					dataframe.loc[(index1, index2, index3), :].at[column] = value
					# Add engine and country here...
					
				accepted_labels += 1
			except KeyError:
				# Source label did not exist 
				invalid_labels += 1
				#print('Missing label not inserted: ' + index2)
	else:
		dataframe = csv_data.set_index(list(index_tuple))
		#print(dataframe.head())
		
	if not(compressed_data):	
		print('Total number of labels: ' + str(accepted_labels + invalid_labels))
		print('Number of accepted labels: ' + str(accepted_labels))
		print('Number of invalid labels: ' + str(invalid_labels))
	
	dataframe = dataframe.reset_index()
	
	print('Structured data')
	print(dataframe.head(10))
	print('Size of dataframe data with Nan:' + str(dataframe.size))
	
	# Check for rows with valid numbers of NaN...
	numbers_of_nan = 0 # Max 400
	#for index, row in dataframe.iterrows():		
		# Some looping... 
			
		# delete row that is invalid due to nr of NaN
		#dataframe.drop(index)
			
	# continue # Check nest row instead
	
	
	#print('Size of dataframe data after criteria for NaN exclusion(' + str(numbers_of_nan) +  ')' + str(dataframe.size))
	
	
	# This is the place to find the amount of NaN...
	nr_of_nan = 0
	nr_of_numbers = 0
	for index, row in dataframe.iterrows():
		for x in range(1, 21):
			for y in range(1, 21):
				if math.isnan(row[str(x) + '_' + str(y)]):
					nr_of_nan += 1
				else:
					nr_of_numbers += 1
	
	nan_percent = 100 * nr_of_nan / (nr_of_nan + nr_of_numbers)
	print('Numbers: ' + str(nr_of_numbers))
	print('Nan: ' + str(nr_of_nan))
	print('Nan in percent: ' + str(nan_percent))
	resultfile.write('\n\rNumbers: ' + str(nr_of_numbers))
	resultfile.write('\n\rNan: ' + str(nr_of_nan))
	resultfile.write('\n\rNan in percent: ' + str(nan_percent))
	
	if not(compressed_data):
		print('Saving frame data to csv file...\n')
		datestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '--')
		datestring = datestring.replace(':', '-')
		#print(dataframe.head())
		dataframe.to_csv('Compressed/volvo_frame--' + datestring + '.csv', sep=';', index = False, index_label = False)
	
	dataframe = dataframe.fillna(value = 0.0) # inplace = True
	print('After filling')
	print(dataframe.head(10))
	
	#trainset, testset = train_test_split(dataframe, test_size=0.4)
	#testset, validationset = train_test_split(testset, test_size=0.5)
	trainset, testset = train_test_split(dataframe, test_size=0.1)
	testset, validationset = train_test_split(testset, test_size=0.5)
	del dataframe
	
	"""
	print(trainset.head())
	print(trainset.size)
	print(testset.head())
	print(testset.size)
	print(validationset.head())
	print(validationset.size)
	"""
	
	return trainset, testset, validationset
	
def get_model_data(dataframe, label_mapping, choosen_label = 'T_CHASSIS'):

	# Clean up the dataframe to be converted into tensorflow datasets (features and labels)
	string_labels = dataframe.pop(choosen_label)
	dataframe = dataframe.loc[:, '1_1':'20_20']
		
	#print('string labels size:' + str(string_labels.size))
	#print('Dataframe for tensor slices')
	#print(dataframe.head(10))
			
	# Assumes initial label_mapping from label data file as input to this function
	next_index = len(label_mapping) # Assumes that label_mapping was built ordered from 0
	for label in string_labels:
		#print(label_mapping[label])
		try:
			intlabel = label_mapping[label] # Only to see if the label is possible to map
		except KeyError:
			print('Found missing label:  + label')
			label_mapping[label] = next_index
			next_index = next_index + 1

	#print('Length of label_mapping: ' + str(len(label_mapping)))
	#print(label_mapping)
			
	# Map all labels to integer representation
	int_labels = pandas.Series()
	for label in string_labels:
		#print(label_mapping[label])
		try:
			intlabel = label_mapping[label]
			new_label = pandas.Series([intlabel])
			#int_labels = pandas.concat([int_labels, new_label], ignore_index=True)
			int_labels = int_labels.append(new_label, ignore_index=True)
		except KeyError:
			print('Error... Missing label: ' + label)
				
			#print(new_label)
				
		
	#int_labels.reset_index()
	print('int labels size:' + str(int_labels.size))
	#print(int_labels)

	return dataframe, string_labels, label_mapping, int_labels
	
	
def train_input_fn(features, labels, batch_size, nr_epochs):
	"""An input function for training"""
	# Convert the inputs to a Dataset.
	dataset = tensorflow.data.Dataset.from_tensor_slices((dict(features), labels))

	# repeat, and batch the examples.
	dataset = dataset.repeat(nr_epochs).batch(batch_size)
	#ds = ds.batch(batch_size).repeat(num_epochs) # num_epochs ?
	
	# Return the dataset.
	return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tensorflow.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

	# Get label_mapping dictionary for choosen label
def get_valid_labels(directory, choosen_label = 'T_CHASSIS'):	
	
	# label mapping from labels file
	datafiles = []
	for item in listdir(directory):
		if isfile(join(directory, item)):
			datafiles.append(directory + item)
	print(datafiles)
	
	label_mapping = {}
	for datafile in datafiles:
		print(datafile)
		label_data = pandas.read_csv(datafile, sep=";")
		
		#print('label structure')
		#print(label_data.head(10))
		
		string_labels = label_data.pop(choosen_label)
		#print(string_labels)
		for index, label in string_labels.items():
			#print(index)
			try:
				intlabel = label_mapping[label] # Only to see if the label is possible to map
			except KeyError:
				#print('Found missing label:  + label')
				label_mapping[label] = index
	
	# Sorts the label_mapping by value (int representation)
	label_mapping = OrderedDict(sorted(label_mapping.items(), key=lambda x: x[1]))
	
	#print(label_mapping)		
	return	label_mapping
			
	
	# Adds labels to the structure as a new columns in the structured dataframe and saves it to a file
	# This combines the data in the "structured datasources" and the data in the labels file.
def add_labels_to_structure(structure_directory, labels_directory):

	#Get the existing structure
	structurefiles = []
	for item in listdir(structure_directory):
		if isfile(join(structure_directory, item)):
			structurefiles.append(structure_directory + item)
	for datafile in structurefiles:
		structured_data = pandas.read_csv(datafile, sep=";", index_col=False)

	#Get mapping from labels file
	labelfiles = []
	for item in listdir(labels_directory):
		if isfile(join(labels_directory, item)):
			labelfiles.append(labels_directory + item)
	for datafile in labelfiles:
		label_data = pandas.read_csv(datafile, sep=";", index_col=False)

		
	structured_data = structured_data.set_index('T_CHASSIS')
	label_data = label_data.set_index('T_CHASSIS')	

	for labelname in label_data.columns:
		if str(labelname) != 'T_CHASSIS':
			structured_data[labelname] = ''	
		# Loop through label_data and fill the structure with new label@ 'T_CHASSIS', labelnames
		for index_sd, row_sd in structured_data.iterrows():
			if labelname != 'T_CHASSIS':
				#value = label_data.loc[(index_sd, labelname)]
				structured_data.loc[(index_sd, labelname)] = label_data.loc[(index_sd, labelname)]
				# dataframe.loc[(index1, index2, index3), :] = numpy.nan
	
	structured_data = structured_data.reset_index()	
		
	"""
	# New columns with empty label values
	for labelname in label_data.columns:
		if str(labelname) != 'T_CHASSIS':
			structured_data[labelname] = ''
	
	# Loop through label_data and fill the structure with new label@ 'T_CHASSIS', labelnames
		for index_ld, row_ld in label_data.iterrows():
			for index_sd, row_sd in structured_data.iterrows():
				for labelname in label_data.columns:
					if str(labelname) != 'T_CHASSIS':
						if row_ld['T_CHASSIS'] == row_sd['T_CHASSIS']:
							row_sd[labelname] = row_ld[labelname]
	"""						
							
	
	print(structured_data.head(10))
	
	print('Saving frame data to csv file...\n')
	datestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '--')
	datestring = datestring.replace(':', '-')
	structured_data.to_csv('Compressed/new_volvo_frame_all_labels--' + datestring + '.csv', sep=';', index = False, index_label = False)
	
	return structured_data

	
#get_data_source_labels('Data_original/')		
#get_valid_labels('Labels/')

#loadData('Testdata/', False)
#loadData('Data_original/', False)
#loadData('Compressed/', True)

#add_labels_to_structure('Compressed/Compressed_single/', 'Labels/')
#add_labels_to_structure('Compressed/Compressed_valid_chassis/', 'Labels/')

