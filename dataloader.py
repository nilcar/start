

import pandas
import numpy
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
import tensorflow
#from tensorflow.python.data import Dataset

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
		csv_data_file = pandas.read_csv(datafile, sep=";", names=CSV_COLUMN_NAMES, header=None)
		csv_data = csv_data.append(csv_data_file)
	
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

	#print(dataframe.head())
	
	"""
	# DEBUG...
	for index, row in dataframe.iterrows():
		for x in range(20):
			for y in range(20):
				column = str(x) + '_' + str(y)
				if row[column] != 0.0:
					print(row[column])
	"""
	
	
	
	trainset, testset = train_test_split(dataframe, test_size=0.2)
	testset, validationset = train_test_split(testset, test_size=0.5)
	"""
	print(trainset.head())
	print(trainset.size)
	print(testset.head())
	print(testset.size)
	print(validationset.head())
	print(validationset.size)
	#DataFrame.to_csv(path)
	"""
	
	return trainset, testset, validationset
	
def get_model_data(dataframe, label_mapping):

	# Clean up the dataframe to be converted into tensorflow datasets
	dataframe = dataframe.reset_index()
	dataframe.pop('truck_type')
	dataframe.pop('truck_date')
	string_labels = dataframe.pop('truck_id')
	print('string labels size:' + str(string_labels.size))
	
	# Holds mapping table: lable to int representation. Also holds number of unique truck_id's
	
	# label mapping from data source files
	#label_mapping_new = dict([(y,x+1) for x,y in enumerate(sorted(set(string_labels.tolist())))])
	
	# Assumes initial label_mapping from label data file, not necessary?
	next_index = len(label_mapping) # Assumes that label_mapping was built ordered from 0
	for label in string_labels:
		#print(label_mapping[label])
		try:
			intlabel = label_mapping[label] # Only to see if the label is possible to map
		except KeyError:
			#print('Missing label: ' + label + ' new index: ' + str(next_index))
			label_mapping[label] = next_index
			next_index = next_index + 1

	print('Length of label_mapping: ' + str(len(label_mapping)))
	#print(label_mapping)
	
	# Map all labels to integer representation
	int_labels = pandas.Series()
	for label in string_labels:
		#print(label_mapping[label])
		try:
			intlabel = label_mapping[label]
			new_label = pandas.Series([intlabel])
			int_labels = pandas.concat([int_labels, new_label], ignore_index=True)
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
	


















