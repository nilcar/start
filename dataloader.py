
import datetime
import pandas
import numpy
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
import tensorflow
#from tensorflow.python.data import Dataset

"""
Returns three pandas dataframes randomously shuffled data with standard index
If compressed is True; data will be read from a csv file with already structured data.
If compressed is False; data will be read from csv source files and then structured
"""

def loadData(directory, compressed_data=False):

	truck_type = 'truck_type'
	truck_id = 'truck_id'
	truck_date = 'truck_date'
	index_tuple = (truck_type, truck_id, truck_date)
	dataframe = pandas.DataFrame()
	dataframe[truck_type] = []
	dataframe[truck_id] = []
	dataframe[truck_date] = []
	
	
	
	if compressed_data:
		directory = 'Compressed/'
	
	CSV_COLUMN_NAMES = ['A', 'B','truck_type', 'truck_id', 'E', 'truck_date', 'G', 'H', 'I', 'J', 'x_index', 'L', 'M', 'N', 'y_index', 'value', 'Q', 'R', 'S']
	
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
			#print('appended cvsfile nr: ' + str(nr))
			nr += 1
	else:
		# Read already structured data
		csv_data = pandas.read_csv(datafiles[0], sep=";", index_col=False) #.drop(['unnamed 0'],axis=1)
	
	print('Csv data from file')
	print(csv_data.head())
	
	print('Shuffeling around the data randomously')
	csv_data = csv_data.reindex(numpy.random.permutation(csv_data.index))
	print('Size of read csv data:' + str(csv_data.size))
	
	if not(compressed_data):
		# Build the target dataframe structure
		print('Building struture and fill it with data\n')
		for x in range(20):
			for y in range(20):
				dataframe[str(x) + '_' + str(y)] = []
			
		dataframe = dataframe.set_index(list(index_tuple))
	
		# fill the dataframe with data
		print('Inserting data...\n')
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
				dataframe.loc[(index1, index2, index3), :] = 0.0 #numpy.nan # Inserts a row with default NaN in each x,y column
				dataframe.loc[(index1, index2, index3), :].at[column] = value
	else:
		csv_data = csv_data.set_index(list(index_tuple))
		dataframe = csv_data
		# dataframe.fillna(value = 0.0)
		
		
	dataframe = dataframe.reset_index()
	print('Structured data')
	print(dataframe.head())
	print('Size of dataframe data:' + str(dataframe.size))
	
	print('Saving frame data to csv file...\n')
	if not(compressed_data):
		datestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '--')
		datestring = datestring.replace(':', '-')
		#dataframe.reset_index(drop = True, inplace = True)
		#print(dataframe.head())
		dataframe.to_csv('Compressed/volvo_frame' + datestring + '.csv', sep=';', index = False, index_label = False)
	
	
	# DEBUG...
	"""
	for index, row in dataframe.iterrows():
		for x in range(20):
			for y in range(20):
				column = str(x) + '_' + str(y)
				if row[column] != numpy.nan:
					print(int(row[column]))
	"""
	
	#print('Describe')
	#dataframe.describe()
	
	trainset, testset = train_test_split(dataframe, test_size=0.2)
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
	
def get_model_data(dataframe, label_mapping):

	# Clean up the dataframe to be converted into tensorflow datasets
	#dataframe = dataframe.reset_index()
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
	

#loadData('Testdata/', False)
#loadData('Data_original/', False)
#loadData('Compressed/', True)
















