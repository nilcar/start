
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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import itertools
#import mysql.connector
#from tensorflow.python.data import Dataset


def loadValidationFrameV2(directory):

	dataframe = pandas.DataFrame()
	datafiles = []
	for item in listdir(directory): 
		if isfile(join(directory, item)):
			datafiles.append(directory + item)

	for datafile in datafiles:
		print('Reading compressed: ' + datafile)
		dataframe = pandas.read_csv(datafile, sep=";", index_col=False)

	dataframe = dataframe.fillna(value = 0.0)	
	print(dataframe.head())
	
	return dataframe


def loadValidationFrameV3(directory):

	dataframe = pandas.DataFrame()
	datafiles = []
	for item in listdir(directory): 
		if isfile(join(directory, item)):
			datafiles.append(directory + item)

	for datafile in datafiles:
		print('Reading compressed: ' + datafile)
		dataframe = pandas.read_csv(datafile, sep=";", index_col=False)

	labelmaskvalidate = (dataframe['PARTITIONNING'] == '3_Validation')
	dataframe = dataframe.loc[labelmaskvalidate]

	dataframe = dataframe.fillna(value = 0.0)	
	print(dataframe.head())
	
	return dataframe
	
def loadTrainTestFrameV3(directory):

	dataframe = pandas.DataFrame()
	trainframe = pandas.DataFrame()
	testframe = pandas.DataFrame()
	datafiles = []
	for item in listdir(directory): 
		if isfile(join(directory, item)):
			datafiles.append(directory + item)

	for datafile in datafiles:
		print('Reading compressed: ' + datafile)
		dataframe = pandas.read_csv(datafile, sep=";", index_col=False)

	labelmasktrain = (dataframe['PARTITIONNING'] == '1_Training')
	trainframe = dataframe.loc[labelmasktrain]
	trainframe = trainframe.fillna(value = 0.0)
	
	labelmasktest = (dataframe['PARTITIONNING'] == '2_Testing')
	testframe = dataframe.loc[labelmasktest]
	testframe = testframe.fillna(value = 0.0)
	
	
	return trainframe, testframe
	

def excludeChassis (dataframe, directory):

	datafiles = []
	for item in listdir(directory): 
		if isfile(join(directory, item)):
			datafiles.append(directory + item)	
	
	for datafile in datafiles:
		highrisk_chassis = pandas.read_csv(datafile, sep=";", index_col=False)
	
	highrisk = {}
	for index, row in highrisk_chassis.iterrows():
		highrisk[row['T_CHASSIS']] = row['T_CHASSIS']
	
	delete_rows = []
	number_of_rows = 0
	for index, row in dataframe.iterrows():
		number_of_rows += 1
		try:
			chassi = highrisk[row['T_CHASSIS']]
		except KeyError:
			delete_rows.append(index)
		
	print('Before delete: ' + str(dataframe.size))
	dataframe = dataframe.drop(delete_rows)
	print('Number of validation rows:' + str(number_of_rows))
	print('Deleted rows:' + str(len(delete_rows)))
	print('After delete: ' + str(dataframe.size))
	
	#sys.exit()
	
	return dataframe
	
"""
Returns three pandas dataframes randomously shuffled data with standard index
Rows can be excluded by allowing a max number of NaN
Fixed selection handles reading of dataframe differently
"""

def loadData(directory, max_nr_of_nan = 0, fixed_selection = True, file_suffix = ''):

	
	dataframe = pandas.DataFrame()
	resultfile = open("Results/model_statistics.txt", "w")
	
	
	datafiles = []
	for item in listdir(directory): 
		if isfile(join(directory, item)):
			datafiles.append(directory + item)
	
	#print(datafiles)
	
	#print('Reading and merging cvs files, merging only if "compressed" is false')
	csv_data = pandas.DataFrame()
	
	# Read already structured data
	for datafile in datafiles:
		print('Reading compressed: ' + datafile)
		csv_data = pandas.read_csv(datafile, sep=";", index_col=False)

		
	print('Csv data from file')
	print(csv_data.head())
	print('Shuffeling around the data randomously')
	csv_data = csv_data.reindex(numpy.random.permutation(csv_data.index))
	print('Size of read csv data:' + str(csv_data.size))
	
	print('Csv data after shuffeling')
	print(csv_data.head())
	
	#dataframe = csv_data.set_index(list(index_tuple))
	dataframe = csv_data
	#print(dataframe.head())
	
	print('Structured data')
	print(dataframe.head())
	
	print('Size of dataframe data with Nan:' + str(dataframe.size))
	
	#dataframe = nan_statistics(dataframe)
	
	dataframe = exclude_rows_with_nan(dataframe, max_nr_of_nan)
	print('Size of dataframe where Nan rows excluded:' + str(dataframe.size))
	
	dataframe = dataframe.fillna(value = 0.0)
	print('After filling')
	print(dataframe.head())
	
	dataframe, first_column, last_column = excludeZerocolumns(dataframe, 1.0, file_suffix)
	
	if fixed_selection:
	
		## For V3
		labelmasktrain = (dataframe['PARTITIONNING'] == '1_Training')
		trainset = dataframe.loc[labelmasktrain]
		labelmasktest = (dataframe['PARTITIONNING'] == '2_Testing')
		testset = dataframe.loc[labelmasktest]
		labelmaskvalidate = (dataframe['PARTITIONNING'] == '3_Validation')
		validationset = dataframe.loc[labelmaskvalidate]
	
	else:
	
		#V1
		
		trainset, testset = train_test_split(dataframe, test_size=0.4)
		testset, validationset = train_test_split(testset, test_size=0.5)
		
		#validationset = excludeChassis (validationset, 'Data2/Highrisk_chassis/')
		
		
		#V2 validation
		"""
		validationset = loadValidationFrameV2('Data2/ReducedV2/')
		validationset = excludeChassis (validationset, 'Data2/Highrisk_chassis/')
		"""
		
		
		
		
		#V2 only
		#trainset, testset, validationset = unique_selection(dataframe)
		
		# V1 train/test V3 validate
		"""
		trainset, testset = train_test_split(dataframe, test_size=0.4)
		testset, validationset = train_test_split(testset, test_size=0.5)
		validationset = loadValidationFrameV3('Data2/V3/')
		validationset = excludeChassis (validationset, 'Data2/Highrisk_chassis/')
		"""
		
		# V3 train/test V1 validate
		"""
		trainset, testset = train_test_split(dataframe, test_size=0.4)
		testset, validationset = train_test_split(testset, test_size=0.5)
		trainset, testset = loadTrainTestFrameV3('Data2/V3/')
		"""
		
	del dataframe
	
	print('Trainset')
	print(trainset.head())
	"""
	print(trainset.size)
	#print(testset.head())
	print(testset.size)
	#print(validationset.head())
	print(validationset.size)
	"""

	resultfile.close()
	
	return trainset, testset, validationset, first_column, last_column
	
def excludeZerocolumns(dataframe, percentage, file_suffix):

	print('Size before Zero deletions: ' + str(dataframe.size))
	resultfile = open("Results/model_results" + file_suffix + ".txt", "a")

	values = pandas.Series()
	number_zeros = 0
	number_values = 0
	delete_columns = []
	first = True
	first_column = ''
	last_column = ''
	
	for x in range(1, 21):
		for y in range(1, 21):
			column = str(x) + '_' + str(y)
			values = pandas.Series(dataframe.loc[:,column])
			number_zeros = 0
			number_values = 0
			for value in values:
				number_values += 1
				if value == 0.0:
					number_zeros += 1
			if 	(number_zeros / number_values) > percentage:
				delete_columns.append(column)
			else:
				if first:
					first = False
					first_column = column
				last_column = column
					
	print('First column: ' + first_column)
	print('Last column: ' + last_column)
					
	for column in delete_columns:
		dataframe.pop(column)
	
	resultfile.write('Number of column deletions: ' + str(len(delete_columns)) + '\n\r')
	print('Number of column deletions: ' + str(len(delete_columns)))
	print('Size after Zero deletions: ' + str(dataframe.size))
	
	return dataframe, first_column, last_column
	
	
	
def get_model_data(dataframe, label_mapping, choosen_label = 'T_CHASSIS', first_column = '1_1', last_column = '20_20'):

	# Clean up the dataframe to be converted into tensorflow datasets (features and labels)
	string_labels = dataframe.pop(choosen_label)
	
	print('CAME here')
	
	dataframe = dataframe.loc[:, first_column:last_column] # loc[:, '1_1':'20_20']

	
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
	
	version_full = tensorflow.__version__
	x, version, y = version_full.split('.')
	print('Versionfull: ' + version_full)
	print('Version: ' + version)
	
	if version >= '5':
		# Return the dataset.
		return dataset
	else:
		return dataset.make_one_shot_iterator().get_next() #for 1.4


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
	
	version_full = tensorflow.__version__
	x, version, y = version_full.split('.')
	print('Versionfull: ' + version_full)
	print('Version: ' + version)
	
	if version >= '5':
		# Return the dataset.
		return dataset
	else:
		return dataset.make_one_shot_iterator().get_next() #for 1.4

		
def getFoldFrame(foldframe_list, kfolds, foldframe):

	if kfolds == 1:
		foldframe_list.append(foldframe)
	else:
		trainframe, restframe = train_test_split(foldframe, train_size= (1 / kfolds))
		foldframe_list.append(trainframe)
		getFoldFrame(foldframe_list, kfolds - 1, restframe)

	return 	foldframe_list
		
		

def getFoldTrainFrames(foldframe_list, testindex):
	
	foldtrainframe = pandas.DataFrame()
	foldtestframe = pandas.DataFrame()
	
	for index in range(len(foldframe_list)):
		if index == testindex:
			foldtestframe = foldtestframe.append(foldframe_list[index])
		else:
			foldtrainframe = foldtrainframe.append(foldframe_list[index])
		
	return foldtrainframe, foldtestframe

	
		
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
		label_data = pandas.read_csv(datafile, sep=";", keep_default_na=False)
		
		#print('label structure')
		print(label_data.head())
		int_representation = 0
		string_labels = label_data.pop(choosen_label)
		#print(string_labels)
		for index, label in string_labels.items():
			#print(index)
			if label == '':
				label = 'Missing_data'
			try:
				intlabel = label_mapping[label] # Only to see if the label is possible to map
			except KeyError:
				#print('Found missing label:  + label')
				label_mapping[label] = int_representation
				int_representation += 1
	
	# Sorts the label_mapping by value (int representation)
	#label_mapping = OrderedDict(sorted(label_mapping.items(), key=lambda x: x[1]))
	
	print(label_mapping)		
	return	label_mapping
			
	
	
def exclude_rows_with_nan(dataframe, max_nr_of_nan = 0):

	#Check for rows with invalid numbers of NaN and delete them...
	
	if max_nr_of_nan == 0:
		return dataframe
	
	nr_of_rows = 0
	nr_of_rows_deleted = 0
	delete_rows = []
	for index, row in dataframe.iterrows():		
		nr_of_nan = 0
		for x in range(1, 21):
			for y in range(1, 21):
				value = row[str(x) + '_' + str(y)]
				if math.isnan(value):
					nr_of_nan += 1
		if nr_of_nan > max_nr_of_nan and max_nr_of_nan > 0:
			#print(str(nr_of_nan) + ' Index: ' + str(index))
			nr_of_rows_deleted += 1
			delete_rows.append(index)
			#print('Appending: ' + str(index))
		nr_of_rows += 1	
			
	#print(delete_rows)
	if nr_of_rows_deleted > 0:
		dataframe = dataframe.drop(delete_rows)
	print('Number of deleted rows: ' + str(nr_of_rows_deleted) + ' of rows total: ' + str(nr_of_rows))
	resultfile = open("Results/model_results.txt", "a")
	resultfile.write('\n\rNumber of deleted rows: ' + str(nr_of_rows_deleted) + ' of rows total: ' + str(nr_of_rows))
	
	return dataframe


def nan_statistics(dataframe):

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
	resultfile = open("Results/model_statistics.txt", "a")
	resultfile.write('\n\rNumbers: ' + str(nr_of_numbers))
	resultfile.write('\n\rNan: ' + str(nr_of_nan))
	resultfile.write('\n\rNan in percent: ' + str(nan_percent))	
	
	return dataframe
	
	
	
	
	
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
		label_data = pandas.read_csv(datafile, sep=";", index_col=False, keep_default_na=False)

		
	structured_data = structured_data.set_index('T_CHASSIS')
	label_data = label_data.set_index('T_CHASSIS')	

	for labelname in label_data.columns:
		if str(labelname) != 'T_CHASSIS':
			structured_data[labelname] = ''	
		# Loop through label_data and fill the structure with new label@ 'T_CHASSIS', labelnames
		for index_sd, row_sd in structured_data.iterrows():
			if labelname != 'T_CHASSIS':
				value = label_data.loc[(index_sd, labelname)]
				if value == '':
					value = 'Missing_data'
				structured_data.loc[(index_sd, labelname)] = str(value)
	
	structured_data = structured_data.reset_index()			
							
	print(structured_data.head(10))
	
	print('Saving frame data to csv file...\n')
	datestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '--')
	datestring = datestring.replace(':', '-')
	structured_data.to_csv('Compressed/new_volvo_frame_all_labels--' + datestring + '.csv', sep=';', index = False, index_label = False)
	
	return structured_data

def print_cm(confusion_matrix, labels, filesuffix):	
	
	if len(labels) == 2:
		classesy = ['Healthy', 'Not healthy']
		classesx = ['Healthy', 'Not healthy']
	else:
		classesy = labels
		classesx = labels

	plt.figure()
	plt.imshow(confusion_matrix, cmap=plt.cm.Blues) # origin='lower' interpolation='nearest'
	plt.colorbar()
	plt.title("Confusion Matrix")
	tick_marks = numpy.arange(len(labels)) #numpy.arange(2)
	plt.xticks(tick_marks, classesx, rotation=45)
	plt.yticks(tick_marks, classesy)
	thresh = confusion_matrix.max() / 2
	for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
		plt.text(j, i, format(confusion_matrix[i, j], 'd'), horizontalalignment="center", color="white" if confusion_matrix[i, j] > thresh else "black")
	plt.tight_layout()
	plt.xlabel('Predicted')
	plt.ylabel('True')
	#plt.show()
	plt.savefig('Results/Confusion-matrix-' + filesuffix + '.png')
	plt.clf()
	
	
def analyse_frame(dataframe):	
	
	resultstring = ''
	nr_of_rows = 0
	nr_of_repaired = 0
	nr_of_valid = 0
	for index, row in dataframe.iterrows():
		nr_of_rows += 1
		try:
			if row['valid'] == 1:
				nr_of_valid += 1
		except:
			None
		if str(row['repaired']) == '1':
			nr_of_repaired += 1
	
	
	resultstring += '\n\rRows: ' + str(nr_of_rows)
	resultstring += '\n\rValid: ' + str(nr_of_valid)
	resultstring += '\n\rRepaired: ' + str(nr_of_repaired)
	
	"""
	print('Rows: ' + str(nr_of_rows))
	print('Valid: ' + str(nr_of_valid))
	print('Repaired: ' + str(nr_of_repaired))
	"""
	
	return resultstring

def unique_selection(dataframe):

	train = {}
	test = {}
	validate = {}
	selector = 0
	df_columns = dataframe.columns
	trainframe = pandas.DataFrame([], columns=df_columns)
	testframe = pandas.DataFrame([], columns=df_columns)
	validateframe = pandas.DataFrame([], columns=df_columns)
	tempframe = pandas.DataFrame()
	rownr = 0
	

	for index, row in dataframe.iterrows():
		if selector >= 3:
			selector = 0

		rownr += 1
		print('Row:' + str(rownr))
			
		try:
			value = train[row['T_CHASSIS']]
			tempframe = pandas.DataFrame([row.values], columns=df_columns)
			trainframe = trainframe.append(tempframe, ignore_index=True)
			continue
		except:
			None
	
		try:
			value = test[row['T_CHASSIS']]
			tempframe = pandas.DataFrame([row.values], columns=df_columns)
			testframe = testframe.append(tempframe, ignore_index=True)
			continue
		except:
			None
			
		try:
			value = validate[row['T_CHASSIS']]
			tempframe = pandas.DataFrame([row.values], columns=df_columns)
			validateframe = validateframe.append(tempframe, ignore_index=True)
			continue
		except:
			None
	
		if selector == 0:
			train[row['T_CHASSIS']] = 1
			tempframe = pandas.DataFrame([row.values], columns=df_columns)
			trainframe = trainframe.append(tempframe, ignore_index=True)
	
		if selector == 1:
			test[row['T_CHASSIS']] = 1
			tempframe = pandas.DataFrame([row.values], columns=df_columns)
			testframe = testframe.append(tempframe, ignore_index=True)
			
		if selector == 2:
			validate[row['T_CHASSIS']] = 1
			tempframe = pandas.DataFrame([row.values], columns=df_columns)
			validateframe = validateframe.append(tempframe, ignore_index=True)
	
		selector += 1
	
	return trainframe, testframe, validateframe
	
def print_roc_curve(y_true, y_prob, filesuffix):

	fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
	roc_auc = auc(fpr, tpr)
	fpr_, tpr_, _ = roc_curve(y_true, y_prob, pos_label=0)
	roc_auc_ = auc(fpr_, tpr_)
	
	plt.figure()
	plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve unhealthy (area = %0.2f)' % roc_auc)
	#plt.plot(fpr_, tpr_, color='red', lw=2, label='ROC curve healthy (area = %0.2f)' % roc_auc_)
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.savefig('Results/ROC_curve-' + filesuffix + '.png')

	
	
#get_data_source_labels('Data_original/')		
#get_valid_labels('Labels/')

#loadData('Testdata/', False)
#loadData('Data_original/', False)
#loadData('Compressed/', True)

#add_labels_to_structure('Compressed/', 'Labels/')
#add_labels_to_structure('Compressed/Compressed_valid_chassis/', 'Labels/')

