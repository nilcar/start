
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


"""
Description:
Defines a dataframe for validation from specified directory
Input:
directory: The directory holding the csv file(s)

Output:
A dataframe holding the selected validation data

Others:
The read csv file(s) must contain columns 1_1 .. 20_20 and a column PARTITIONNING where the value 3_Validation is
set for all rows to be selected for validation. 

"""

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

"""
Description:
Defines a dataframe for train and test from specified directory
Input:
directory: The directory holding the csv file(s)

Output:
Two dataframes holding the selected train and test data

Others:
The read csv file(s) must contain columns 1_1 .. 20_20 and a column PARTITIONNING where the value 1_Training and 2_Testing is
set for all rows to be selected for the dataset respectively.

"""	

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
	
"""
Description:
This function can be used to exclude non-high risk chassis

Input:
A dataframe with training data
Directory where file with high risk chassis are defined

Output:
A dataframe with non-high risk chassi samples deleted

Others:

"""

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
	
	return dataframe
	

"""
Description:
Returns three pandas dataframes randomously shuffled data with standard index

Input:
Directory holding one csv file with data
Fixed selection handles reading of dataframe differently and selects sets for training, test and validation differently
File_suffix for log files

Output:
Three datasets for training , test and validation
Also the name of the first and last datacolumns

Others:

"""

def loadData(directory, fixed_selection = True, file_suffix = ''):

	dataframe = pandas.DataFrame()
	resultfile = open("Results/model_statistics.txt", "w")
	
	
	datafiles = []
	for item in listdir(directory): 
		if isfile(join(directory, item)):
			datafiles.append(directory + item)
	
	# Read already structured data
	for datafile in datafiles:
		print('Reading compressed: ' + datafile)
		dataframe = pandas.read_csv(datafile, sep=";", index_col=False)

	print('Csv data from file')
	print(dataframe.head())
	print('Shuffeling around the data randomously')
	dataframe = dataframe.reindex(numpy.random.permutation(dataframe.index))
	print('Size of read csv data:' + str(dataframe.size))
	
	print('Structured randomised data')
	print(dataframe.head())
	
	dataframe = dataframe.fillna(value = 0.0)
	#print('After NaN filling')
	#print(dataframe.head())
	
	# value 1.0 means no exclusion.
	dataframe, first_column, last_column = excludeZerocolumns(dataframe, 1.0, file_suffix)
	
	if fixed_selection:
	
		## For V3 date selection
		labelmaskvalidate = ((dataframe['Send_Date'] >= '2017-08-15') & (dataframe['Send_Date'] <= '2018-02-15'))
		traintestset = dataframe.loc[labelmaskvalidate]
		trainset, testset = train_test_split(traintestset, train_size=0.7)
		labelmaskvalidate = ((dataframe['Send_Date'] >= '2018-05-15') & (dataframe['Send_Date'] <= '2018-08-15'))
		validationset = dataframe.loc[labelmaskvalidate]
		
		## For V3 org
		"""
		labelmasktrain = (dataframe['PARTITIONNING'] == '1_Training')
		trainset = dataframe.loc[labelmasktrain]
		labelmasktest = (dataframe['PARTITIONNING'] == '2_Testing')
		testset = dataframe.loc[labelmasktest]
		labelmaskvalidate = (dataframe['PARTITIONNING'] == '3_Validation')
		validationset = dataframe.loc[labelmaskvalidate]
		"""
		#validationset = excludeChassis (validationset, 'Data2/Highrisk_chassis/')
	
	else:
	
		#V1
		trainset, testset = train_test_split(dataframe, test_size=0.4)
		testset, validationset = train_test_split(testset, test_size=0.5)
		#validationset = excludeChassis (validationset, 'Data2/Highrisk_chassis/')
		
		# V1 train/test V3 validate
		"""
		trainset, testset = train_test_split(dataframe, test_size=0.4)
		testset, validationset = train_test_split(testset, test_size=0.5)
		validationset = loadValidationFrameV3('Data2/V3/')
		#validationset = excludeChassis (validationset, 'Data2/Highrisk_chassis/')
		"""
		
		# V3 train/test V1 validate
		"""
		trainset, testset = train_test_split(dataframe, test_size=0.4)
		testset, validationset = train_test_split(testset, test_size=0.5)
		trainset, testset = loadTrainTestFrameV3('Data2/V3/')
		#validationset = excludeChassis (validationset, 'Data2/Highrisk_chassis/')
		"""
		
	del dataframe
	resultfile.close()
	
	return trainset, testset, validationset, first_column, last_column

"""
Description:
A function to be used for exclusion columns with a certain amoun of xero values

Input:
The dataframe to check
The percentage for exceeding zero numbers, 1.0 nothin will be excluded

Output:
The dataframe cleaned from zero columns
The first and last column names that is left

Others:

"""	
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
	
	
"""
Description:
Separates data from labeldata for choosen label

Input:
The dataframe to be separated
Labelmapping {string to integer values}
Choosen label for the true label values
First and last column to extract data from (inclusively)

Output:
Data frame with data
Stringlabels representing original label values
The untouched label_mapping
Label values as integer representation

Others:

"""	
def get_model_data(dataframe, label_mapping, choosen_label = 'T_CHASSIS', first_column = '1_1', last_column = '20_20'):

	string_labels = dataframe.pop(choosen_label)
	dataframe = dataframe.loc[:, first_column:last_column] # loc[:, '1_1':'20_20']

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
	
"""
Description:
Only used for the DNN-classifier

Input:

Output:
A Tensorflow dataset for training and test

Others:

"""	
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

"""
Description:
Only used for the DNN-classifier

Input:

Output:
A Tensorflow dataset for validation/prediction

Others:

"""
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

"""
Description:
Splits the dataframe for using Kfolding

Input:
An empty list
Number of folds
The dataframe with train and test data to split

Output:
A list of dataframes for each fold

Others:

"""		
def getFoldFrame(foldframe_list, kfolds, foldframe):

	if kfolds == 1:
		foldframe_list.append(foldframe)
	else:
		trainframe, restframe = train_test_split(foldframe, train_size= (1 / kfolds))
		foldframe_list.append(trainframe)
		getFoldFrame(foldframe_list, kfolds - 1, restframe)

	return 	foldframe_list
		
		
"""
Description:
Puts together a trainingset and a testset from a list of dataframes
Testindex telling which frame to testframe

Input:
A list of dataframes
Index to be the testframe

Output:
A trainframe and a testframe

Others:

"""
def getFoldTrainFrames(foldframe_list, testindex):
	
	foldtrainframe = pandas.DataFrame()
	foldtestframe = pandas.DataFrame()
	
	for index in range(len(foldframe_list)):
		if index == testindex:
			foldtestframe = foldtestframe.append(foldframe_list[index])
		else:
			foldtrainframe = foldtrainframe.append(foldframe_list[index])
		
	return foldtrainframe, foldtestframe

	
"""
Description:
Prints a plotted Confusion matrix to the resultfolder with file-suffix

Input:
A confusion matrix with values
Labels do define the axis
Filesuffix to append the saved file.

Output:
A written file.

Others:

"""
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
	
"""
Description:
Gives a string with total rows and specific row numbers for chooosen_label (implies that value is 0 or 1)

Input:
The dataframe for staistics calculation
The choosen labe for specific statistics

Output:
A string with statistics

Others:
If label 'valid' doesnt exist it responds 0

"""	
def analyse_frame(dataframe, choosen_label):	
	
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
		if str(row[choosen_label]) == '1':
			nr_of_repaired += 1
	
	resultstring += '\n\rRows: ' + str(nr_of_rows)
	resultstring += '\n\rValid: ' + str(nr_of_valid)
	resultstring += '\n\rRepaired: ' + str(nr_of_repaired)
	
	return resultstring


"""
Description:
Plots a ROC-curve for positive label = 1

Input:
True labelvalues
Probability values for the positive label (1)
File-suffix for the saved file

Output:
A written file.

Others:

"""	
def print_roc_curve(y_true, y_prob, filesuffix):

	fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
	roc_auc = auc(fpr, tpr)
	
	plt.figure()
	plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve unhealthy (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.savefig('Results/ROC_curve-' + filesuffix + '.png')
	plt.clf()

"""
Description:
Plots a hisstogram showing the distribution for the predicted probability values

Input:
Predicted probability values, likely for positive label 1
File-suffix for the saved file

Output:
A written file.

Others:

"""
def print_probabilities(probabilities, file_suffix):

	probabilities.plot(kind='hist', bins=40)
	#x1,x2,y1,y2 = plt.axis()
	#plt.axis((x1,x2,y1,400))
	plt.title("Probabilities Unhealthy")
	plt.grid(True)
	plt.savefig("Results/Probabilities-" + file_suffix + ".png")
	plt.clf()	
	



