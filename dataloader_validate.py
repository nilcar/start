
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

def loadValidationFrame(directory):

	dataframe = pandas.DataFrame()
	datafiles = []
	for item in listdir(directory): 
		if isfile(join(directory, item)):
			datafiles.append(directory + item)

	for datafile in datafiles:
		print('Reading compressed: ' + datafile)
		dataframe = pandas.read_csv(datafile, sep=";", index_col=False)

	#labelmaskvalidate = (dataframe['Send_Date'] > '2018-08-15' & dataframe['Send_Date'] < '2018-08-55')
	#labelmaskvalidate = (dataframe['Send_Date'] == '2018-08-15')
	labelmaskvalidate = (dataframe['PARTITIONNING'] == '3_Validation')
	dataframe = dataframe.loc[labelmaskvalidate]

	dataframe = dataframe.fillna(value = 0.0)	
	print(dataframe.head())
	
	return dataframe


"""
Description:
Separates data from labeldata for choosen label

Input:
The dataframe to be separated
Labelmapping {string to integer values}
Choosen label for the true label values

Output:
Data frame with data
Stringlabels representing original label values
The untouched label_mapping
Label values as integer representation

Others:

"""	
	
def get_model_data(dataframe, label_mapping, choosen_label = 'T_CHASSIS'):

	# Clean up the dataframe to be converted into tensorflow datasets (features and labels)
	string_labels = dataframe.pop(choosen_label)
	dataframe = dataframe.loc[:, '1_1':'20_20']
			
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


