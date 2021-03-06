
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


	
def get_model_data(dataframe, choosen_label = 'T_CHASSIS'):

	chassis = dataframe.pop(choosen_label)
	dataframe = dataframe.loc[:, '1_1':'20_20']

	return dataframe, chassis
	

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


def print_probabilities(probabilities, file_suffix):

	probabilities.plot(kind='hist', bins=40)
	#x1,x2,y1,y2 = plt.axis()
	#plt.axis((x1,x2,y1,400))
	plt.title("Probabilities Unhealthy")
	plt.grid(True)
	plt.savefig("Results/Probabilities-" + file_suffix + ".png")
	plt.clf()


