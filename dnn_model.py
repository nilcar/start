

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pandas
import tensorflow
import argparse
from os import listdir
from os.path import isfile, join

import dataloader

"""
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')
"""

def main(argv):

	"""
	args = parser.parse_args(argv)
	#args = parser.parse_args(argv[1:])
	print(args)
	return
	"""
	
	batch_size = 100
	train_steps = 1000
	nr_epochs = None
	hidden_units = [10, 10]
	
	label_path = 'Labels/'
	data_path = 'Testdata/'
	
	#Get three structured separate dataframes from data sources
	trainframe, testframe, validationframe = dataloader.loadData('Testdata/')
	label_mapping = dict() # If labels from label data file is not used
	
	"""
	# label mapping from labels files
	datafiles = []
	for item in listdir('Labels/'): 
		if isfile(join(label_path, item)):
			datafiles = datafiles.append(label_path + item)
	print(datafiles)
	
	csv_label_data = pandas.DataFrame()
	for datafile in datafiles:
		print(datafile)
		label_data = pandas.read_csv(datafile, sep=",")
		csv_label_data = csv_label_data.append(label_data)
		
	string_labels = csv_label_data.pop('truck_id')
	label_mapping = dict([(y,x+1) for x,y in enumerate(sorted(set(string_labels.tolist())))])
	
	"""
	
	"""
	csv_labels = pandas.read_csv('Labels/Volvo_labels_nominal.csv', sep=",")
	#print(csv_labels)
	all_truck_ids = csv_labels.pop('T_CHASSIS')
	"""
	
	# Train model data
	trainset, labels_training, label_mapping_train, int_labels_train = dataloader.get_model_data(trainframe, label_mapping)
	
	# Test model data
	testset, labels_test, label_mapping_test, int_labels_test = dataloader.get_model_data(testframe, label_mapping_train)
	
	# Validate model data
	validationset, labels_validate, label_mapping, int_labels_validate = dataloader.get_model_data(validationframe, label_mapping_test)
	
	### Model training
	my_feature_columns = []
	for key in trainset.keys():
		my_feature_columns.append(tensorflow.feature_column.numeric_column(key=key))
	
	# Build 2 hidden layer DNN with 10, 10 units respectively.
	# Two hidden layers of 10 nodes each.
	# The model must choose between x classes.
	print('Number of unique trucks, n_classes: ' + str(len(label_mapping)))
	#print('Number of unique trucks, n_classes: ' + str(int_labels.size))
	classifier = tensorflow.estimator.DNNClassifier(feature_columns=my_feature_columns,hidden_units=hidden_units,n_classes=len(label_mapping))
	#classifier = tensorflow.estimator.DNNClassifier(feature_columns=my_feature_columns,hidden_units=[10, 10],n_classes=int_labels.size)
	
    ### Train the Model.
	print('\nModel training\n\n\n')
	classifier.train(input_fn=lambda:dataloader.train_input_fn(trainset, int_labels_train, batch_size, nr_epochs),steps=train_steps)

	### Test the model
	print('\nModel testing\n\n\n')
	# Evaluate the model.
	eval_result = classifier.evaluate(input_fn=lambda:dataloader.eval_input_fn(testset, int_labels_test, batch_size))
	print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
	
	### Evaluate the model
	print('\nModel evaluation\n\n\n')
	expected = list(label_mapping.keys())
	predictions = classifier.predict(input_fn=lambda:dataloader.eval_input_fn(validationset, labels=None, batch_size=batch_size))
	template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
	
	for pred_dict, expec in zip(predictions, expected):
		class_id = pred_dict['class_ids'][0]
		probability = pred_dict['probabilities'][class_id]
		print(template.format(expected[class_id], 100 * probability, expec))
	
	
	
	
	
	
if __name__ == '__main__':
    tensorflow.logging.set_verbosity(tensorflow.logging.INFO)
    tensorflow.app.run(main('Data/'))
	
	
	
	
	
	
