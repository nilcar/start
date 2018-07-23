

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pandas
import tensorflow
import argparse
from os import listdir
from os.path import isfile, join
import datetime

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
	
	batch_size = 100 # 100
	train_steps = 1000 # 1000
	nr_epochs = None
	hidden_units = [10, 10] # [10, 10] [400, 400] [400, 400, 400, 400]
	choosen_label = 'T_CHASSIS'
	
	label_path = 'Labels/'
	data_path = 'Testdata/' # 'Data_original/' 'Testdata/'
	structured_data_path = 'Compressed/' # 'Compressed_valid_chassis'
	
	resultfile = open("Results/model_results.txt", "a")
	
	resultfile.write('\n\rModel training: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n\n\r')
	resultfile.write('Layer setting: ' + str(hidden_units) + '\n\r')
	resultfile.write('Train steps: ' + str(train_steps) + '\n\r')
	resultfile.write('Number epochs: ' + str(nr_epochs) + '\n\r')
	resultfile.write('Batchsize: ' + str(batch_size) + '\n\r')
	resultfile.write('Choosen label: ' + choosen_label + '\n\r')
	resultfile.flush()
	
	# Label_mapping holds key value pairs where key is the label and value its integer representation
	label_mapping = dataloader.get_valid_labels(label_path, choosen_label) # Labels from labels file only
	
	#Get three structured separate dataframes from data sources
	#trainframe, testframe, validationframe = dataloader.loadData(data_path, False, label_mapping)
	trainframe, testframe, validationframe = dataloader.loadData(structured_data_path, True, label_mapping)
	
	# Train model data
	trainset, labels_training, label_mapping, int_labels_train = \
		dataloader.get_model_data(trainframe, label_mapping, choosen_label)
	
	# Test model data
	testset, labels_test, label_mapping, int_labels_test = \
		dataloader.get_model_data(testframe, label_mapping, choosen_label)
	
	# Validate model data
	validationset, labels_validate, label_mapping, int_labels_validate = \
		dataloader.get_model_data(validationframe, label_mapping, choosen_label)
	
	### Model training
	my_feature_columns = []
	for key in trainset.keys():
		my_feature_columns.append(tensorflow.feature_column.numeric_column(key=key))

	# opt = GradientDescentOptimizer(learning_rate=0.1) ??	
		
	# The model must choose between x classes.
	print('Number of unique trucks, n_classes: ' + str(len(label_mapping)))
	#print('Number of unique trucks, n_classes: ' + str(int_labels.size))
	classifier = tensorflow.estimator.DNNClassifier \
		(feature_columns=my_feature_columns,hidden_units=hidden_units,n_classes=len(label_mapping))
	#classifier = tensorflow.estimator.DNNClassifier \
	#	(feature_columns=my_feature_columns,hidden_units=hidden_units,n_classes=len(label_mapping), model_dir='Volvo_model')
	
    ### Train the Model.
	print('\nModel training\n\r\n\r\n')
	#resultfile.write('\nModel training: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n\n\n')
	classifier.train(input_fn=lambda:dataloader.train_input_fn(trainset, int_labels_train, batch_size, nr_epochs), steps=train_steps)

	### Test the model
	print('\n\rModel testing\n\n\n')
	resultfile.write('\nModel testing\n\r\n\r\n')
	# Evaluate the model.
	eval_result = classifier.evaluate(input_fn=lambda:dataloader.eval_input_fn(testset, int_labels_test, batch_size))
	print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
	resultfile.write('\n\rTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
	
	### Evaluate the model
	print('\nModel evaluation\n\n\n')
	resultfile.write('\n\rModel evaluation\n\r\n\r\n')
	expected = list(label_mapping.keys())
	predictions = classifier.predict(input_fn=lambda:dataloader.eval_input_fn(validationset, labels=None, batch_size=batch_size))
	template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
	
	predictfile = open("Results/predictions.txt", "w")
	
	for pred_dict, expec in zip(predictions, expected):
		class_id = pred_dict['class_ids'][0]
		probability = pred_dict['probabilities'][class_id]
		#print(template.format(expected[class_id], 100 * probability, expec))
		resultfile.write('\n\r')
		resultfile.write(template.format(expected[class_id], 100 * probability, expec))
		
		if str(expected[class_id]) == str(expec):
			predictfile.write('Percent: ' + str(100 * probability) + ' T_CHASSIS: ' + str(expec) + '\n\r')
	
	resultfile.write('\n\r******************************\n\r')
	
	
	
if __name__ == '__main__':
    tensorflow.logging.set_verbosity(tensorflow.logging.INFO)
    tensorflow.app.run(main('Data/')) # So far only a dummy argument...
	
	
	
	
	
	
