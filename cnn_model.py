

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pandas
import numpy
import tensorflow
import argparse
from os import listdir
from os.path import isfile, join
import datetime
import sys
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import dataloader # Holds helpfunctions and print functions for statistics
import cnn_config # Holds at least one network configuration (one must be pointed out here for creation of the estimator)

"""
Description:
This is the main programfile in order to train, test and validate a CNN network solving a binary classification problem for given indata.
The program is implemented using "Tensorflow" backend library version 1.11 (Python version 3.5)

Input:
batch_size: The size of the batch size to be used for training (recommended:100)
train_steps: The number of steps to train the model, (recommended > 5000)
nr_epochs: The number of epochs to train the model, (recommended:0 implies None in the model training)
choosen_label: The label with "True" values for healthiness of a truck (unhealthy shall be marked to 1, 0 otherwise)
data_path: The path to a folder holding one and only one csv kommaseparated datafile with at least columns 1_1 .. 20_20 for features and one column(choosen_label) holding the labeldata.
fixed_selection: Can be true or false, implies two different predefined data-selections, see function loadData in file dataloader.
suffix: Only a filename suffix in order to separate printed results to predefined "Result" folder for different program runs.

Output:
A trained and saved model to the directory pointed out, see code below creating the estimator.
Two printed files holding training and prediction statistics.
Two Confusion matrices, a ROC-curve and a distribution histogram for unhealthy trucks probability values.

Others:
- Requires a folder 'Results' in the same directory as this programfile where results will be stored.
- The defined network shall be in the cnn_config.py file, the specific network is pointed out creating the estimator, see code below.
- Example for usage: python3.5 cnn_model.py --batch_size 100 --train_steps 1000 nr_epochs 0 --fixed_selection false --choosen_label T_CHASSIS --data_path Validationdata/ --suffix Suffix
"""

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')
parser.add_argument('--nr_epochs', default=0, type=int, help='number of epochs')
parser.add_argument('--choosen_label', default='T_CHASSIS', type=str, help='the label to train and evaluate on (unhealthy=1)')
parser.add_argument('--data_path', default='Data/', type=str, help='path to data source file, only one file')
parser.add_argument('--fixed_selection', default='False', type=str, help='Makes it possible to have two different selections')
parser.add_argument('--suffix', default='', type=str, help='To separate result filenames and model stored')


def main(argv):

	args = parser.parse_args(argv[1:])
	batch_size = args.batch_size
	train_steps = args.train_steps
	nr_epochs =  args.nr_epochs # None
	if nr_epochs == 0:
		nr_epochs = None
	
	choosen_label = args.choosen_label
	
	if args.fixed_selection.lower() == 'false':
		fixed_selection = False
	else:
		fixed_selection = True
	
	data_path = args.data_path
	
	file_suffix = '-' + choosen_label + str(args.train_steps) + '-' + args.suffix
	kfolds = 5 # Splitting training/test dataset into kfolds sets. Total number of steps becomes kfolds*steps 
	
	resultfile = open("Results/model_results" + file_suffix + ".txt", "w")
	resultfile.write('\n\rModel training: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n\n\r')
	resultfile.write('Train steps: ' + str(train_steps) + '\n\r')
	resultfile.write('Number epochs: ' + str(nr_epochs) + '\n\r')
	resultfile.write('Batchsize: ' + str(batch_size) + '\n\r')
	resultfile.write('Choosen label: ' + choosen_label + '\n\r')
	resultfile.write('Fixed_selection: ' + str(fixed_selection) + '\n\r')
	resultfile.write('Data path: ' + str(data_path) + '\n\r')
	resultfile.write('Kfold: ' + str(kfolds) + '\n\r')
	
	# Label_mapping holds key value pairs where key is the label and value its integer representation.
	# Here unhealthy is 1 for both column and integer representation.
	label_mapping = {0:0, 1:1}
	resultfile.write('Label mapping: ' + str(label_mapping) + '\n\r')
	
	inverted_label_mapping = {}
	for key, value in label_mapping.items():
		inverted_label_mapping[value] = key
	
	resultfile.write('Inverted label mapping: ' + str(inverted_label_mapping) + '\n\r')
	resultfile.flush()
	
	#Get three structured separate dataframes from data sources, first and last feature column names
	trainframe, testframe, validationframe, first_column, last_column = dataloader.loadData(data_path, fixed_selection, file_suffix)
	resultfile.flush()
	
	if kfolds <= 1:
	
		# Prints some dataframe statistics to the resultfile.
		frameinfo = dataloader.analyse_frame(trainframe, choosen_label)
		resultfile.write('\n\rTrainframe:\n\r')
		resultfile.write(frameinfo)
		frameinfo = dataloader.analyse_frame(testframe, choosen_label)
		resultfile.write('\n\r\n\rTestframe:\n\r')
		resultfile.write(frameinfo)
		frameinfo = dataloader.analyse_frame(validationframe, choosen_label)
		resultfile.write('\n\r\n\rValidationframe:\n\r')
		resultfile.write(frameinfo)
	
		# Train model data
		trainset, labels_training, label_mapping, int_labels_train = \
			dataloader.get_model_data(trainframe, label_mapping, choosen_label, first_column, last_column)
		
		# Test model data
		testset, labels_test, label_mapping, int_labels_test = \
			dataloader.get_model_data(testframe, label_mapping, choosen_label, first_column, last_column)
		
		# Validate model data
		validationset, labels_validate, label_mapping, int_labels_validate = \
			dataloader.get_model_data(validationframe, label_mapping, choosen_label, first_column, last_column)
		
		#Numpy representation, setting to double (default) may cause exceptions for some optimizers
		train_data = trainset.values.astype(numpy.float32)
		train_labels = int_labels_train.values
		
		test_data = testset.values.astype(numpy.float32)
		test_labels = int_labels_test.values
		
		validate_data = validationset.values.astype(numpy.float32)
		
		print(train_data.shape)
		print(test_data.shape)
		print(validate_data.shape)
		
		print(train_labels.shape)
		print(test_labels.shape)
		
		# Define the specific model functions for training, test and validation
		cnn_train_input_fn = tensorflow.estimator.inputs.numpy_input_fn(x={"x": train_data}, y=train_labels, batch_size=batch_size, num_epochs=nr_epochs, shuffle=True)
		cnn_eval_input_fn = tensorflow.estimator.inputs.numpy_input_fn(x={"x": test_data}, y=test_labels, num_epochs=1, shuffle=False)
		cnn_validate_input_fn = tensorflow.estimator.inputs.numpy_input_fn(x={"x": validate_data}, y=None, num_epochs=1, shuffle=False)
		
		# Create the Estimator with a predefined model function
		classifier = tensorflow.estimator.Estimator(model_fn=cnn_config.cnn_model_dnn5CL3_fn, model_dir='/data/Tensorflow/' + file_suffix)
		
		### Train the Model.
		print('\nModel training\n\r\n\r\n')

		classifier.train(input_fn=cnn_train_input_fn,steps=train_steps)
		
		### Test the model
		print('\n\r\n\rModel testing\n\n\n')
		resultfile.write('\n\r\n\rModel testing\n\r')
		# Evaluate the model.
		
		eval_result = classifier.evaluate(input_fn=cnn_eval_input_fn)
		
		print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
		resultfile.write('\n\rTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
		resultfile.write('\n\rEval result:\n\r' + str(eval_result))
		
	else:
	
		foldframe = trainframe.append(testframe)
		foldframe = foldframe.reindex(numpy.random.permutation(foldframe.index)) # shuffle around data
		foldtrainframe = pandas.DataFrame()
		foldtestframe = pandas.DataFrame()
		foldframe_list = []
	
		foldframe_list = dataloader.getFoldFrame(foldframe_list, kfolds, foldframe)
		
		frameinfo = dataloader.analyse_frame(validationframe, choosen_label)
		resultfile.write('\n\r\n\rValidationframe:\n\r')
		resultfile.write(frameinfo)
		
		# Validate model data
		validationset, labels_validate, label_mapping, int_labels_validate = \
			dataloader.get_model_data(validationframe, label_mapping, choosen_label, first_column, last_column)
		
		testresults = []
	
		for testindex in range(kfolds):
	
			foldtrainframe, foldtestframe = dataloader.getFoldTrainFrames(foldframe_list, testindex)
		
			frameinfo = dataloader.analyse_frame(foldtrainframe, choosen_label)
			resultfile.write('\n\rTrainframe:\n\r')
			resultfile.write(frameinfo)
			frameinfo = dataloader.analyse_frame(foldtestframe, choosen_label)
			resultfile.write('\n\r\n\rTestframe:\n\r')
			resultfile.write(frameinfo)
		
			# Train model data
			trainset, labels_training, label_mapping, int_labels_train = \
				dataloader.get_model_data(foldtrainframe, label_mapping, choosen_label, first_column, last_column)
			
			# Test model data
			testset, labels_test, label_mapping, int_labels_test = \
				dataloader.get_model_data(foldtestframe, label_mapping, choosen_label, first_column, last_column)
			
			#Numpy representation, setting to double (default) may cause exceptions for some optimizers
			train_data = trainset.values.astype(numpy.float32)
			train_labels = int_labels_train.values
		
			test_data = testset.values.astype(numpy.float32)
			test_labels = int_labels_test.values
		
			validate_data = validationset.values.astype(numpy.float32)
			
			print(train_data.shape)
			print(test_data.shape)
			print(validate_data.shape)
			
			print(train_labels.shape)
			print(test_labels.shape)
			
			# Define the specific model functions for training, test and validation
			cnn_train_input_fn = tensorflow.estimator.inputs.numpy_input_fn(x={"x": train_data},y=train_labels,batch_size=batch_size,num_epochs=nr_epochs,shuffle=True)
			cnn_eval_input_fn = tensorflow.estimator.inputs.numpy_input_fn(x={"x": test_data},y=test_labels,num_epochs=1,shuffle=False)
			cnn_validate_input_fn = tensorflow.estimator.inputs.numpy_input_fn(x={"x": validate_data},y=None,num_epochs=1,shuffle=False)
			
			# Create the Estimator with a predefined model function (predefined CNN network)
			classifier = tensorflow.estimator.Estimator(model_fn=cnn_config.cnn_model_dnn5CL3_fn, model_dir='/data/Tensorflow/' + file_suffix)
			
			### Train the Model.
			print('\nModel training\n\r\n\r\n')

			classifier.train(input_fn=cnn_train_input_fn, steps=train_steps)
			
			### Test the model
			print('\n\r\n\rModel testing\n\n\n')
			resultfile.write('\n\r\n\rModel testing\n\r')
			# Evaluate the model.
			
			eval_result = classifier.evaluate(input_fn=cnn_eval_input_fn)
			
			print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
			resultfile.write('\n\rK-fold:' + str(testindex + 1))
			resultfile.write('\n\rTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
			resultfile.write('\n\rEval result:\n\r' + str(eval_result))
			testresults.append(eval_result['accuracy'])
	
		average = 0.0
		for value in testresults:
			average += value
		
		resultfile.write('\n\rAverage testresult:' + str(average / len(testresults)))
	
	### Evaluate the model
	print('\nModel evaluation\n\n\n')
	resultfile.write('\n\rModel evaluation\n\r\n')
	expected = list(int_labels_validate) # The integer representation of the labels. Converts with: inverted_label_mapping() to label
	# Get the predictionsresults from trained model
	predictions = classifier.predict(input_fn=cnn_validate_input_fn)
	
	template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
	predictfile = open("Results/predictions" + file_suffix + ".txt", "w")
	
	number_of_matches = 0
	number_of_validations = 0
	y_true = []
	y_predicted = []
	y_probability = []
	total_probability = 0
	y_predicted_new = []
	limit = 0.96 # Unhealthy trucks with probability value under this limit will be reclassified to healthy
	unhealthy_probabilities = pandas.Series()
	
	# Calculation of and printing of prediction results.
	for pred_dict, expec in zip(predictions, expected):
		class_id = pred_dict['class_ids']
		probability = pred_dict['probabilities'][class_id]
		resultfile.write('\n\r')
		resultfile.write(template.format(inverted_label_mapping[class_id], 100 * probability, inverted_label_mapping[expec]))
		number_of_validations += 1
		y_true.append(inverted_label_mapping[expec])
		y_predicted.append(inverted_label_mapping[class_id])
		y_probability.append(pred_dict['probabilities'][1]) # For positive label in ROC-curve (unhealthy)
		
		# Collect probability values for unhealthy trucks for plotting
		if inverted_label_mapping[class_id] == 1:
			unhealthy_probabilities = unhealthy_probabilities.append(pandas.Series([probability]))
		
		# Reclassify unhealthy truks whos probability is under choosen limit
		if inverted_label_mapping[class_id] == 1 and probability < limit:
			y_predicted_new.append(0)
		else:
			y_predicted_new.append(class_id)
		
		# Calculate the number of correct predicted classes and print to file
		if str(inverted_label_mapping[class_id]) == str(inverted_label_mapping[expec]):
			predictfile.write('Percent: ' + str(100 * probability) + '  ' + choosen_label + ': ' + str(inverted_label_mapping[expec]) + '\n\r')
			number_of_matches += 1
			total_probability += 100 * probability
					
	confusion_matrix_result = confusion_matrix(y_true, y_predicted, labels=list(label_mapping.keys()).sort()) # labels=[0,1]
	print(confusion_matrix_result)
	# CM regarding with potentially reclassified samples
	confusion_matrix_new = confusion_matrix(y_true, y_predicted_new, labels=list(label_mapping.keys()).sort()) # labels=[0,1]
	print(confusion_matrix_new)
	
	dataloader.print_cm(confusion_matrix_result, list(label_mapping.keys()), file_suffix)
	dataloader.print_cm(confusion_matrix_new, list(label_mapping.keys()), file_suffix + 'New')
	dataloader.print_roc_curve(numpy.array(y_true), numpy.array(y_probability), file_suffix)
	dataloader.print_probabilities(unhealthy_probabilities, file_suffix)
	
	predictfile.write('\n\rNumber of matches in percent: ' + str(100 * number_of_matches / number_of_validations))
	predictfile.write('\n\rTotal: ' + str(number_of_validations))
	predictfile.write('\n\rMatches: ' + str(number_of_matches))
	predictfile.write('\n\rAverage matches probability: ' + str(total_probability / number_of_matches))
	resultfile.write('\n\r******************************\n\r')
	resultfile.close()
	predictfile.close()
	
if __name__ == '__main__':
    tensorflow.logging.set_verbosity(tensorflow.logging.INFO)
    tensorflow.app.run(main)
	
	
	
	
	
	
