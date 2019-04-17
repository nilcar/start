

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function


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

#import dataloader # Holds helpfunctions and print functions for statistics
#import cnn_config # Holds at least one network configuration (one must be pointed out here for creation of the estimator)

import dataloader_predict
import cnn_config_predict
import cnn_config_saved

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

#tensorflow.logging.set_verbosity(tensorflow.logging.INFO)
#tensorflow.app.run()


	
	

def predict_on_model(data_directory, model_path, choosen_label):


	#tensorflow.logging.set_verbosity(tensorflow.logging.INFO)
	#tensorflow.app.run()
		
	print('Function called: ' + str(__name__))
	
	
	
	data_path = data_directory
	
	file_suffix = '-' + choosen_label
	label_mapping = {0:0, 1:1}
	inverted_label_mapping = {}
	for key, value in label_mapping.items():
		inverted_label_mapping[value] = key
		
	print('Open resultfile.')
	
	resultfile = open("Results/model_results" + file_suffix + ".txt", "w")
	resultfile.write('\n\rModel training: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n\n\r')
	resultfile.write('Choosen label: ' + choosen_label + '\n\r')
	resultfile.write('Data path: ' + str(data_path) + '\n\r')
	resultfile.write('Label mapping: ' + str(label_mapping) + '\n\r')
	resultfile.write('Inverted label mapping: ' + str(inverted_label_mapping) + '\n\r')
	resultfile.flush()
	
	#Get the dataframe to validate on
	validationframe = dataloader_predict.loadValidationFrame(data_path)
	
	# Get validation model data
	validationset, labels_validate, label_mapping, int_labels_validate = \
		dataloader_predict.get_model_data(validationframe, label_mapping, choosen_label)
	
	validate_data = validationset.values.astype(numpy.float32)
	
	cnn_validate_input_fn = tensorflow.estimator.inputs.numpy_input_fn(x={"x": validate_data}, y=None, num_epochs=1, shuffle=False)
	
	classifier = tensorflow.estimator.Estimator(model_fn=cnn_config_saved.cnn_model_dnn5CL3_fn, model_dir=model_path)
	
	
	### Evaluate the model
	print('\nModel evaluation\n\n\n')
	resultfile.write('\n\rModel evaluation\n\r\n')
	expected = list(int_labels_validate) # The integer representation of the labels. Converts with: inverted_label_mapping() to label
	# Make the predictions on the supplied data on the saved model.
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
	limit = 0.96
	unhealthy_probabilities = pandas.Series()
	
	for pred_dict, expec in zip(predictions, expected):
		class_id = pred_dict['class_ids']
		probability = pred_dict['probabilities'][class_id]
		resultfile.write('\n\r')
		resultfile.write(template.format(inverted_label_mapping[class_id], 100 * probability, inverted_label_mapping[expec]))
		number_of_validations += 1
		y_true.append(inverted_label_mapping[expec])
		y_predicted.append(inverted_label_mapping[class_id])
		y_probability.append(pred_dict['probabilities'][1]) # For positive label in ROC-curve
		
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
	
	dataloader_predict.print_cm(confusion_matrix_result, list(label_mapping.keys()), file_suffix)
	dataloader_predict.print_cm(confusion_matrix_new, list(label_mapping.keys()), file_suffix + 'New')
	dataloader_predict.print_roc_curve(numpy.array(y_true), numpy.array(y_probability), file_suffix)
	dataloader_predict.print_probabilities(unhealthy_probabilities, file_suffix)
	
	predictfile.write('\n\rNumber of matches in percent: ' + str(100 * number_of_matches / number_of_validations))
	predictfile.write('\n\rTotal: ' + str(number_of_validations))
	predictfile.write('\n\rMatches: ' + str(number_of_matches))
	predictfile.write('\n\rAverage matches probability: ' + str(total_probability / number_of_matches))
	resultfile.write('\n\r******************************\n\r')
	resultfile.close()
	predictfile.close()
	
	
	
	
def main(argv):
	
	#tensorflow.logging.set_verbosity(tensorflow.logging.INFO)
	#tensorflow.app.run()
	
	print('Tensorflow running')
	
	#cnn_model_predict.predict_on_model('Data2/V3/', '/data/Tensorflow/CNN/-repaired15000-CNN_Kfold5_Normal_800_CL3_ADAGR_01_LL20_NP_DR02_1234_DIL3_dnn5_1/', 'repaired')
	predict_on_model('Data2/V3/', '/data/Tensorflow/CNN/-repaired15000-CNN_Kfold5_Normal_800_CL3_ADAGR_01_LL20_NP_DR02_1234_DIL3_dnn5_1/', 'repaired')
	

if __name__ == '__main__':
    tensorflow.logging.set_verbosity(tensorflow.logging.INFO)
    tensorflow.app.run(main)	
	
	
