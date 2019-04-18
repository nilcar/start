

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

import dataloader_validate # Holds helpfunctions and print functions for statistics
import cnn_config_saved # Holds at least the network configuration of at least one saved model


"""
Description:
This is the main programfile in order to validate (make predictions) on a saved trained CNN model (the path to the model is pointed out when creating the estimator, see code below)
The program is implemented using "Tensorflow" backend library version 1.11 (Python version 3.5)

Input:
choosen_label: The datacolumn holding truth about beeing repaired (0,1), 1 = repaired
data_path: The path to a directory holding a single datafile to predict on
suffix: Only a suffix given to files printed out in Result directory

Output:
Several files showing prediction results...

Others:
- Requires a folder 'Results' in the same directory as this programfile where results will be stored.
- The defined network shall be in the cnn_config_saved.py file, the specific network is pointed out creating the estimator, see code below.
  It is important that the pointed CNN network in cnn_config_saved.py file complies to the saved model!
- Example for usage: python3.5 cnn_model_validate.py --choosen_label T_CHASSIS --data_path Validationdata/ --suffix Suffix
"""

parser = argparse.ArgumentParser()
parser.add_argument('--choosen_label', default='T_CHASSIS', type=str, help='the label to evaluate')
parser.add_argument('--data_path', default='Data/', type=str, help='path to one data source file for prediction')
parser.add_argument('--suffix', default='', type=str, help='To separate result filenames')

def main(argv):

	args = parser.parse_args(argv[1:])
	
	choosen_label = args.choosen_label
	data_path = args.data_path
	file_suffix = '-' + choosen_label + '-' + args.suffix
	# Label_mapping holds key value pairs where key is the label and value its integer representation
	label_mapping = {0:0, 1:1}
	inverted_label_mapping = {}
	for key, value in label_mapping.items():
		inverted_label_mapping[value] = key
	
	resultfile = open("Results/model_results" + file_suffix + ".txt", "w")
	resultfile.write('\n\rModel training: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n\n\r')
	resultfile.write('Choosen label: ' + choosen_label + '\n\r')
	resultfile.write('Data path: ' + str(data_path) + '\n\r')
	resultfile.write('Label mapping: ' + str(label_mapping) + '\n\r')
	resultfile.write('Inverted label mapping: ' + str(inverted_label_mapping) + '\n\r')
	resultfile.flush()
	
	#Get the dataframe to validate on
	validationframe = dataloader_validate.loadValidationFrame(data_path)
	
	# Get validation model data
	validationset, labels_validate, label_mapping, int_labels_validate = \
		dataloader_validate.get_model_data(validationframe, label_mapping, choosen_label)
	
	validate_data = validationset.values.astype(numpy.float32)
	
	cnn_validate_input_fn = tensorflow.estimator.inputs.numpy_input_fn(x={"x": validate_data}, y=None, num_epochs=1, shuffle=False)
	
	# Create the Estimator with saved model from appointed directory
	#classifier = tensorflow.estimator.Estimator(model_fn=cnn_config_saved.cnn_model_dnn5_fn, model_dir='/data/Tensorflow/CNN/-repaired10000-CNN_Kfold5_Normal_800_CL2_ADAGR_01_LL20_NP_DR02_1234_DIL2_dnn5_1/')
	#classifier = tensorflow.estimator.Estimator(model_fn=cnn_config_saved.cnn_model_dnn5_fn, model_dir='/data/Tensorflow/CNN/-repaired10000-CNN_Kfold5_Normal_800_CL2_ADAGR_01_LL20SM_NP_DR02_1234_DIL2_dnn5_1/')
	classifier = tensorflow.estimator.Estimator(model_fn=cnn_config_saved.cnn_model_dnn5CL3_fn, model_dir='/data/Tensorflow/CNN/-repaired15000-CNN_Kfold5_Normal_800_CL3_ADAGR_01_LL20_NP_DR02_1234_DIL3_dnn5_1/')
	
	#classifier = tensorflow.estimator.Estimator(model_fn=cnn_config_saved.cnn_model_dnn5CL3_fn, model_dir='/data/Tensorflow/CNN/X/')
	
	
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
	
	dataloader_validate.print_cm(confusion_matrix_result, list(label_mapping.keys()), file_suffix)
	dataloader_validate.print_cm(confusion_matrix_new, list(label_mapping.keys()), file_suffix + 'New')
	dataloader_validate.print_roc_curve(numpy.array(y_true), numpy.array(y_probability), file_suffix)
	dataloader_validate.print_probabilities(unhealthy_probabilities, file_suffix)
	
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
	
	
	
	
	
	
