

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

import dataloader_validate



# Example: python3.5 dnn_model_validate.py --batch_size 100 --train_steps 1000 --hidden_units 20x20 --choosen_label T_CHASSIS --data_path Validationdata/

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')
parser.add_argument('--hidden_units', default='10x10', type=str, help='layout for hidden layers')
parser.add_argument('--choosen_label', default='T_CHASSIS', type=str, help='the label to train and evaluate')
parser.add_argument('--data_path', default='Compressed/', type=str, help='path to data source files or compressed file')
parser.add_argument('--suffix', default='', type=str, help='To separate result filenames')


def main(argv):

	args = parser.parse_args(argv[1:])
	batch_size = args.batch_size
	hidden_units_arg = list(args.hidden_units.split('x'))
	hidden_units = []
	for layer in hidden_units_arg:
		hidden_units.append(int(layer))
	
	choosen_label = args.choosen_label
	data_path = args.data_path
	file_suffix = '-' + choosen_label + '-' + args.hidden_units + '-' + '-' + args.suffix
	dropout = None
	# Label_mapping holds key value pairs where key is the label and value its integer representation
	label_mapping = {0:0, 1:1}
	inverted_label_mapping = {}
	for key, value in label_mapping.items():
		inverted_label_mapping[value] = key
	
	resultfile = open("Results/model_results" + file_suffix + ".txt", "w")
	resultfile.write('\n\rModel training: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n\n\r')
	resultfile.write('Layer setting: ' + str(hidden_units) + '\n\r')
	resultfile.write('Batchsize: ' + str(batch_size) + '\n\r')
	resultfile.write('Choosen label: ' + choosen_label + '\n\r')
	resultfile.write('Data path: ' + str(data_path) + '\n\r')
	resultfile.write('Dropout: ' + str(dropout) + '\n\r')
	resultfile.write('Label mapping: ' + str(label_mapping) + '\n\r')
	resultfile.write('Inverted label mapping: ' + str(inverted_label_mapping) + '\n\r')
	resultfile.flush()
	
	#Get three structured separate dataframes from data sources
	validationframe = dataloader_validate.loadValidationFrame(data_path)
	
	# Validate model data
	validationset, labels_validate, label_mapping, int_labels_validate = \
		dataloader_validate.get_model_data(validationframe, label_mapping, choosen_label)
		
	### Model training
	my_feature_columns = []
	for key in validationset.keys():
		my_feature_columns.append(tensorflow.feature_column.numeric_column(key=key))

	# The model must choose between x classes.
	print('Number of unique labels, n_classes: ' + str(len(label_mapping)))
		
	# optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=0.1) ?
	# optimizer = tensorflow.train.AdagradOptimizer(learning_rate=0.1) ?
	# optimizer = tensorflow.train.AdagradDAOptimizer(learning_rate=0.1, global_step= ?) global_step=train_steps?	
	# optimizer = tensorflow.train.AdamOptimizer(learning_rate=0.1) ?
	optimizer = tensorflow.train.ProximalAdagradOptimizer(learning_rate=0.01, l1_regularization_strength=0.01)
	#optimizer = 'Adagrad'
		
	classifier = tensorflow.estimator.DNNClassifier \
		(feature_columns=my_feature_columns,hidden_units=hidden_units,n_classes=len(label_mapping), dropout=dropout, batch_norm=False, optimizer=optimizer, model_dir='/data/Tensorflow/-repaired-400x400x200x20-100000-V3_100000_Test/')
	
	### Evaluate the model
	print('\nModel evaluation\n\n\n')
	resultfile.write('\n\rModel evaluation\n\r\n')
	expected = list(int_labels_validate) # The integer representation of the labels. Converts with: inverted_label_mapping() to label
	predictions = classifier.predict(input_fn=lambda:dataloader_validate.eval_input_fn(validationset, labels=None, batch_size=batch_size))
	template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
	
	predictfile = open("Results/predictions" + file_suffix + ".txt", "w")
	
	number_of_matches = 0
	number_of_validations = 0
	y_true = []
	y_predicted = []
	y_probability = []
	
	for pred_dict, expec in zip(predictions, expected):
		class_id = pred_dict['class_ids'][0]
		probability = pred_dict['probabilities'][class_id]
		resultfile.write('\n\r')
		resultfile.write(template.format(inverted_label_mapping[class_id], 100 * probability, inverted_label_mapping[expec]))
		number_of_validations += 1
		y_true.append(inverted_label_mapping[expec])
		y_predicted.append(inverted_label_mapping[class_id])
		y_probability.append(probability)
		
		if str(inverted_label_mapping[class_id]) == str(inverted_label_mapping[expec]):
			predictfile.write('Percent: ' + str(100 * probability) + '  ' + choosen_label + ': ' + str(inverted_label_mapping[expec]) + '\n\r')
			number_of_matches += 1

	confusion_matrix_result = confusion_matrix(y_true, y_predicted, labels=list(label_mapping.keys()).sort()) # labels=[0,1]
	print(confusion_matrix_result)
	#dataloader_validate.print_cm(confusion_matrix_result, list(label_mapping.keys()), file_suffix)
	#dataloader_validate.print_roc_curve(numpy.array(y_true), numpy.array(y_probability), list(label_mapping.keys()), file_suffix)
	
	predictfile.write('\n\rNumber of matches in percent: ' + str(100 * number_of_matches / number_of_validations))
	predictfile.write('\n\rTotal: ' + str(number_of_validations))
	predictfile.write('\n\rMatches: ' + str(number_of_matches))
	resultfile.write('\n\r******************************\n\r')
	resultfile.close()
	predictfile.close()
	
if __name__ == '__main__':
    tensorflow.logging.set_verbosity(tensorflow.logging.INFO)
    tensorflow.app.run(main) # So far only a dummy arguments...
	
	
	
	
	
	
