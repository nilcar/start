

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

import dataloader



# Example: python dnn_model.py --batch_size 100 --train_steps 1000 --hidden_units 20,20  --nr_epochs 0 --choosen_label T_CHASSIS --label_path Labels/ --data_path Compressed/Compressed_valid_all_labels/Single/ --compressed_data True --max_nr_nan 0 --fixed_selection False

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')
parser.add_argument('--hidden_units', default='10x10', type=str, help='layout for hidden layers')
parser.add_argument('--nr_epochs', default=0, type=int, help='number of epochs')
parser.add_argument('--choosen_label', default='T_CHASSIS', type=str, help='the label to train and evaluate')
parser.add_argument('--label_path', default='Labels/', type=str, help='where one labels file is located')
parser.add_argument('--data_path', default='Compressed/', type=str, help='path to data source files or compressed file')
parser.add_argument('--compressed_data', default='True', type=str, help='if true structured data will be used only one file false means data source files and a structured file will be produced')
parser.add_argument('--max_nr_nan', default=0, type=int, help='number of nan per row for exclusion')
parser.add_argument('--fixed_selection', default='True', type=str, help='If true selection is done by truck_date')
parser.add_argument('--suffix', default='', type=str, help='To separate result filenames')


def main(argv):

	args = parser.parse_args(argv[1:]) # argv[1:] argv
	#parser.print_help()
	#print(args)
	#sys.exit()
	
	batch_size = args.batch_size # 100
	#print('Batch_size: ' + str(batch_size))
	train_steps = args.train_steps # 100000 # 1000
	nr_epochs =  args.nr_epochs # None
	if nr_epochs == 0:
		nr_epochs = None
	hidden_units_arg = list(args.hidden_units.split('x')) # [10, 10] #args.hidden_units # [400, 400] # [10, 10] [400, 400] [400, 400, 400, 400]
	hidden_units = []
	for layer in hidden_units_arg:
		hidden_units.append(int(layer))
	
	choosen_label = args.choosen_label # 'GEARBOX_' # 'T_CHASSIS' 'COUNTRY' 'ENGINE_TYPE' 'BRAND_TYPE' GEARBOX_
	max_nr_nan = args.max_nr_nan # 0
	if args.fixed_selection.lower() == 'false':
		fixed_selection = False
	else:
		fixed_selection = True
	#fixed_selection = args.fixed_selection # True
	
	if args.compressed_data.lower() == 'false':
		compressed_data = False
	else:
		compressed_data = True
	#compressed_data = False #args.compressed_data
	
	label_path = args.label_path # 'Labels/'
	data_path = args.data_path # 'Data_original/' # 'Data_original/' 'Testdata/'
	
	file_suffix = '-' + choosen_label + '-' + args.hidden_units + '-' + str(args.train_steps) + '-' + args.suffix
	dropout = None
	kfolds = 5
	
	#structured_data_path = 'Compressed/Compressed_valid_all_labels/' # 'Compressed_valid_chassis' Compressed/Compressed_single/
	
	#sys.exit()
	
	resultfile = open("Results/model_results" + file_suffix + ".txt", "w")
	
	resultfile.write('\n\rModel training: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n\n\r')
	resultfile.write('Layer setting: ' + str(hidden_units) + '\n\r')
	resultfile.write('Train steps: ' + str(train_steps) + '\n\r')
	resultfile.write('Number epochs: ' + str(nr_epochs) + '\n\r')
	resultfile.write('Batchsize: ' + str(batch_size) + '\n\r')
	resultfile.write('Choosen label: ' + choosen_label + '\n\r')
	resultfile.write('Max_nr_nan: ' + str(max_nr_nan) + '\n\r')
	resultfile.write('Fixed_selection: ' + str(fixed_selection) + '\n\r')
	resultfile.write('Compressed_data: ' + str(compressed_data) + '\n\r')
	resultfile.write('Label path: ' + str(label_path) + '\n\r')
	resultfile.write('Data path: ' + str(data_path) + '\n\r')
	resultfile.flush()
	
	
	# Label_mapping holds key value pairs where key is the label and value its integer representation
	label_mapping = dataloader.get_valid_labels(label_path, choosen_label) # Labels from labels file only
	
	inverted_label_mapping = {}
	for key, value in label_mapping.items():
		inverted_label_mapping[value] = key
	
	
	#Get three structured separate dataframes from data sources
	trainframe, testframe, validationframe = dataloader.loadData(data_path, compressed_data, label_mapping, max_nr_nan, fixed_selection)
	
	if kfolds <= 1:
	
		frameinfo = dataloader.analyse_frame(trainframe)
		resultfile.write('\n\rTrainframe:\n\r')
		resultfile.write(frameinfo)
		frameinfo = dataloader.analyse_frame(testframe)
		resultfile.write('\n\r\n\rTestframe:\n\r')
		resultfile.write(frameinfo)
		frameinfo = dataloader.analyse_frame(validationframe)
		resultfile.write('\n\r\n\rValidationframe:\n\r')
		resultfile.write(frameinfo)
	
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

		# The model must choose between x classes.
		print('Number of unique labels, n_classes: ' + str(len(label_mapping)))
		#print('Number of unique trucks, n_classes: ' + str(int_labels.size))
		
		# optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=0.1) ?
		# optimizer = tensorflow.train.AdagradOptimizer(learning_rate=0.1) ?
		# optimizer = tensorflow.train.AdagradDAOptimizer(learning_rate=0.1, global_step= ?) global_step=train_steps?	
		# optimizer = tensorflow.train.AdamOptimizer(learning_rate=0.1) ?
		optimizer = tensorflow.train.ProximalAdagradOptimizer(learning_rate=0.01, l1_regularization_strength=0.001)
		#optimizer = 'Adagrad'
		
		classifier = tensorflow.estimator.DNNClassifier \
			(feature_columns=my_feature_columns,hidden_units=hidden_units,n_classes=len(label_mapping), dropout=dropout, batch_norm=False, optimizer=optimizer, model_dir='/data/Tensorflow/' + file_suffix) # , batch_norm=True ,optimizer=optimizer
		
		### Train the Model.
		print('\nModel training\n\r\n\r\n')
		#resultfile.write('\nModel training: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n\n\n')
		classifier.train(input_fn=lambda:dataloader.train_input_fn(trainset, int_labels_train, batch_size, nr_epochs), steps=train_steps)

		### Test the model
		print('\n\r\n\rModel testing\n\n\n')
		resultfile.write('\n\r\n\rModel testing\n\r')
		# Evaluate the model.
		
		eval_result = classifier.evaluate(input_fn=lambda:dataloader.eval_input_fn(testset, int_labels_test, batch_size))
		print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
		resultfile.write('\n\rTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
		resultfile.write('\n\rEval result:\n\r' + str(eval_result))
		
	else:
	
		foldframe = trainframe.append(testframe)
		foldtrainframe = pandas.DataFrame()
		foldtestframe = pandas.DataFrame()
		foldframe_list = []
	
		foldframe_list = dataloader.getFoldFrame(foldframe_list, kfolds, foldframe)
	
		#print('Foldframe_list: ' + str(len(foldframe_list)))
		#print(foldframe_list[0].head())
		
		frameinfo = dataloader.analyse_frame(validationframe)
		resultfile.write('\n\r\n\rValidationframe:\n\r')
		resultfile.write(frameinfo)
		
		# Validate model data
		validationset, labels_validate, label_mapping, int_labels_validate = \
			dataloader.get_model_data(validationframe, label_mapping, choosen_label)
		
		#sys.exit()
		
		testresults = []
	
		for testindex in range(kfolds):
	
			foldtrainframe, foldtestframe = dataloader.getFoldTrainFrames(foldframe_list, testindex)
			#print('Trainsize: ' + str(foldtrainframe.size))
			#print('Trainsize: ' + str(foldtestframe.size))
		
			frameinfo = dataloader.analyse_frame(foldtrainframe)
			resultfile.write('\n\rTrainframe:\n\r')
			resultfile.write(frameinfo)
			frameinfo = dataloader.analyse_frame(foldtestframe)
			resultfile.write('\n\r\n\rTestframe:\n\r')
			resultfile.write(frameinfo)
		
			# Train model data
			trainset, labels_training, label_mapping, int_labels_train = \
				dataloader.get_model_data(foldtrainframe, label_mapping, choosen_label)
			
			# Test model data
			testset, labels_test, label_mapping, int_labels_test = \
				dataloader.get_model_data(foldtestframe, label_mapping, choosen_label)
			
			### Model training
			my_feature_columns = []
			for key in trainset.keys():
				my_feature_columns.append(tensorflow.feature_column.numeric_column(key=key))

			# The model must choose between x classes.
			print('Number of unique labels, n_classes: ' + str(len(label_mapping)))
			#print('Number of unique trucks, n_classes: ' + str(int_labels.size))
			
			# optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=0.1) ?
			# optimizer = tensorflow.train.AdagradOptimizer(learning_rate=0.1) ?
			# optimizer = tensorflow.train.AdagradDAOptimizer(learning_rate=0.1, global_step= ?) global_step=train_steps?	
			# optimizer = tensorflow.train.AdamOptimizer(learning_rate=0.1) ?
			optimizer = tensorflow.train.ProximalAdagradOptimizer(learning_rate=0.01, l1_regularization_strength=0.001)
			#optimizer = 'Adagrad'
			
			classifier = tensorflow.estimator.DNNClassifier \
				(feature_columns=my_feature_columns,hidden_units=hidden_units,n_classes=len(label_mapping), dropout=dropout, batch_norm=False, optimizer=optimizer, model_dir='/data/Tensorflow/' + file_suffix) # , batch_norm=True ,optimizer=optimizer
			
			### Train the Model.
			print('\nModel training\n\r\n\r\n')
			classifier.train(input_fn=lambda:dataloader.train_input_fn(trainset, int_labels_train, batch_size, nr_epochs), steps=train_steps)

			### Test the model
			print('\n\r\n\rModel testing\n\n\n')
			resultfile.write('\n\r\n\rModel testing\n\r')
			
			eval_result = classifier.evaluate(input_fn=lambda:dataloader.eval_input_fn(testset, int_labels_test, batch_size))
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
	expected = list(int_labels_validate) #list(label_mapping.keys())
	predictions = classifier.predict(input_fn=lambda:dataloader.eval_input_fn(validationset, labels=None, batch_size=batch_size))
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
		#print(template.format(expected[class_id], 100 * probability, expec))
		resultfile.write('\n\r')
		resultfile.write(template.format(inverted_label_mapping[expected[class_id]], 100 * probability, inverted_label_mapping[expec]))
		number_of_validations += 1
		y_true.append(inverted_label_mapping[expec])
		y_predicted.append(inverted_label_mapping[expected[class_id]])
		y_probability.append(probability)
		
		if str(inverted_label_mapping[expected[class_id]]) == str(inverted_label_mapping[expec]):
			predictfile.write('Percent: ' + str(100 * probability) + '  ' + choosen_label + ': ' + str(inverted_label_mapping[expec]) + '\n\r')
			number_of_matches += 1
		#else:
		#	print('No match')

	#print(list(label_mapping.keys()))	
	confusion_matrix_result = confusion_matrix(y_true, y_predicted, labels=list(label_mapping.keys())) # labels=[0,1]
	print(confusion_matrix_result)
	dataloader.print_cm(confusion_matrix_result, list(label_mapping.keys()), file_suffix)
	dataloader.print_roc_curve(numpy.array(y_true), numpy.array(y_probability), list(label_mapping.keys()), file_suffix)
	
	predictfile.write('\n\rNumber of matches in percent: ' + str(100 * number_of_matches / number_of_validations))
	predictfile.write('\n\rTotal: ' + str(number_of_validations))
	predictfile.write('\n\rMatches: ' + str(number_of_matches))
	resultfile.write('\n\r******************************\n\r')
	resultfile.close()
	predictfile.close()
	
if __name__ == '__main__':
    tensorflow.logging.set_verbosity(tensorflow.logging.INFO)
    tensorflow.app.run(main) # So far only a dummy arguments...
	
	
	
	
	
	
