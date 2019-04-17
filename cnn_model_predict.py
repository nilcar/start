

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
	
	#Get the dataframe to validate on
	validationframe = dataloader_predict.loadValidationFrame(data_directory)
	
	# Get validation model data
	validationset, chassis = dataloader_predict.get_model_data(validationframe, choosen_label)
	
	validate_data = validationset.values.astype(numpy.float32)
	
	cnn_validate_input_fn = tensorflow.estimator.inputs.numpy_input_fn(x={"x": validate_data}, y=None, num_epochs=1, shuffle=False)
	
	classifier = tensorflow.estimator.Estimator(model_fn=cnn_config_saved.cnn_model_dnn5CL3_fn, model_dir=model_path)
	
	### Evaluate the model
	predictions = classifier.predict(input_fn=cnn_validate_input_fn)
	
	number_of_validations = 0
	y_predicted = [] # pandas.Series()
	y_probability = [] # pandas.Series()

	for pred_dict in predictions:
		class_id = pred_dict['class_ids']
		probability = pred_dict['probabilities'][class_id]
		if number_of_validations < 5:
			print('Classsification: ' + str(class_id) + ' Probability: ' + str(probability * 100))
		number_of_validations += 1
		#y_predicted.append(pandas.Series([class_id])) # Classification
		#y_probability.append(pandas.Series([probability * 100])) # In percent
		y_predicted.append(class_id) # Classification
		y_probability.append(probability * 100) # In percent
		
		
		#print('Classsification: ' + str(pandas.Series([class_id])) + ' Probability: ' + str(pandas.Series([probability * 100])))
		
	dataframe = pandas.DataFrame()
	dataframe['Chassi_nr'] = chassis
	dataframe['Classification'] = pandas.Series(y_predicted).values
	dataframe['Prediction Value'] = pandas.Series(y_probability).values
	
	print(dataframe.head())
	
	
	
	
	
def main():
	
	#tensorflow.logging.set_verbosity(tensorflow.logging.INFO)
	#tensorflow.app.run()
	
	print('Tensorflow running')
	
	#cnn_model_predict.predict_on_model('Data2/V3/', '/data/Tensorflow/CNN/-repaired15000-CNN_Kfold5_Normal_800_CL3_ADAGR_01_LL20_NP_DR02_1234_DIL3_dnn5_1/', 'repaired')
	predict_on_model('Data2/V3/', '/data/Tensorflow/CNN/-repaired15000-CNN_Kfold5_Normal_800_CL3_ADAGR_01_LL20_NP_DR02_1234_DIL3_dnn5_1/', 'T_CHASSIS')
	

if __name__ == '__main__':
    tensorflow.logging.set_verbosity(tensorflow.logging.INFO)
    tensorflow.app.run(main)	
	
	
