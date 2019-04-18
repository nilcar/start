


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


import dataloader_predict # Utility functions for data handling
import cnn_config_saved # Holding definition of saved model

"""
Description:
The function is supposed to give predictions given data inputs and a given saved and pre-trained model

Input:
data_directory: The directory holding a single input file for predictions
model_path: The complete path to a saved model to predict on
choosen_label: The column name of the specific chassi id

Output:
A Pandas dataframe holding the result for the prediction.

Others:
Should be called from a main function, see below

"""

def predict_on_model(data_directory, model_path, choosen_label):

	#Get the dataframe to predict on
	validationframe = dataloader_predict.loadValidationFrame(data_directory)
	
	# Get prediction model data
	validationset, chassis = dataloader_predict.get_model_data(validationframe, choosen_label)
	validate_data = validationset.values.astype(numpy.float32)
	
	cnn_validate_input_fn = tensorflow.estimator.inputs.numpy_input_fn(x={"x": validate_data}, y=None, num_epochs=1, shuffle=False)
	
	classifier = tensorflow.estimator.Estimator(model_fn=cnn_config_saved.cnn_model_dnn5CL3_fn, model_dir=model_path)
	
	### Predict using the model
	predictions = classifier.predict(input_fn=cnn_validate_input_fn)
	
	number_of_validations = 0
	y_predicted = []
	y_probability = []

	for pred_dict in predictions:
		class_id = pred_dict['class_ids']
		probability = pred_dict['probabilities'][class_id]
		#if number_of_validations < 5:
		#	print('Classsification: ' + str(class_id) + ' Probability: ' + str(probability * 100))
		number_of_validations += 1
		y_predicted.append(class_id) # Classification
		y_probability.append(probability * 100) # In percent
		
	dataframe = pandas.DataFrame()
	dataframe['Chassi_nr'] = chassis.values
	dataframe['Classification'] = pandas.Series(y_predicted).values
	dataframe['Prediction Value'] = pandas.Series(y_probability).values
	
	#print(dataframe.head())
	
	return dataframe
	
	
	
	
	
def main():
	
	dataframe = predict_on_model('Data2/V3/', '/data/Tensorflow/CNN/-repaired15000-CNN_Kfold5_Normal_800_CL3_ADAGR_01_LL20_NP_DR02_1234_DIL3_dnn5_1/', 'T_CHASSIS')
	print(dataframe.head())

if __name__ == '__main__':
    tensorflow.logging.set_verbosity(tensorflow.logging.INFO)
    tensorflow.app.run(main)	
	
	
