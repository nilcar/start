


from os import listdir
from os.path import isfile, join
import pandas
import numpy
import math
import datetime
import matplotlib.pyplot as plt




# Prints statistical data to std out and writes a csv file with all labels merged from labels file and data source files
def labels_statistics(directory):

	label_path = 'Labels/'
	CSV_COLUMN_NAMES = ['A', 'B','truck_type', 'T_CHASSIS', 'E', 'truck_date', 'G', 'H', 'I', 'J', 'x_index', 'L', 'M', 'N', 'y_index', 'value', 'Q', 'R', 'S']

	# label mapping from labels file
	datafiles = []
	for item in listdir(label_path): 
		if isfile(join(label_path, item)):
			datafiles.append(label_path + item)
	print(datafiles)
	
	label_mapping = {}
	for datafile in datafiles:
		print(datafile)
		label_data = pandas.read_csv(datafile, sep=";")
		string_labels = label_data.pop('T_CHASSIS')
		for index, label in string_labels.items():
			label_mapping[label] = index
		#label_mapping = dict([(y,x+1) for x,y in enumerate(sorted(set(string_labels.tolist())))])
	
	# Label mapping from data source files
	datafiles = []
	for item in listdir(directory): 
		if isfile(join(directory, item)):
			datafiles.append(directory + item)
	
	nr = 1
	missing_labels = 0
	next_index = len(label_mapping) # Assumes that label_mapping was built ordered from 0
	for datafile in datafiles:
		#print(datafile)
		csv_data_file = pandas.read_csv(datafile, sep=";", names=CSV_COLUMN_NAMES, header=None, index_col=False)
		#print(csv_data_file.head())
		string_labels = csv_data_file.pop('T_CHASSIS')
		del csv_data_file
		print('Read cvsfile nr: ' + str(nr))
		nr += 1
		for label in string_labels:
			#print(label_mapping[label])
			try:
				intlabel = label_mapping[label] # Only to see if the label is possible to map from labels in label file...
			except KeyError:
				#print('Missing label: ' + label + ' new index: ' + str(next_index))
				label_mapping[label] = next_index
				next_index += 1
				missing_labels += 1

	print('Length of label_mapping: ' + str(len(label_mapping)))
	print('Missing labels nr: ' + str(missing_labels))
	print('Percent missing labels: ' + str(missing_labels / len(label_mapping)))
	
	dataframe = pandas.DataFrame()
	#dataframe['Chassi_nr'] = []
	#dataframe['Int_label'] = []
	dataframe['Chassi_nr'] = label_mapping.keys()
	dataframe['Int_label'] = label_mapping.values()
	datestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '--')
	datestring = datestring.replace(':', '-')
	dataframe.to_csv('volvo_labels' + datestring + '.csv', sep=';', index = False, index_label = False)
		

def column_statistics(directory, zero_values_excluded = False, upper_limit = 0, label = '', specifics = ''):

	resultfile = open("Histograms/statistics_results.txt", "a")

	zero_values = 'with zero values'
	upper_limit_info = ''
	upper_limit_file = ''
	
	datafiles = []
	for item in listdir(directory): 
		if isfile(join(directory, item)):
			datafiles.append(directory + item)

	dataframe = pandas.read_csv(datafiles[0], sep=";", index_col=False)
	
	if label != '' and specifics != '':
		labelmask = (dataframe[label] == specifics)
		dataframe = dataframe.loc[labelmask]
	
	dataframe = dataframe.loc[:, '1_1':'20_20']
	#print(dataframe.head(10))

	"""
	# NaN statistics
	# For column
	number_of_nan_values = pandas.Series()
	for x in range(1, 21):
		for y in range(1, 21):
			number_of_nan = 0
			column = str(x) + '_' + str(y)
			values = dataframe.loc[:,column]
			for value in values:
				if math.isnan(value):
					number_of_nan += 1
			number_of_nan_values= number_of_nan_values.append(pandas.Series([number_of_nan]), ignore_index=True)
	# For row
	number_of_nan_values_row = pandas.Series()
	for index, row in dataframe.iterrows():
		number_of_nan_row = 0
		for x in range(1, 21):
			for y in range(1, 21):
				value = row[str(x) + '_' + str(y)]
				if math.isnan(value):
					number_of_nan_row += 1
		number_of_nan_values_row= number_of_nan_values_row.append(pandas.Series([number_of_nan_row]), ignore_index=True)
	"""
	
	if upper_limit > 0:
		upper_limit_info = ' Upper_limit: ' + str(upper_limit)
		upper_limit_file = ' Upper_limit'
		for index, row in dataframe.iterrows():
			for x in range(1, 21):
				for y in range(1, 21):
					value = row[str(x) + '_' + str(y)]
					if value > upper_limit:
						row[str(x) + '_' + str(y)] = numpy.nan
						#print(row[str(x) + '_' + str(y)])
				
	#print(dataframe.head(10))
	
	if zero_values_excluded:
		dataframe = dataframe.replace(0.0, numpy.nan)
		zero_values = 'without zero values'
	
	"""
	# NaN statistics
	# For column
	number_of_nan_values = pandas.Series()
	for x in range(1, 21):
		for y in range(1, 21):
			number_of_nan = 0
			column = str(x) + '_' + str(y)
			values = dataframe.loc[:,column]
			for value in values:
				if math.isnan(value):
					number_of_nan += 1
			number_of_nan_values= number_of_nan_values.append(pandas.Series([number_of_nan]), ignore_index=True)
	"""
	
	# For row
	number_of_nan_values_row = pandas.Series()
	for index, row in dataframe.iterrows():
		number_of_nan_row = 0
		for x in range(1, 21):
			for y in range(1, 21):
				value = row[str(x) + '_' + str(y)]
				if math.isnan(value):
					number_of_nan_row += 1
		number_of_nan_values_row= number_of_nan_values_row.append(pandas.Series([number_of_nan_row]), ignore_index=True)
	
	number_of_nan_values = pandas.Series()
	min_values = pandas.Series()
	max_values = pandas.Series()
	mean_values = pandas.Series()
	std_values = pandas.Series()
	std3_values = pandas.Series()
	std_over_mean_values = pandas.Series()
	std2_over_mean_values = pandas.Series()
	std3_over_mean_values = pandas.Series()
	std5_over_mean_values = pandas.Series()
	max_over_std3_values = pandas.Series()
	number_max_over_std3 = 0
	column_values = pandas.Series()
	for x in range(1, 21):
		for y in range(1, 21):
			column = str(x) + '_' + str(y)
			number_over_mean_std = 0
			number_over_mean_std2 = 0
			number_over_mean_std3 = 0
			number_over_mean_std5 = 0
			number_over_std3 = 0
			number_of_nan = 0
			
			min_values = min_values.append(pandas.Series(dataframe.loc[:,column].min()))
			max_values = max_values.append(pandas.Series(dataframe.loc[:,column].max()))
			mean_values = mean_values.append(pandas.Series(dataframe.loc[:,column].mean()))
			std_values = std_values.append(pandas.Series(dataframe.loc[:,column].std()))
			
			values = dataframe.loc[:,column]
			for value in values:
				if math.isnan(value):
					number_of_nan += 1
			
				std = values.std()
				std2 = std * 2
				std3 = std * 3
				std5 = std * 5
				mean = values.mean()
				
				if value > std3:
					number_over_std3 += 1
				
				difference_from_mean = abs(value - mean)
				if difference_from_mean > std:
					number_over_mean_std += 1
				if difference_from_mean > std2:
					number_over_mean_std2 += 1
				if difference_from_mean > std3:
					number_over_mean_std3 += 1
				if difference_from_mean > std5:
					number_over_mean_std5 += 1
			
			number_of_nan_values= number_of_nan_values.append(pandas.Series([number_of_nan]), ignore_index=True)
			std3_values = std3_values.append(pandas.Series([number_over_std3]), ignore_index=True)
			std_over_mean_values = std_over_mean_values.append(pandas.Series([number_over_mean_std]), ignore_index=True)
			std2_over_mean_values = std2_over_mean_values.append(pandas.Series([number_over_mean_std2]), ignore_index=True)
			std3_over_mean_values = std3_over_mean_values.append(pandas.Series([number_over_mean_std3]), ignore_index=True)
			std5_over_mean_values = std5_over_mean_values.append(pandas.Series([number_over_mean_std5]), ignore_index=True)
			
			# All columns together
			column_values = column_values.append(values, ignore_index=True)

	for index, value in max_values.iteritems():
		if value > std3_values.loc[index]:
			number_max_over_std3 += 1
			
	max_over_std3_values = max_over_std3_values.append(pandas.Series([number_max_over_std3]), ignore_index=True)

	
	#print(column_values)
			
	value_min = column_values.min()
	value_max = column_values.max()
	value_mean = column_values.mean()
	value_std = column_values.std()
	std3_level = value_std * 3
	
	number_over_std3 = 0
	for value in column_values:
		if value > std3_level:
			number_over_std3 += 1
	
	resultfile.write('For all columns together: ' + zero_values + upper_limit_info + '\n\r')	
	resultfile.write('Number of values ' + zero_values + ': ' + str(column_values.size) + '\n\r')
	resultfile.write('Min value ' + zero_values + ': ' + str(value_min) + '\n\r')
	resultfile.write('Max value ' + zero_values + ': ' + str(value_max) + '\n\r')
	resultfile.write('Mean value ' + zero_values + ': ' + str(value_mean) + '\n\r')
	resultfile.write('Std value: ' + zero_values + ': ' + str(value_std) + '\n\r')
	resultfile.write('Over 3 std value: ' + zero_values + ': ' + str(number_over_std3) + '\n\r')
	
	max_over_std3_values.plot(kind='hist', bins=40)
	#x1,x2,y1,y2 = plt.axis()
	#plt.axis((x1,x2,y1,400))
	plt.title("Max value over std3 columns " + zero_values + upper_limit_info + ' \nNumber of values: ' + str(column_values.size))
	plt.grid(True)
	plt.savefig("Histograms/Max_value_over_std3_columns-" + zero_values + upper_limit_file + ".png")
	plt.clf()
	
	number_of_nan_values_row.plot(kind='hist', bins=40)
	#x1,x2,y1,y2 = plt.axis()
	#plt.axis((x1,x2,y1,400))
	plt.title("Number of NaN values row " + zero_values + upper_limit_info + ' \nNumber of values: ' + str(column_values.size))
	plt.grid(True)
	plt.savefig("Histograms/Number_of_NaN_values_row-" + zero_values + upper_limit_file + ".png")
	plt.clf()
	
	number_of_nan_values.plot(kind='hist', bins=40)
	#x1,x2,y1,y2 = plt.axis()
	#plt.axis((x1,x2,y1,400))
	plt.title("Number of NaN values column " + zero_values + upper_limit_info + ' \nNumber of values: ' + str(column_values.size))
	plt.grid(True)
	plt.savefig("Histograms/Number_of_NaN_values_column-" + zero_values + upper_limit_file + ".png")
	plt.clf()
	
	mean_values.plot(kind='hist', bins=40)
	#x1,x2,y1,y2 = plt.axis()
	#plt.axis((x1,x2,y1,400))
	plt.title("Mean values " + zero_values + upper_limit_info + ' \nNumber of values: ' + str(column_values.size))
	plt.grid(True)
	plt.savefig("Histograms/Mean_values-" + zero_values + upper_limit_file + ".png")
	plt.clf()
		
	min_values.plot(kind='hist', bins=40)
	#x1,x2,y1,y2 = plt.axis()
	#plt.axis((x1,x2,y1,400))
	plt.title("Min values " + zero_values + upper_limit_info + ' \nNumber of values: ' + str(column_values.size))
	plt.grid(True)
	plt.savefig("Histograms/Min_values-" + zero_values + upper_limit_file + ".png")
	plt.clf()
	
	max_values.plot(kind='hist', bins=40)
	#x1,x2,y1,y2 = plt.axis()
	#plt.axis((x1,x2,y1,400))
	plt.title("Max values " + zero_values + upper_limit_info + ' \nNumber of values: ' + str(column_values.size))
	plt.grid(True)
	plt.savefig("Histograms/Max_values-" + zero_values + upper_limit_file + ".png")
	plt.clf()
	
	std_values.plot(kind='hist', bins=40)
	#x1,x2,y1,y2 = plt.axis()
	#plt.axis((x1,x2,y1,400))
	plt.title("Standard deviation values " + zero_values + upper_limit_info + ' \nNumber of values: ' + str(column_values.size))
	plt.grid(True)
	plt.savefig("Histograms/Std_values-" + zero_values + upper_limit_file + ".png")
	plt.clf()
	
	std3_values.plot(kind='hist', bins=40)
	#x1,x2,y1,y2 = plt.axis()
	#plt.axis((x1,x2,y1,400))
	plt.title("Standard deviation times 3 values " + zero_values + upper_limit_info + ' \nNumber of values: ' + str(column_values.size))
	plt.grid(True)
	plt.savefig("Histograms/Std3_values-" + zero_values + upper_limit_file + ".png")
	plt.clf()
	#plt.show()
	
	std_over_mean_values.plot(kind='hist', bins=40)
	#x1,x2,y1,y2 = plt.axis()
	#plt.axis((x1,x2,y1,400))
	plt.title("Standard deviation over mean values " + zero_values + upper_limit_info + ' \nNumber of values: ' + str(column_values.size))
	plt.grid(True)
	plt.savefig("Histograms/Std_over_mean_values-" + zero_values + upper_limit_file + ".png")
	plt.clf()
	#plt.show()
	
	std2_over_mean_values.plot(kind='hist', bins=40)
	#x1,x2,y1,y2 = plt.axis()
	#plt.axis((x1,x2,y1,400))
	plt.title("Standard deviation 2 over mean values " + zero_values + upper_limit_info + ' \nNumber of values: ' + str(column_values.size))
	plt.grid(True)
	plt.savefig("Histograms/Std2_over_mean_values-" + zero_values + upper_limit_file + ".png")
	plt.clf()
	#plt.show()
	
	std3_over_mean_values.plot(kind='hist', bins=40)
	#x1,x2,y1,y2 = plt.axis()
	#plt.axis((x1,x2,y1,400))
	plt.title("Standard deviation 3 over mean values " + zero_values + upper_limit_info + ' \nNumber of values: ' + str(column_values.size))
	plt.grid(True)
	plt.savefig("Histograms/Std3_over_mean_values-" + zero_values + upper_limit_file + ".png")
	plt.clf()
	#plt.show()
	
	std5_over_mean_values.plot(kind='hist', bins=40)
	#x1,x2,y1,y2 = plt.axis()
	#plt.axis((x1,x2,y1,400))
	plt.title("Standard deviation 5 over mean values " + zero_values + upper_limit_info + ' \nNumber of values: ' + str(column_values.size))
	plt.grid(True)
	plt.savefig("Histograms/Std5_over_mean_values-" + zero_values + upper_limit_file + ".png")
	plt.clf()
	#plt.show()
	
	plt.title("mean values(x) std_values(y) " + zero_values + upper_limit_info + ' \nNumber of values: ' + str(column_values.size))
	plt.scatter(mean_values, std_values, c='r', marker='.') # s=2  (marker size) marker='.' (marker_type)
	plt.grid(True)
	plt.savefig("Histograms/scattered_mean_std_values-" + zero_values + upper_limit_file + ".png")
	plt.clf()

	
	
	resultfile.close()

	return

"""
column_statistics('Compressed/Compressed_valid_chassis/', False) # Compressed_single Compressed_valid_chassis	
column_statistics('Compressed/Compressed_valid_chassis/', True) # Compressed_single Compressed_valid_chassis	
column_statistics('Compressed/Compressed_valid_chassis/', False, 3000) # Compressed_single Compressed_valid_chassis	
column_statistics('Compressed/Compressed_valid_chassis/', True, 3000) # Compressed_single Compressed_valid_chassis
"""

column_statistics('Compressed/', False, 0) # Compressed_single Compressed_valid_chassis	

#column_statistics('Compressed/', False, 0, 'COUNTRY', 'USA') # Compressed_single Compressed_valid_chassis	
"""
column_statistics('Compressed/', True) # Compressed_single Compressed_valid_chassis	
column_statistics('Compressed/', False, 3000) # Compressed_single Compressed_valid_chassis	
column_statistics('Compressed/', True, 3000) # Compressed_single Compressed_valid_chassis
"""

		
#labels_statistics('Data_original/')
		
		
		
		
	
	
	
	







