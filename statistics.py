


from os import listdir
from os.path import isfile, join
import pandas
import numpy
import math
import datetime
import matplotlib.pyplot as plt
from collections import OrderedDict




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
	dataframe.to_csv('frame_labels' + datestring + '.csv', sep=';', index = False, index_label = False)
		

# Plots 2D histograma for a choosen Label, one histogram per label value		
def Label_statistics_value(directory, label):

	datafiles = []
	for item in listdir(directory): 
		if isfile(join(directory, item)):
			datafiles.append(directory + item)

	dataframe_temp = pandas.read_csv(datafiles[0], sep=";", index_col=False)	
	dataframe = pandas.read_csv(datafiles[0], sep=";", index_col=False)	
		
	dataseries = dataframe_temp.pop(label)
	label_values = {}	
	for index, value in dataseries.iteritems():
		try:
			data_value = label_values[value]
			label_values[value] += 1
		except KeyError:
			label_values[value] = 1

	
	for key in label_values.keys():
		mean_values = pandas.Series()
		labelmask = (dataframe[label] == key)
		dataframe_label = dataframe.loc[labelmask]
		dataframe_label = dataframe_label.loc[:, '1_1':'20_20']
		for x in range(1, 21):
			for y in range(1, 21):
				column = str(x) + '_' + str(y)
				mean_values = mean_values.append(pandas.Series(dataframe_label.loc[:,column].mean()))
		
		mean_values = mean_values.replace(numpy.nan, 0.0)
		#print(mean_values.size)
		
		mean_values_np = mean_values.values
		#print(mean_values_np)
		mean_values_np = numpy.reshape(mean_values_np, (20,20), order='C')
		
		#mean_values.plot(kind='hist', bins=40, logy=True)
		#plt.hist2d(mean_values, mean_values, bins=(20, 20), cmap=plt.cm.jet, range=(20,20))
		#plt.axis((0,20,0,20))
		plt.imshow(mean_values_np, cmap=plt.cm.jet, origin='lower') # shape=(20,20)
		plt.colorbar()
		plt.title("Mean values column: " + key + ' \nNumber of values: ' + str(dataframe_label.size))
		
		#fig, ax = plt.subplots()
		plt.yticks([0.0,1.0, 2.0,3.0,4.0,5.0,6-0,7-0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0])
		plt.xticks([0.0,1.0, 2.0,3.0,4.0,5.0,6-0,7-0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0])
		
		plt.grid(b=True)
		plt.savefig("Histograms/Mean_values-" + key + ".png")
		plt.clf()
		
		
# Plots barplots for a choosen label, The ten highest values are presented for train. test and validation sets		
def Label_statistics(directory, label):


	datafiles = []
	for item in listdir(directory): 
		if isfile(join(directory, item)):
			datafiles.append(directory + item)

	dataframe = pandas.read_csv(datafiles[0], sep=";", index_col=False)

	dataframe = dataframe.set_index('truck_date')
	dataframe.index = pandas.to_datetime(dataframe.index)
		
	trainset = dataframe.loc['2016-1-1':'2016-5-31']
	testset = dataframe.loc['2016-6-1':'2016-6-10']
	validationset = dataframe.loc['2016-6-11':'2018-12-31']
		
	trainset = trainset.reset_index()
	testset = testset.reset_index()
	validationset = validationset.reset_index()
	
	dataseries_train = trainset.pop(label)
	label_dict_train = {}
	for index, value in dataseries_train.iteritems():
		try:
			data_value = label_dict_train[value]
			label_dict_train[value] += 1
		except KeyError:
			label_dict_train[value] = 1	
		
	labelframe_train = pandas.DataFrame()
	labelframe_train = labelframe_train.append(label_dict_train, ignore_index = True)
	labelframe_train = labelframe_train.sort_values(by = 0, ascending = False, axis = 1)
	
	dataseries_test = testset.pop(label)
	label_dict_test = {}
	for index, value in dataseries_test.iteritems():
		try:
			data_value = label_dict_test[value]
			label_dict_test[value] += 1
		except KeyError:
			label_dict_test[value] = 1	
		
	labelframe_test = pandas.DataFrame()
	labelframe_test = labelframe_test.append(label_dict_test, ignore_index = True)
	labelframe_test = labelframe_test.sort_values(by = 0, ascending = False, axis = 1)
	
	
	dataseries_val = validationset.pop(label)
	label_dict_val = {}
	for index, value in dataseries_val.iteritems():
		try:
			data_value = label_dict_val[value]
			label_dict_val[value] += 1
		except KeyError:
			label_dict_val[value] = 1	
		
	labelframe_val = pandas.DataFrame()
	labelframe_val = labelframe_val.append(label_dict_val, ignore_index = True)
	labelframe_val = labelframe_val.sort_values(by = 0, ascending = False, axis = 1)
	

	index_nr = 0
	for column in labelframe_train.columns:	
		if index_nr > 9:
			labelframe_train.pop(column)
		index_nr += 1
	
	index_nr = 0
	for column in labelframe_test.columns:	
		if index_nr > 9:
			labelframe_test.pop(column)
		index_nr += 1
	

	index_nr = 0
	for column in labelframe_val.columns:	
		if index_nr > 9:
			labelframe_val.pop(column)
		index_nr += 1
	
	print(labelframe_train.head(10))
	print(labelframe_test.head(10))
	print(labelframe_val.head(10))
	
	labelframe_train.plot(kind='bar')
	#plt.show()
	plt.title("Trainset " + label)
	plt.grid(True)
	plt.savefig("Histograms/Trainset_" + label + ".png")
	plt.clf()
	
	labelframe_test.plot(kind='bar')
	#plt.show()
	plt.title("Testset " + label)
	plt.grid(True)
	plt.savefig("Histograms/Testset_" + label + ".png")
	plt.clf()
	
	labelframe_val.plot(kind='bar')
	#plt.show()
	plt.title("Validationset " + label)
	plt.grid(True)
	plt.savefig("Histograms/Validationset_" + label + ".png")
	plt.clf()
	
	
		
# Plots a large set of plots, can be downselected by up to two labels and their values
def column_statistics(directory, zero_values_excluded = False, upper_limit = 0, label = '', specifics = '', label2 = '', specifics2 = ''):

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
		if label2 != '' and specifics2 != '':
			labelmask = (dataframe[label2] == specifics2)
			dataframe = dataframe.loc[labelmask]
	
	dataframe = dataframe.loc[:, '1_1':'20_20']
	#print(dataframe.head(10))

	
	# NaN statistics before replacements
	# For column
	number_of_nan_values_default = pandas.Series()
	for x in range(1, 21):
		for y in range(1, 21):
			number_of_nan_default = 0
			column = str(x) + '_' + str(y)
			values = dataframe.loc[:,column]
			for value in values:
				if math.isnan(value):
					number_of_nan_default += 1
			number_of_nan_values_default = number_of_nan_values_default.append(pandas.Series([number_of_nan_default]), ignore_index=True)
	# For row
	number_of_nan_values_row_default = pandas.Series()
	for index, row in dataframe.iterrows():
		number_of_nan_row_default = 0
		for x in range(1, 21):
			for y in range(1, 21):
				value = row[str(x) + '_' + str(y)]
				if math.isnan(value):
					number_of_nan_row_default += 1
		number_of_nan_values_row_default = number_of_nan_values_row_default.append(pandas.Series([number_of_nan_row_default]), ignore_index=True)
	
	
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
		if value > (std3_values.loc[index]):
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
	
	resultfile.write('\n\rFor all columns together: ' + zero_values + upper_limit_info + '\n\r')	
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
	
	number_of_nan_values_row_default.plot(kind='hist', bins=40, logy=True)
	#x1,x2,y1,y2 = plt.axis()
	#plt.axis((x1,x2,y1,400))
	plt.title("Number of NaN values row_default " + zero_values + upper_limit_info + ' \nNumber of values: ' + str(column_values.size))
	plt.grid(True)
	plt.savefig("Histograms/Number_of_NaN_values_row_default-" + zero_values + upper_limit_file + ".png")
	plt.clf()
	
	number_of_nan_values_default.plot(kind='hist', bins=40)
	#x1,x2,y1,y2 = plt.axis()
	#plt.axis((x1,x2,y1,400))
	plt.title("Number of NaN values column_default " + zero_values + upper_limit_info + ' \nNumber of values: ' + str(column_values.size))
	plt.grid(True)
	plt.savefig("Histograms/Number_of_NaN_values_column_default-" + zero_values + upper_limit_file + ".png")
	plt.clf()
	
	number_of_nan_values_row.plot(kind='hist', bins=40, logy=True)
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
	
	
# prints a set of histograms subsetted by sql_query (Should be: SELECT * FROM data_valid_all_labels WHERE ...)
def column_statisticsDB(mydb_connector, zero_values_excluded = False, upper_limit = 0, sql_query = 'Sql_query'):

	resultfile = open("Histograms/statistics_resultsDB.txt", "a")

	zero_values = 'with zero values'
	upper_limit_info = ''
	upper_limit_file = ''
	
	"""
	mydb = mysql.connector.connect(
		host="localhost",
		user="root",
		passwd="roghog",
		database="roger")
	"""
		
	dataframe = pandas.read_sql(sql_query, mydb_connector)

	dataframe = dataframe.loc[:, '1_1':'20_20']
	#print(dataframe.head(10))

	
	# NaN statistics before replacements
	# For column
	number_of_nan_values_default = pandas.Series()
	for x in range(1, 21):
		for y in range(1, 21):
			number_of_nan_default = 0
			column = str(x) + '_' + str(y)
			values = dataframe.loc[:,column]
			for value in values:
				if math.isnan(value):
					number_of_nan_default += 1
			number_of_nan_values_default = number_of_nan_values_default.append(pandas.Series([number_of_nan_default]), ignore_index=True)
	# For row
	number_of_nan_values_row_default = pandas.Series()
	for index, row in dataframe.iterrows():
		number_of_nan_row_default = 0
		for x in range(1, 21):
			for y in range(1, 21):
				value = row[str(x) + '_' + str(y)]
				if math.isnan(value):
					number_of_nan_row_default += 1
		number_of_nan_values_row_default = number_of_nan_values_row_default.append(pandas.Series([number_of_nan_row_default]), ignore_index=True)
	
	
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
		if value > (std3_values.loc[index]):
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
	
	resultfile.write('\n\rFor all columns together: ' + zero_values + upper_limit_info + '\n\r')	
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
	
	number_of_nan_values_row_default.plot(kind='hist', bins=40, logy=True)
	#x1,x2,y1,y2 = plt.axis()
	#plt.axis((x1,x2,y1,400))
	plt.title("Number of NaN values row_default " + zero_values + upper_limit_info + ' \nNumber of values: ' + str(column_values.size))
	plt.grid(True)
	plt.savefig("Histograms/Number_of_NaN_values_row_default-" + zero_values + upper_limit_file + ".png")
	plt.clf()
	
	number_of_nan_values_default.plot(kind='hist', bins=40)
	#x1,x2,y1,y2 = plt.axis()
	#plt.axis((x1,x2,y1,400))
	plt.title("Number of NaN values column_default " + zero_values + upper_limit_info + ' \nNumber of values: ' + str(column_values.size))
	plt.grid(True)
	plt.savefig("Histograms/Number_of_NaN_values_column_default-" + zero_values + upper_limit_file + ".png")
	plt.clf()
	
	number_of_nan_values_row.plot(kind='hist', bins=40, logy=True)
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
column_statistics('Compressed/Compressed_valid_all_labels/', False) #  'COUNTRY', 'USA' 	
column_statistics('Compressed/Compressed_valid_all_labels/', True) # 
column_statistics('Compressed/Compressed_valid_all_labels/', False, 3000) # 	
column_statistics('Compressed/Compressed_valid_all_labels/', True, 3000) # 
"""

#column_statistics('Compressed/', False, 0) # Compressed_valid_chassis	
#column_statistics('Compressed/', False, 0, 'COUNTRY', 'USA') # Compressed_valid_chassis
#column_statistics('Compressed/', False, 0, 'COUNTRY', 'USA', 'GEARBOX_', 'ATO2612D') # Compressed_valid_chassis	

#Label_statistics('Compressed/Compressed_valid_all_labels/', 'GEARBOX_') # GEARBOX_ ENGINE_TYPE COUNTRY TRUCK_TYPE BRAND_TYPE T_CHASSIS Compressed_valid_all_labels/
		
#labels_statistics('Data_original/')
		
#Label_statistics_value('Compressed/Compressed_valid_all_labels/', 'COUNTRY') # Compressed_valid_all_labels/
		
#column_statisticsDB(mydb_connector, zero_values_excluded = False, upper_limit = 0, sql_query = 'Sql_query'):
		
		
	
	
	
	







