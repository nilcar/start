


from os import listdir
from os.path import isfile, join
import pandas
import datetime
import matplotlib.pyplot as plt




# Prints statistical data to std out and writes a csv file with all labels merged from labels file and data source files
def labels_statistics(directory):

	label_path = 'Labels/'
	CSV_COLUMN_NAMES = ['A', 'B','truck_type', 'truck_id', 'E', 'truck_date', 'G', 'H', 'I', 'J', 'x_index', 'L', 'M', 'N', 'y_index', 'value', 'Q', 'R', 'S']

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
		string_labels = csv_data_file.pop('truck_id')
		del csv_data_file
		print('Read cvsfile nr: ' + str(nr))
		nr += 1
		for label in string_labels:
			#print(label_mapping[label])
			try:
				intlabel = label_mapping[label] # Only to see if the label is possible to map
			except KeyError:
				#print('Missing label: ' + label + ' new index: ' + str(next_index))
				label_mapping[label] = next_index
				next_index += 1
				missing_labels += 1

	print('Length of label_mapping: ' + str(len(label_mapping)))
	print('Missing labels nr: ' + str(missing_labels))
	print('Percent missing labels: ' + str(missing_labels / len(label_mapping)))
	
	dataframe = pandas.DataFrame()
	dataframe['Chassi_nr'] = []
	dataframe['Int_label'] = []
	dataframe['Chassi_nr'] = label_mapping.keys()
	dataframe['Int_label'] = label_mapping.values()
	datestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '--')
	datestring = datestring.replace(':', '-')
	dataframe.to_csv('volvo_labels' + datestring + '.csv', sep=';', index = False, index_label = False)
		
		
		
#labels_statistics('Testdata/')	
#labels_statistics('Data_original/')
		
		
		
		
	
	
	
	







