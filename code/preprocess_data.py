import csv
import pandas as pd


important_features = ['race', 'gender', 'age', 'admission_type_id', 
					'discharge_disposition_id', 'admission_source_id', 'time_in_hospital', 
					'num_lab_procedures', 'num_procedures', 
					'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'diag_1', 
					'diag_2', 'diag_3', 'number_diagnoses', 'max_glu_serum', 'A1Cresult', 'metformin', 
					'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 
					'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
					'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 
					'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 
					'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed', 'readmitted']

race_mappings = {
	'?': 0,
	'Caucasian': 1,
	'AfricanAmerican': 2,
	'Asian': 3,
	'Hispanic': 4,
	'Other': 5
}

gender_mappings = {
	'Unknown/Invalid': 0,
	'Female': 1,
	'Male': 2
}

age_mappings = {
	'[0-10)': 5,
	'[10-20)': 15,
	'[20-30)': 25,
	'[30-40)': 35,
	'[40-50)': 45,
	'[50-60)': 55,
	'[60-70)': 65, 
	'[70-80)': 75,
	'[80-90)': 85,
	'[90-100)': 95,
	'[100-110)': 105
}

readmitted_mappings = {
	'NO': 0,
	'<30': 1,
	'>30': 2
}

medications_mappings = {
	'No': 0,
	'NO': 0,
	'Steady': 1,
	'Up': 2,
	'Down': 3
}

change_in_medications_mappings = {
	'No': 0,
	'Ch': 1
}

diabetes_meds_prescribed_mappings = {
	'No': 0,
	'Yes': 1
}

special_cases = ['max_glu_serum', 'A1Cresult'] #if it's None, set it equal to 0, otherwise remove the > sign and leave the number


# def process_first_line(line):
# 	'''
# 	works with column heads, ignoring this for now
# 	'''
# 	return line


# 	useless_features = ['patient_nbr', 'weight', 'encounter_id']

# 	for feature in useless_features:
# 		line.remove(feature)
# 	return line


def process_line(line):
	#line is a list of every element in a row of the csv file
	line[0] = race_mappings[line[0]] #race
	line[1] = gender_mappings[line[1]] #gender
	line[2] = age_mappings[line[2]] #age

	if line[17] == 'None':
		line[17] = 0
	else:
		line[17] = line[17][1:]
	if line[18] == 'None':
		line[18] = 0
	else:
		line[18] = line[18][1:]
	for i in range(19, 42):
		line[i] = medications_mappings[line[i]]

	line[42] = change_in_medications_mappings[line[42]]
	line[43] = diabetes_meds_prescribed_mappings[line[43]]
	line[44] = readmitted_mappings[line[44]]
	return line




def process_csv_file(in_file_name, out_file_name):
	'''
	takes in the diabetes file and preprocesses it into a format easier to apply ML methods on.

	'''
	#first we remove the irrelevant columns:
	f = pd.read_csv(in_file_name)
	keep_columns = important_features
	new_f = f[keep_columns]
	new_f.to_csv(in_file_name[:-4] + '_intermediate.csv', index = False)

	with open(in_file_name[:-4] + '_intermediate.csv', 'r') as f_in:
		reader = csv.reader(f_in, delimiter=",")
		with open(out_file_name, 'w') as f_out:
			writer = csv.writer(f_out, delimiter=',')
			line_count = 0
			for line in reader:
				if line_count == 0:
					writer.writerow(line)
					line_count += 1
					continue
				cleaned_line = process_line(line)
				writer.writerow(cleaned_line)
				line_count += 1
				if line_count % 1000 == 0:
					print(str(line_count) + ' lines processed.')
			f_out.close()
		f_in.close()

def main():
	print('Processing raw diabetes file...')
	in_file_name = '../dataset_diabetes/diabetic_data.csv'
	out_file_name = '../dataset_diabetes/cleaned_diabetic_data.csv'
	process_csv_file(in_file_name, out_file_name)



if __name__ == "__main__":
	main()