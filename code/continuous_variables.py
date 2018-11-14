import pandas as pd



continuous_column_names = ['gender', 'age', 'time_in_hospital', 
						'num_lab_procedures', 'num_procedures', 
						'num_medications', 'number_outpatient', 
						'number_emergency', 'number_inpatient', 
						'number_diagnoses', 'change', 'diabetesMed',
						'readmitted']



def main():
	print('Splitting off continuous variables...')
	in_file_name = '../dataset_diabetes/cleaned_diabetic_data.csv'
	out_file_name = '../dataset_diabetes/cleaned_continuous_diabetic_data.csv'
	f = pd.read_csv(in_file_name)
	new_f = f[continuous_column_names]
	new_f.to_csv(out_file_name, index = False)




if __name__ == "__main__":
	main()