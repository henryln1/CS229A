import csv




k_means_assignment_csv = '../dataset_diabetes/assignments.csv'
dataset_csv = '../dataset_diabetes/cleaned_diabetic_data.csv'


original_labels_dict = {
	'0': set(),
	'1': set(),
	'2': set()
}


k_means_labels_dict = {
	'0': set(),
	'1': set(),
	'2': set()
}

with open(dataset_csv, 'r') as f:
	csv_reader = csv.reader(f, delimiter=',')
	line_count = 0
	for row in csv_reader:
		if row[0] == 'race':
			continue
		grouping = row[-1]
		original_labels_dict[grouping].add(line_count)
		line_count += 1

with open(k_means_assignment_csv, 'r') as f:
	csv_reader = csv.reader(f, delimiter=',')	
	line_count = 0
	for row in csv_reader:
		grouping = row[0]
		grouping = str(int(grouping) - 1)
		k_means_labels_dict[grouping].add(line_count)
		line_count += 1



print('Size of each groupings')
print('Original Label 0: ', len(original_labels_dict['0']))
print('Original Label 1: ', len(original_labels_dict['1']))
print('Original Label 2: ', len(original_labels_dict['2']))

print('K Means Label 0: ', len(k_means_labels_dict['0']))
print('K Means Label 0: ', len(k_means_labels_dict['1']))
print('K Means Label 0: ', len(k_means_labels_dict['2']))

print('Grouping 0')
print('Overlap with K_means label 0: ', len(original_labels_dict['0'].intersection(k_means_labels_dict['0'])) / len(original_labels_dict['0']))
		
print('Overlap with K_means label 1: ', len(original_labels_dict['0'].intersection(k_means_labels_dict['1'])) / len(original_labels_dict['0']))

print('Overlap with K_means label 2: ', len(original_labels_dict['0'].intersection(k_means_labels_dict['2'])) / len(original_labels_dict['0']))

print('Grouping 1')
print('Overlap with K_means label 0: ', len(original_labels_dict['1'].intersection(k_means_labels_dict['0'])) / len(original_labels_dict['1']))
		
print('Overlap with K_means label 1: ', len(original_labels_dict['1'].intersection(k_means_labels_dict['1'])) / len(original_labels_dict['1']))

print('Overlap with K_means label 2: ', len(original_labels_dict['1'].intersection(k_means_labels_dict['2'])) / len(original_labels_dict['1']))

print('Grouping 2')
print('Overlap with K_means label 0: ', len(original_labels_dict['2'].intersection(k_means_labels_dict['0'])) / len(original_labels_dict['2']))
		
print('Overlap with K_means label 1: ', len(original_labels_dict['2'].intersection(k_means_labels_dict['1'])) / len(original_labels_dict['2']))

print('Overlap with K_means label 2: ', len(original_labels_dict['2'].intersection(k_means_labels_dict['2'])) / len(original_labels_dict['2']))





