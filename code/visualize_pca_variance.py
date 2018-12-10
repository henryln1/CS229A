import matplotlib.pyplot as plt



eigenvalues = [2.1779, 1.4633, 1.3622, 
			1.1666, 1.0146, 0.9590, 
			0.8924, 0.7333, 0.6845, 
			0.6200, 0.4911, 0.4350]

variances = [x / sum(eigenvalues) for x in eigenvalues]

names = ['Gender', 'Age', 'Time in Hospital', 
		'Number of Lab Procedures', 'Number of Procedures',
		'Number of Medications', 'Outpatient Visits', 'Emergency Room Visits',
		'Inpatient Visits', 'Number of Diagnoses', 'Change Detected', 'Diabetes Medication']

x_axis = [x for x in range(1, 13)]

plt.plot(x_axis, variances, color = 'r')
plt.xlabel('Principal Component')
plt.ylabel('Percentage Variance Captured')
plt.title('Variance per Principal Component')
plt.savefig('../visualization_pca.png')