clear all
close all
X = csvread('cleaned_continuous_diabetic_data.csv',1,0);
max_iters = 100;

X_norm = featureNormalize(X(:,1:12));
filename = 'cleaned_continuous_diabetic_data_normalized.csv';
xlswrite(filename,[X_norm X(:,13)],1)
