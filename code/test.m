clear all
close all
data = csvread('cleaned_continuous_diabetic_data.csv',1,0);
X = data(:,1:size(data,2)-1);
[X_norm, mu, sigma] = featureNormalize(X);
[U S] = pca(X_norm);
sum(S)
sum(sum(S))