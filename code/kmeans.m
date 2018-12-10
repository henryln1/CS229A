clear all
close all
X = csvread('cleaned_continuous_diabetic_data_normalized.csv',1,0);
max_iters = 100;
K = 3;
assignments = zeros(size(X,1),3);

% Gender, age
temp = [X(:,1) X(:,2) X(:,end)];
initial_centroids = kMeansInitCentroids(temp, K);
[centroids, idx] = runkMeans(temp, initial_centroids, max_iters);
assignments(:,1) = idx;

% Time in hospital, num lab procedures, num procedures
temp = [X(:,3) X(:,4) X(:,5) X(:,end)];
initial_centroids = kMeansInitCentroids(temp, K);
[centroids, idx] = runkMeans(temp, initial_centroids, max_iters);
assignments(:,2) = idx;

% Num medications, diabetes medication
temp = [X(:,6) X(:,12) X(:,end)];
initial_centroids = kMeansInitCentroids(temp, K);
[centroids, idx] = runkMeans(temp, initial_centroids, max_iters);
assignments(:,3) = idx;

filename = 'assignments.xlsx';
xlswrite(filename,assignments,1)