clear all
close all
X = csvread('cleaned_continuous_diabetic_data_normalized.csv',1,0);
max_iters = 100;
K = 3;
assignments = zeros(size(X,1),10);
for i = 1:10
    initial_centroids = kMeansInitCentroids(X, K);
    [centroids, idx] = runkMeans(X, initial_centroids, max_iters);
    assignments(:,i) = idx;
end

filename = 'assignments.xlsx';
xlswrite(filename,assignments,1)

%{
% Gender, age
temp = [X(:,1) X(:,2)];
initial_centroids = kMeansInitCentroids(temp, K);
[centroids, idx] = runkMeans(temp, initial_centroids, max_iters);
assignments(:,1) = idx;

% Time in hospital, num lab procedures, num procedures
temp = [X(:,3) X(:,4) X(:,5)];
initial_centroids = kMeansInitCentroids(temp, K);
[centroids, idx] = runkMeans(temp, initial_centroids, max_iters);
assignments(:,2) = idx;

% Num medications, diabetes medication
temp = [X(:,6) X(:,12)];
initial_centroids = kMeansInitCentroids(temp, K);
[centroids, idx] = runkMeans(temp, initial_centroids, max_iters);
assignments(:,3) = idx;

filename = 'assignments.xlsx';
xlswrite(filename,assignments,1)
%}