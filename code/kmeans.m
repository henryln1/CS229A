clear all
close all
X = csvread('cleaned_continuous_diabetic_data.csv',1,0);
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