function centroids = kMeansInitCentroids(X, K)

order = randperm(K);
for i = order
    centroids(i,:) = X(i,:);
end

end