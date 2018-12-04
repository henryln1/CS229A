function idx = findClosestCentroids(X, centroids)

K = size(centroids, 1);
idx = zeros(size(X,1), 1);

dist = zeros(size(X,1), K);
for i = 1:K
    dist(:,i) = sum(bsxfun(@minus, X(:,1:end-1), centroids(i,1:end-1)).^2, 2);
end
[temp, idx] = min(dist');
idx = idx';
end