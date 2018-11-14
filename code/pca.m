% GRADED FUNCTION: pca
function [U, S] = pca(X)

[m, n] = size(X);    % Useful values
U = zeros(n);        % Return the following variables correctly.
S = zeros(n);

% ====================== YOUR CODE HERE ======================
[U,S,V] = svd((1/m)*X'*X);

% =============================================================
end 