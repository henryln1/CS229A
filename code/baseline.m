
data = csvread('cleaned_continuous_diabetic_data.csv',1,0);
size(data,1)
m_train = int32(size(data,1)*0.6)
m_val = m_train + int32((size(data,1)*0.2))
m_test = size(data,1)
X_train = data(1:m_train, 1:size(data,2)-1);
y_train = data(1:m_train, size(data,2));
X_val = data(m_train+1:m_val, 1:size(data,2)-1);
y_val = data(m_train+1:m_val, size(data,2));
X_test = data(m_val+1:m_test,1:size(data,2)-1);
y_test = data(m_val+1:m_test, size(data,2));
[m, n] = size(X_train);

function g = sigmoid(z)

g = zeros(size(z));     % return this correctly

g = exp(-1 .* z) + 1;
g = 1.0 ./ g;

end

function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y);                % Number of training examples
J = 0;                        % Set J to the cost
grad = zeros(size(theta));    % Set grad to the gradient
h = sigmoid(X*theta);
J = sum(((-1.*y).*log(h)) - ((1-y).*log(1-h))) ./ m;
J = J + (lambda / (2*m))*(dot(theta, theta) - (theta(1)*theta(1)));
grad = (sum(X .* (h - y), 1) ./ m)';
grad_reg = (lambda/(1.*m))*theta;
grad_reg(1) = 0;
grad = grad .+ grad_reg;

end

function [all_theta] = oneVsAll(X, y, num_labels, lambda, c)

m = size(X, 1);                                      % Some useful variables
n = size(X, 2);
all_theta = zeros(num_labels, n);                % You need to return the following variable correctly

% ====================== YOUR CODE HERE ======================
initial_theta = zeros(n, 1); 
options = optimset('GradObj', 'on', 'MaxIter', 250);
[all_theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)),initial_theta, options);
% =============================================================
end

function [p, probs] = predict(all_theta, X)

m = size(X, 1);                                % Useful values
num_labels = size(all_theta, 1);
p = zeros(size(X, 1), 1);                      % Return the following variable correctly 
h = sigmoid(X*all_theta);
probs = h(:,2);
[dummy p] = max(h, [], 2);
end

function [X,y,all_theta,accuracy,p, probs] = learn(data, X, y, feature_cols, predict_class, lambda)
[m,n] = size(X);
#for i = 1:size(feature_cols,2)
#    X = [X data(:,feature_cols(i))];
#endfor
y_ = (y == predict_class);
m = size(X,1);

[all_theta] = oneVsAll(X, y, 2, lambda, predict_class);
[p, probs] = predict([zeros(size(all_theta,1),1) all_theta], X);
p = p .- 1;
accuracy = mean(double(p == y_)) * 100;
end

function [accuracy_1, accuracy_2, accuracy_3, multi_accuracy] = get_accuracies(X, y, all_theta1, all_theta2, all_theta3)

y_ = (y == 0);
[p1, probs] = predict([zeros(size(all_theta1,1),1) all_theta1], X);
p1 = p1 .- 1;
accuracy_1 = mean(double(p1 == y_)) * 100;

y_ = (y == 1);
[p2, probs] = predict([zeros(size(all_theta2,1),1) all_theta2], X);
p2 = p2 .- 1;
accuracy_2 = mean(double(p2 == y_)) * 100;

y_ = (y == 2);
[p3, probs] = predict([zeros(size(all_theta3,1),1) all_theta3], X);
p3 = p3 .- 1;
accuracy_3 = mean(double(p3 == y_)) * 100;

all_probs = [p1 p2 p3];
[dummy, multi_p] = max(all_probs, [], 2);
multi_p = multi_p .- 1;
multi_accuracy = mean(double(multi_p == y)) * 100;
end

lambda = 3;
[m,n] = size(X_train)
[X1,Y1,all_theta1, accuracy_1, p1] = learn(data, X_train, y_train, 1:n, 0, lambda);
[X2,Y2,all_theta2, accuracy_2, p2] = learn(data, X_train, y_train, 1:n, 1, lambda);
[X3,Y3,all_theta3, accuracy_3, p3] = learn(data, X_train, y_train, 1:n, 2, lambda);

accuracy_1
accuracy_2
accuracy_3
all_probs = [p1 p2 p3];
[dummy, multi_p] = max(all_probs, [], 2);
multi_p = multi_p .- 1;
multi_accuracy = mean(double(multi_p == y_train)) * 100

[va_1, va_2, va_3, multi_va] = get_accuracies(X_val, y_val, all_theta1, all_theta2, all_theta3);
[ta_1, ta_2, ta_3, multi_ta] = get_accuracies(X_test, y_test, all_theta1, all_theta2, all_theta3);
va_1
va_2
va_3
multi_va 

ta_1
ta_2
ta_3
multi_ta




