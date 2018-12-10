
warning ("off");
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

function [all_theta] = oneVsAll(X, y, num_labels, lambda, c, num_iters)

m = size(X, 1);                                      % Some useful variables
n = size(X, 2);
all_theta = zeros(num_labels, n);                % You need to return the following variable correctly

% ====================== YOUR CODE HERE ======================
initial_theta = zeros(n, 1); 
options = optimset('GradObj', 'on', 'MaxIter', num_iters);
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

function [X,y,all_theta,accuracy,p, probs] = learn(data, X, y, feature_cols, predict_class, lambda, iters)
[m,n] = size(X);

y_ = (y == predict_class);
m = size(X,1);

[all_theta] = oneVsAll(X, y, 2, lambda, predict_class, iters);
[p, probs] = predict([zeros(size(all_theta,1),1) all_theta], X);
p = p .- 1;
accuracy = mean(double(p == y_)) * 100;
end

function [accuracy_1, accuracy_2, accuracy_3, multi_accuracy, p1, p2, p3, multi_p] = get_accuracies(X, y, all_theta1, all_theta2, all_theta3)

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
#all_probs = [p1 p2];
[dummy, multi_p] = max(all_probs, [], 2);
multi_p = multi_p .- 1;
multi_accuracy = mean(double(multi_p == y)) * 100;
end

lambdas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
num_iters = [800, 1600, 3200]
trainingP = [];
validationP = [];
testingP = [];
trainingR = [];
validationR = [];
testingR = [];
trainingF = [];
validationF = [];
testingF = [];
training_accuracy = [];
validation_accuracy = [];
testing_accuracy = [];
training_multi = [];
validation_multi = [];
testing_multi = [];
lambda = 0.1;
#for i = 1:length(num_iters)

# Training
#lambda = lambdas(i)
iters = 800;
[m,n] = size(X_train);
[X1,Y1,all_theta1, accuracy_1, p1] = learn(data, X_train, y_train, 1:n, 0, lambda, iters);
[X2,Y2,all_theta2, accuracy_2, p2] = learn(data, X_train, y_train, 1:n, 1, lambda, iters);
[X3,Y3,all_theta3, accuracy_3, p3] = learn(data, X_train, y_train, 1:n, 2, lambda, iters);

training_accuracy = [training_accuracy; [accuracy_1 accuracy_2 accuracy_3]]
#training_accuracy = [training_accuracy; [accuracy_1 accuracy_2]]

all_probs = [p1 p2 p3];
#all_probs = [p1 p2];
[dummy, multi_p] = max(all_probs, [], 2);
multi_p = multi_p .- 1;

alltruth0 = 1.0*sum(y_train == 0);
alltruth1 = 1.0*sum(y_train == 1);
alltruth2 = 1.0*sum(y_train == 2);

allchosen0 = 1.0*sum(multi_p == 0);
allchosen1 = 1.0*sum(multi_p == 1);
allchosen2 = 1.0*sum(multi_p == 2);

correct0 = 1.0*sum((multi_p == 0).*(y_train==0));
correct1 = 1.0*sum((multi_p == 1).*(y_train==1));
correct2 = 1.0*sum((multi_p == 2).*(y_train==2));

trainP0 = correct0/allchosen0;
trainP1 = correct1/allchosen1;
trainP2 = correct2/allchosen2;

trainR0 = correct0/alltruth0;
trainR1 = correct1/alltruth1;
trainR2 = correct2/alltruth2;

trainF0 = 2.0*((trainP0*trainR0)/(trainP0+trainR0));
trainF1 = 2.0*((trainP1*trainR1)/(trainP1+trainR1));
trainF2 = 2.0*((trainP2*trainR2)/(trainP2+trainR2));

trainingP = [trainingP ; [trainP0 trainP1 trainP2]]
trainingR = [trainingR ; [trainR0 trainR1 trainR2]]
trainingF = [trainingF ; [trainF0 trainF1 trainF2]]
#trainingP = [trainingP ; [trainP0 trainP1]]
#trainingR = [trainingR ; [trainR0 trainR1]]
#trainingF = [trainingF ; [trainF0 trainF1]]

multi_accuracy = mean(double(multi_p == y_train)) * 100;
training_multi = [training_multi ; multi_accuracy]

[va_1, va_2, va3, multi_va, vp1, vp2, vp3, multi_pv] = get_accuracies(X_val, y_val, all_theta1, all_theta2, all_theta3);
[ta_1, ta_2, ta_3, multi_ta, tp1, tp2, tp3, multi_pt] = get_accuracies(X_test, y_test, all_theta1, all_theta2, all_theta3);

# Validation
validation_accuracy = [validation_accuracy ; [va_1 va_2 va_3]]
#validation_accuracy = [validation_accuracy ; [va_1 va_2]];
validation_multi = [validation_multi ; multi_va];

alltruth0 = 1.0*sum(y_val == 0);
alltruth1 = 1.0*sum(y_val == 1);
alltruth2 = 1.0*sum(y_val == 2);

allchosen0 = 1.0*sum(multi_pv == 0);
allchosen1 = 1.0*sum(multi_pv == 1);
allchosen2 = 1.0*sum(multi_pv == 2);

correct0 = 1.0*sum((multi_pv == 0).*(y_val==0));
correct1 = 1.0*sum((multi_pv == 1).*(y_val==1));
correct2 = 1.0*sum((multi_pv == 2).*(y_val==2));

valP0 = correct0/allchosen0;
valP1 = correct1/allchosen1;
valP2 = correct2/allchosen2;

valR0 = correct0/alltruth0;
valR1 = correct1/alltruth1;
valR2 = correct2/alltruth2;

valF0 = 2.0*((valP0*valR0)/(valP0+valR0));
valF1 = 2.0*((valP1*valR1)/(valP1+valR1));
valF2 = 2.0*((valP2*valR2)/(valP2+valR2));

validationP = [validationP ; [valP0 valP1 valP2]]
validationR = [validationR ; [valR0 valR1 valR2]]
validationF = [validationF ; [valF0 valF1 valF2]]

#validationP = [validationP ; [valP0 valP1]]
#validationR = [validationR ; [valR0 valR1]]
#validationF = [validationF ; [valF0 valF1]]

# Testing
testing_accuracy = [testing_accuracy ; [ta_1 ta_2 ta_3]]
#testing_accuracy = [testing_accuracy ; [ta_1 ta_2]]
testing_multi = [testing_multi ; multi_ta]

alltruth0 = 1.0*sum(y_test == 0);
alltruth1 = 1.0*sum(y_test == 1);
alltruth2 = 1.0*sum(y_test == 2);

allchosen0 = 1.0*sum(multi_pt == 0);
allchosen1 = 1.0*sum(multi_pt == 1);
allchosen2 = 1.0*sum(multi_pt == 2);

correct0 = 1.0*sum((multi_pt == 0).*(y_test==0));
correct1 = 1.0*sum((multi_pt == 1).*(y_test==1));
correct2 = 1.0*sum((multi_pt == 2).*(y_test==2));

testP0 = correct0/allchosen0;
testP1 = correct1/allchosen1;
testP2 = correct2/allchosen2;

testR0 = correct0/alltruth0;
testR1 = correct1/alltruth1;
testR2 = correct2/alltruth2;

testF0 = 2.0*((testP0*testR0)/(testP0+testR0));
testF1 = 2.0*((testP1*testR1)/(testP1+testR1));
testF2 = 2.0*((testP2*testR2)/(testP2+testR2));

testingP = [testingP ; [testP0 testP1 testP2]]
testingR = [testingR ; [testR0 testR1 testR2]]
testingF = [testingF ; [testF0 testF1 testF2]]

#testingP = [testingP ; [testP0 testP1]]
#testingR = [testingR ; [testR0 testR1]]
#testingF = [testingF ; [testF0 testF1]]
