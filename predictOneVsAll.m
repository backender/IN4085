function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Make predictions using learned logistic regression parameters (one-vs-all).
% Set p to a vector of predictions (from 1 to num_labels).
%
%     [x11 x12]   [t11 t12]'   [p11 ... p1k]
% A = [x21 x22] * [t21 t22]  = [p21 ... p2k] 
%     [xm1 xm2]   [tk1 tk2]    [pm1 ... pmk]
% = 
%      [p11 ... p1k]'                                                     [k_pmax1]
% max( [p21 ... p2k]  ) = [(pmax1, k_pmax1) ... (pmaxm, k_pmaxm)]' = _2 = [k_...  ]
%      [pm1 ... pmk]                                                      [k_pmaxm]

A = sigmoid(X * all_theta');
[maxSig, maxk] = max(A');
p = maxk';


% =========================================================================


end
