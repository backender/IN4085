function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% Compute the cost of a particular choice of theta.
% Set J to the cost.
% Compute the partial derivatives and set grad to the partial
% derivatives of the cost w.r.t. each parameter in theta
%

hx = sigmoid(X * theta);
J = (-y .* log(hx)) - ((1-y) .* log(1-hx));
J = sum(J) / m;
grad = (X' * (sigmoid(X * theta) - y)) / m;

% Theta for regularization
thetaT = theta;
thetaT(1) = 0;

correctionCost = sum(thetaT .^ 2) * (lambda / (2 * m));
J = J + correctionCost;

correctionGrad = thetaT * (lambda / m);
grad = grad + correctionGrad;


grad = grad(:);

end
