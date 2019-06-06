function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
z = X * theta;
g = sigmoid(z);

pos = -y.' * log(g);
neg = -(1 - y.') * log(1 - g);

% You need to return the following variables correctly 
grad = (g - y).' * X / m;

J = (pos + neg) / m + (theta(2:n).' * theta(2:n)) * lambda / (2 * m);
grad(2:n) = grad(2:n) + theta(2:n).' * lambda / m;

%grad2(2:n) = (g(2:n,:) - y(2:n,:)).' * X(2:n,:) / m + theta(2:n).' * lambda / m;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
