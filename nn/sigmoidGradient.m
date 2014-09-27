function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).
% ¥ò' = ¥ò(1-¥ò)

method = 2;
if 1 == method
    g = sigmoid(z) .* ( ones(size(z)) - sigmoid(z) );
elseif 2 == method
    g = ones(size(z));
    g(z<0) = 0.01;
end
end
