function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   J = SIGMOID(z) computes the sigmoid of z.

method = 2;
if 1 == method
    g = 1.0 ./ (1.0 + exp(-z));
elseif 2 == method
    g = z;
    g(z<0) = 0.01 * z(z<0);
end
end
