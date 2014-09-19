function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
num_hidden_layers = size(hidden_layer_size, 2);
Theta = cell(num_hidden_layers, 1);
pos = 0;
for i = 1 : num_hidden_layers
  if 1 == i
    input_size = input_layer_size;
  else
    input_size = hidden_layer_size(1, i);
  end
  if num_hidden_layers == i
    output_size = num_labels;
  else
    output_size = hidden_layer_size(1, i + 1);
  end
  Theta{i} = reshape(nn_params(pos + 1 : pos + output_size * (input_size + 1)), ...
                     output_size, (input_size + 1));
  pos = pos + output_size * (input_size + 1);
end

% Setup some useful variables
m = size(X, 1);

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ ones(m,1), X ];
Z = 0;

for i = 1 : num_hidden_layers
	if 1 == i
		a = sigmoid(X * Theta{i}');
	else
		a = [ones(size(a, 1), 1), a];
		a = sigmoid(a * Theta{i}');
	end
	Z = Z + sum(sum(Theta{i}(:,2:end).^2));
end

h = a;
yy = zeros( size(y,1), num_labels );
for i = 1 : size(y,1)
	yy(i,y(i)) = 1;
end

J = sum(sum(-yy.*log(h)-(1.-yy).*log(1.-h))) / m;
J = J + Z * lambda / (2*m);

D = cell(num_hidden_layers, 1);
Theta_grad = cell(num_hidden_layers, 1);

for t = 1 : m
	a = cell(num_hidden_layers + 1, 1);
	z = cell(num_hidden_layers + 1, 1);
	d = cell(num_hidden_layers + 1, 1);

	for i = 1 : num_hidden_layers
		if 1 == i
			a{i} = X(t, :)';
		else
			a{i} = [1; sigmoid(z{i})];
		end
		z{i+1} = Theta{i} * a{i};
		if size(Theta, 1) == i
			a{i+1} = sigmoid(z{i+1});
		end
	end

	for i = num_hidden_layers + 1 : -1 : 2
		if num_hidden_layers + 1 == i
			yy = zeros(num_labels, 1);
			yy(y(t)) = 1;
			d{i} = a{i} - yy;
		else
			d{i} = Theta{i}' * d{i+1} .* [0; sigmoidGradient(z{i})];
			d{i} = d{i}(2:end);
		end
	end

	for i = 1 : num_hidden_layers
		if 1 == t
			D{i} = zeros(size(d{i+1} * a{i}'));
		end
		D{i} = D{i} + d{i+1} * a{i}';
	end
end

grad = [];
for i = 1 : num_hidden_layers
	D{i} = D{i} / m;
	Theta{i}(:,1) = zeros(size(Theta{i}, 1), 1);
	Theta_grad{i} = D{i} + lambda * Theta{i} / m;
	% Unroll gradients
	grad = [grad; Theta_grad{i}(:)];
end

end
