% Copyright (C) 2014 Jin-Hwa Kim
%
% Author: Jin-Hwa Kim (jhkim@bi.snu.ac.kr)
% Created: August 16 2014
%
% Train the model for a given data.

function [edges, p] = arw_train(edges, x, y, r, lr, epsilon)

	p_init = zeros(1, size(edges, 1));
	p_init(1, 1:size(x,2)) = x;
	p_prev = p_init;
	max_diff = 1;
	iter = 1;
	% trace_mat = zeros(100, size(edges, 1));

	while max_diff > epsilon
		p = p_prev * edges;
		p_next = (1-r) * p_init + r * p;
		max_diff = max(abs(p_next - p_prev));
		% trace_mat(iter,:) = p_prev;
		p_prev = p_next;
		iter = iter + 1;
		if 0 == mod(iter, 100)
			fprintf('iter = %d, diff = %.5f\n', iter, max_diff);
		end
	end

	% Hebbian learning
	lf = ones(size(edges,1),1) * (p_next - p_init);
	edges = edges + lr * lf;

	% Only positive number is allowed.
	[row,col] = find(edges<0);
	for i = 1 : size(row, 1)
		edges(row(i),col(i)) = 0;
	end

	% Normalize to make the model follow the Markov chain constraint.
	% Sum of each row is one.
	edges = edges ./ (sum(edges, 2) * ones(size(edges, 1), 1)');

	% trace_mat = trace_mat(1:iter-1,:);
	p = p_next;

	fprintf('end with iter = %d, diff = %.5f\n', iter, max_diff);

end