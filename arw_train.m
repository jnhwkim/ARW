% Copyright (C) 2014 Jin-Hwa Kim
%
% Author: Jin-Hwa Kim (jhkim@bi.snu.ac.kr)
% Created: August 16 2014
%
% Train the model for a given data.

function [edges, p] = arw_train(edges, x, y, r, lr, epsilon)

	learning_at_end = false;
	learning_at_prop = false;
	ar_learning = true;

	p_init = zeros(1, size(edges, 1));
	% x = x / sum(x);
	% p_init(1, 1:size(x,2)) = -x;
	p_init(1, end-size(y,2)+1:end) = y;
	p_prev = p_init;
	max_diff = 1;
	iter = 1;
	% trace_mat = zeros(100, size(edges, 1));

	while max_diff > epsilon
		% Propagation
		p = p_prev * edges;
		p_next = (1-r) * p_init + r * p;
		
		% Hebbian learning at each propagation
		if learning_at_prop
			lf = p_prev' * p_next;
			edges = edges + lr * lf;
			% Normalize to make the model follow the Markov chain constraint.
			% Sum of each row is one.
			edges = arw_norm(edges);
		end

		max_diff = max(abs(p_next - p_prev));
		% trace_mat(iter,:) = p_prev;
		p_prev = p_next;
		iter = iter + 1;
		if 0 == mod(iter, 1000)
			fprintf('iter = %d, diff = %.5f\n', iter, max_diff);
		end
	end

	p = p_next;

	% Attractor-repeller learning
	if ar_learning
		for i = 1 : size(edges, 1)
			target = find(edges(i,:)~=0);
			edges(i,target) = edges(i,target) + lr * (edges(i,target) .* p(1,target));
		end
		% Normalize to make the model follow the Markov chain constraint.
		% Sum of each row is one.
		edges = arw_norm(edges);
	end

	% Hebbian learning at end
	if learning_at_end
		lf = p_prev' * p_next;
		edges = edges + lr * lf;
		% Normalize to make the model follow the Markov chain constraint.
		% Sum of each row is one.
		edges = arw_norm(edges);
	end

	% trace_mat = trace_mat(1:iter-1,:);

	fprintf('end with iter = %d, diff = %.5f\n', iter, max_diff);

end