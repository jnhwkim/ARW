function y = arw_test(edges, x, r, epsilon)

	p_init = zeros(1, size(edges, 1));
	x = x / sum(x);
	p_init(1, 1:size(x, 2)) = x;
	p_prev = p_init;
	iter = 1;
	max_diff = 1;

	while max_diff > epsilon
		% Propagation
		p = p_prev * edges;
		p_next = (1-r) * p_init + r * p;

		max_diff = max(abs(p_next - p_prev));
		% trace_mat(iter,:) = p_prev;
		p_prev = p_next;
		iter = iter + 1;
		if 0 == mod(iter, 1000)
			fprintf('iter = %d, diff = %.5f\n', iter, max_diff);
		end
	end

	y = p_next;

end