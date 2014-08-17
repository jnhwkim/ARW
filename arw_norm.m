function [edges] = arw_norm(edges)

	% Only positive number is allowed.
	[row,col] = find(edges<0);
	for i = 1 : size(row, 1)
		edges(row(i),col(i)) = 0;
	end
	
	% Normalize to make the model follow the Markov chain constraint.
	% Sum of each row is one.
	edges = edges ./ (sum(edges, 2) * ones(size(edges, 1), 1)');
end