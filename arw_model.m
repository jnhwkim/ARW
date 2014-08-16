% Copyright (C) 2014 Jin-Hwa Kim
%
% Author: Jin-Hwa Kim (jhkim@bi.snu.ac.kr)
% Created: August 16 2014
%
% Return the model of Attractive Random Walk.
% @param n number of nodes in the layers (same number of nodes per a layer)
% @param l number of layers

function [edges] = arw_model(n, l, sparseness)

	% Fully-connected and randomly weighted edges.
	edges = rand(n * l, n * l);

	% Sinks check and pruning according to a given sparseness.
	for i = 1 : size(edges, 1)
		if 1 && all(edges(i) == 0)
			disp('Zero vector!');
		end
		prune = randsample(size(edges, 2), round(size(edges, 2) * (1-sparseness)));
		edges(i, prune) = 0;
	end

	% Normalize to make the model follow the Markov chain constraint.
	% Sum of each row is one.
	edges = edges ./ (sum(edges, 2) * ones(size(edges, 1), 1)');

end