% Copyright (C) 2014 Jin-Hwa Kim
%
% Author: Jin-Hwa Kim (jhkim@bi.snu.ac.kr)
% Created: August 16 2014
%
% Return the model of Attractive Random Walk.
% @param n number of nodes 

function [edges] = arw_model(n, sparseness)

	% Fully-connected and randomly weighted edges.
	edges = rand(n);

	% Sinks check and pruning according to a given sparseness.
	for i = 1 : size(edges, 1)
		if 1 && all(edges(i) == 0)
			disp('Zero vector!');
		end
		prune = randsample(size(edges, 2), round(size(edges, 2) * (1-sparseness)));
		edges(i, prune) = 0;
	end

	edges(1:10, 1:10) = 0;
	edges(1:10, end-9:end) = 0;
	edges(end-9:end, end-9:end) = 0;
	edges(end-9:end, 1:10) = 0;

	% Normalize to make the model follow the Markov chain constraint.
	% Sum of each row is one.
	edges = arw_norm(edges);

end