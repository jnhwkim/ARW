% Copyright (C) 2014 Jin-Hwa Kim
%
% Author: Jin-Hwa Kim (jhkim@bi.snu.ac.kr)
% Created: August 16 2014
%
% Train the model for a given data.

addpath('mnist');

feature_size = 784;
layers = 3;

%% Loading the data
images = loadMNISTImages('train-images-idx3-ubyte')';
labels = loadMNISTLabels('train-labels-idx1-ubyte');

[edges] = arw_model(feature_size, layers, 0.1);

y = labels(1);
r = 0.9;
lr = 0.1;
epsilon = 0.1 .^ 9;
sample_size = 1000;
trace_mat = zeros(sample_size, size(edges,2));

for i = 1:sample_size
	x = images(i,:);
	[edges, p] = arw_train(edges, x, y, r, lr, epsilon);
	trace_mat(i,:) = p;
end

trace_ordered = zeros(size(trace_mat,1),size(trace_mat,2));

idx = 1;
for i = 0 : 9
	sel = find(labels(1:sample_size,1)==i)';
	sel = sel(end-10:end);
	trace_ordered(idx:idx+size(sel, 2)-1, :) = trace_mat(sel,:);
	idx = idx + size(sel, 2);
end

f = figure(1);
% subplot('Position', [0 0 0.5 1]);
% imshow(edges*256, jet(32));
subplot('Position', [0 0 1 1]);
imshow(trace_mat(:,:)*256, jet(32));
set(f, 'Position', [0 0 600 300]);