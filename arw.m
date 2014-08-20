% Copyright (C) 2014 Jin-Hwa Kim
%
% Author: Jin-Hwa Kim (jhkim@bi.snu.ac.kr)
% Created: August 16 2014
%
% Train the model for a given data.

addpath('mnist');

feature_size = 784;
layers = 3;
% feature_size = 10;
% layers = 5;

%% Loading the data
images = loadMNISTImages('train-images-idx3-ubyte')';
labels = loadMNISTLabels('train-labels-idx1-ubyte');

[edges] = arw_model(feature_size * layers + 10, 1);
edges_init = edges;

r = 0.5;
lr = 1;
epsilon = 0.1 .^ 5;
sample_size = 1000;
trace_mat = zeros(sample_size, size(edges,2));

for i = 1:sample_size
	y_idx = mod(9+labels(i),10)+1;
	y = zeros(1,10); y(y_idx) = 1;
    x = images(i,:);
	% x = y;
	[edges, p] = arw_train(edges, x, y, r, lr, epsilon);
	trace_mat(i,:) = p;
end

trace_ordered = zeros(100,size(trace_mat,2));

idx = 1;
for i = 0 : 9
	sel = find(labels(1:sample_size,1)==i)';
	sel = sel(end-9:end);
	trace_ordered(idx:idx+size(sel, 2)-1, :) = trace_mat(sel,:);
	idx = idx + size(sel, 2);
end

f = figure(1);
subplot('Position', [0 0.5 0.5 0.5]);
imshow(edges_init*64, jet(32));
subplot('Position', [0 0 0.5 0.5]);
imshow(edges*64, jet(32));
subplot('Position', [0.5 0.5 0.5 0.5]);
imshow(trace_mat*256, jet(32));
subplot('Position', [0.5 0 0.5 0.5]);
imshow(trace_ordered(:,end-9:end)*256, jet(32));
set(f, 'Position', [0 200 1200 600]);

correct = 0;
for i = sample_size+1:sample_size+200
	y_idx = mod(9+labels(i),10)+1;
    x = images(i,:);
	% x = zeros(1,10); x(y_idx) = 1;
	y = arw_test(edges, x, r, epsilon);
	[c,idx] = max(y(end-9:end));
	predict_idx = mod(9+idx,10)+1;
	if y_idx == predict_idx
		correct = correct + 1;
	end
end

fprintf('Accuracy = %.2f\n', correct / sample_size);