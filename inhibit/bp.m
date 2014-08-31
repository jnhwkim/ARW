%% Simulation for discovery of the inhibitory signal.

function history = bp(num_of_features, dropout)
%num_of_features = 10;

%% Populates X
X = zeros(pow2(num_of_features), num_of_features);
% 0 0 1; 0 1 0
% 0 1 1; 1 0 0
% 1 0 1; 1 1 0
% 1 1 1; 0 0 0

for i = 1 : size(X, 1) - 1
	x = zeros(num_of_features);
	reminder = i;
	for j = 1 : num_of_features
		X(i, j) = floor(reminder / pow2(num_of_features - j));
		reminder = reminder - X(i, j) * pow2(num_of_features - j);
	end
end

%% Populates y
%% First feature is an inhibitory signal..
y = zeros(size(X, 1), 1);

for i = 1 : size(X, 1)
	if 1 == X(i, 1)
		y(i) = 0;
	elseif 1 == max(X(i, 2:end))
		y(i) = 1;
	end
end

[y, X];

diff = 1;
lr = 0.01;
w = (rand(num_of_features, 1) - 0.5);
history = cell(3,1);
history{1} = zeros(1000, num_of_features);
history{2} = zeros(1000, 1);
iter = 1;
while iter < 500
	diff = 0;
	for i = 1 : size(X, 1)
		idx = randsample(num_of_features, floor(num_of_features * dropout));
		X_s = X(i,:);
		X_s(idx) = 0;

		z = 0 + X_s * w;
		a = relu(z);
		cost = y(i) - a;
		adj = (1-a) * a * X_s' * cost;
		w = w + lr * adj;
		diff = diff + (y(i) - a)^2;
	end
	history{1}(iter,:) = w;
	if 1 == max(isnan(w))
		break;
	end
	history{2}(iter) = diff;
	iter = iter + 1;
end

history{1} = history{1}(1:iter-1,:);
history{2} = history{2}(1:iter-1,:);

figure(1);
plot(history{1});
figure(2);
plot(history{2});
