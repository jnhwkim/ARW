%% Machine Learning Online Class - Exercise 4 Neural Network Learning

%  Modified for the deep neural networks.
%  By Jin-Hwa Kim (jhkim@bi.snu.ac.kr)
% 

%% Initialization
clear ; close all; clc

%% Verbose
verbose = false;
initial = true;
iter = 300;

%% Setup the parameters you will use for this exercise
input_layer_size  = 784;  % 20x20 Input Images of Digits
hidden_layer_size = [300 300];   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
                          
%% =========== Learning History (iter=100) ==============
%  Test Error           Base     ReLu   LReL    Dropout                              
%  300x300              89.61    55.53  88.28   62.09

%% =========== Learning History (iter=300) ==============
%  Test Error           Base     ReLu   LReL    Dropout
%  300x300              91.28           92.63   

disp(datestr(now));

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

%% Loading the data
addpath('../mnist');
X = loadMNISTImages('train-images-idx3-ubyte')';
y = loadMNISTLabels('train-labels-idx1-ubyte');
X_test = loadMNISTImages('t10k-images-idx3-ubyte')';
y_test = loadMNISTLabels('t10k-labels-idx1-ubyte');
y(y==0) = 10;
y_test(y_test==0) = 10;

% Load Small Data (5000x400)
% load('ex4data1.mat');

m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

if verbose
    displayData(X(sel, :));

    fprintf('Program paused. Press enter to continue.\n');
    pause;
end


%% ================ Part 6: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

if initial
    fprintf('\nInitializing Neural Network Parameters ...\n')

    num_hidden_layers = size(hidden_layer_size, 2);
    initial_Theta = cell(num_hidden_layers, 1);
    initial_nn_params = [];

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
      initial_Theta{i} = randInitializeWeights(input_size, output_size);

      % Unroll parameters
      initial_nn_params = [initial_nn_params; initial_Theta{i}(:)];
    end
else
    load('/Users/Calvin/Github/ARW/nn/history/iter100/300x300lrel_nn_params.mat');
    initial_nn_params = nn_params;
end

if verbose
%% =============== Part 7: Implement Backpropagation ===============
%  Once your cost matches up with ours, you should proceed to implement the
%  backpropagation algorithm for the neural network. You should add to the
%  code you've written in nnCostFunction.m to return the partial
%  derivatives of the parameters.
%
    fprintf('\nChecking Backpropagation... \n');

    %  Check gradients by running checkNNGradients
    checkNNGradients;

    fprintf('\nProgram paused. Press enter to continue.\n');
    pause;
end


%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
batch_size = 1;
options = optimset('MaxIter', batch_size, 'Display', 'off');

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

nn_params = initial_nn_params;
accuracy = zeros(iter, 1);
tic();

for i = 1 : iter                           
    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    [nn_params, cost] = fmincg(costFunction, nn_params, options);

    % Obtain Theta back from nn_params
    Theta = vec2theta(nn_params, input_layer_size, hidden_layer_size, num_labels);

    pred = predict(Theta, X_test);
    accuracy(i) = mean(double(pred == y_test)) * 100;
    elapse = toc();
    fprintf('Iteration %4i | Accuracy: %4.2f | Time: %8.1f\r', ...
        i*batch_size, accuracy(i), elapse);
end

% Save the results
save('history/300x300drop_accuracy.mat','accuracy');
save('history/300x300drop_nn_params.mat','nn_params');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Part 9: Visualize Weights =================
%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta{1}(:, 2:end));

