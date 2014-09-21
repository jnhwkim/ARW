function Theta = vec2theta(nn_params, ...
                           input_layer_size, ...
                           hidden_layer_size, ...
                           num_labels)
%VEC2THETA Summary of this function goes here
%   Detailed explanation goes here

% Obtain Theta back from nn_params
num_hidden_layers = size(hidden_layer_size, 2);
Theta = cell(num_hidden_layers, 1);
pos = 0;
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
  Theta{i} = reshape(nn_params(pos + 1 : ...
                     pos + output_size * (input_size + 1)), ...
                     output_size, (input_size + 1));
  pos = pos + output_size * (input_size + 1);
end

end

