function y = relu(x)
    if x >= -0.5
        y = x + 0.5;
    else 
        y = 0;
    end
end