function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

fprintf('--------------------------------------------------------------------------------\n');
fprintf('start searching best [C, sigma] values\n');
error_min = inf;
val = [0.01 0.03 0.1 0.3 1 3 10 30];

for C = val
  for sigma = val
    fprintf('Train and evaluate (on cross validation set) for\n[_C, _sigma] = [%f %f]\n', C, sigma);
    mod = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    z = mean(double(svmPredict(mod, Xval) ~= yval));
    fprintf('prediction error: %f\n', z);
    if( z <= error_min )
      fprintf('error_min updated!\n');
      C = C;
      sigma = sigma;
      error_min = z;
      fprintf('[C, sigma] = [%f %f]\n', C, sigma);
    end
    fprintf('--------\n');
  end
end

fprintf('\nfinish searching.\nBest value [C, sigma] = [%f %f] with prediction error = %f\n\n', C, sigma, error_min);
fprintf('--------------------------------------------------------------------------------\n');

% =========================================================================

end