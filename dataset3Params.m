function [C, sigma] = dataset3Params(X, y, Xval, yval)
%select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns the optimal C and 
%   sigma based on a cross-validation set.

C = 0.3;
sigma = 0.1;
C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
mat = zeros(64,3);

% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))

for i = 1:8
    for j = 1:8

        model = svmTrain(X, y, C_values(i), @(X, Xval)gaussianKernel(X, Xval, sigma_values(j)));
        predictions = svmPredict(model, Xval);
        errPrediction = mean(double(predictions ~= yval));
        mat((i-1)*8+j,:) = [C_values(i) sigma_values(j) errPrediction];
    end
end

row =find(mat(:, 3)==min(mat(:, 3)));
C = mat(row,1);
sigma = mat(row,2);


