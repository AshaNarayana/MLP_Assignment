
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Experiments for  Configuration 1  %
%   - logsig for the hidden layer     %
%   - logsig for the output layer     %
%   - mean squared error              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


close all, clear all, clc 

import MLP.*


% Data + preprocessing
data = load('datasets/caltech101_silhouettes_28.mat');
X = data.X / 255;
Y = full(ind2vec(data.Y(:)' + 1));
names = data.classnames;

% Initialize results
results = [];


% Get current split ratios
trainRatio = 0.8;
valRatio = 0.1;
testRatio = 0.1;


% Randomly split the dataset
[trainInd, valInd, testInd] = dividerand(size(X, 1), trainRatio, valRatio, testRatio);
trainData = X(trainInd, :)';
valData = X(valInd, :)';
testData = X(testInd, :)';
trainLabels = Y(:, trainInd);
valLabels = Y(:, valInd);
testLabels = Y(:, testInd);


% MLP
hiddenUnits = 500;
hiddenLayer = 'logsig';
outputLayer = 'logsig';
costFunction = 'mse';

network = feedforwardnet(hiddenUnits);
network.layers{1}.transferFcn = hiddenLayer;
network.layers{2}.transferFcn = outputLayer; 
network.performFcn = costFunction; 

network.outputs{end}.processFcns = {};


% Set training parameters
network.trainParam.epochs = 100; 
network.trainFcn = 'traingdm'; % trainlm breaks
network.trainParam.lr = 0.01;
network.trainParam.mc = 0.8;

network.trainParam.showWindow = true;  % Show training window


% Train the network
[network, ~] = train(network, trainData, trainLabels);


% Validate the network
valPredictions = network(valData);
[~, valPredictedClasses] = max(valPredictions);
[~, valActualClasses] = max(valLabels);
valAccuracy = mean(valPredictedClasses == valActualClasses) * 100;

valError = gsubtract(valLabels, valPredictions); 
fprintf('Validation Error: %.2f%%\n', valError);
fprintf('Validation Accuracy: %.2f%%\n', valAccuracy);