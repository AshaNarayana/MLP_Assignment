
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiments for  Configuration 1    %
%   - logsig for the hidden layer     %
%   - logsig for the output layer     %
%   - mean squared error              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


close all, clear all, clc 

import MLP.*


% Parameters to test
numRuns = 5;

hiddenUnitsList = [50, 200, 500];
splits = [0.8, 0.1, 0.1; 0.4, 0.2, 0.4; 0.1, 0.1, 0.8];


% Data
data = load('datasets/caltech101_silhouettes_28.mat');
X = data.X / 255;
Y = full(ind2vec(data.Y(:)' + 1));
names = data.classnames;


% Initialize results
results = [];


% Test experiments
for h = 1:length(hiddenUnitsList)
    for s = 1:size(splits, 1)

        % Initialize accuracy tracking
        accuracies = zeros(1, numRuns);
        mses = zeros(1, numRuns);

        for run = 1:numRuns
    
            % Display progress
            fprintf('Run %d, Hidden Units: %d, Split: [%0.1f/%0.1f/%0.1f]\n', ...
                run, hiddenUnitsList(h), splits(s, :));
    
            % Get current split ratios
            trainRatio = splits(s, 1);
            valRatio = splits(s, 2);
            testRatio = splits(s, 3);
    
            % Randomly split the dataset
            [trainInd, valInd, testInd] = dividerand(size(X, 1), trainRatio, valRatio, testRatio);
            trainData = X(trainInd, :)';
            valData = X(valInd, :)';
            testData = X(testInd, :)';
            trainLabels = Y(:, trainInd);
            valLabels = Y(:, valInd);
            testLabels = Y(:, testInd);
    
            % MLP
            mlp = MLP(hiddenUnitsList(h),'logsig', 'logsig', 'mse');

            % Set training parameters: Only for Configuration 1
            mlp.network.trainFcn = 'traingdx';
            mlp.network.trainParam.lr = 0.1;
            mlp.network.trainParam.mc = 0.9;

            % Training and Testing
            mlp = mlp.train(trainData, trainLabels, valData, valLabels);
            mlp = mlp.test(testData, testLabels);

            accuracies(run)  = mlp.accuracy;
            mses(run) = mlp.mserror;
        end

        % Compute mean accuracy
        mean_accuracies = mean(accuracies);
        mean_mses = mean(mses);
        fprintf('\nMean MSE: %.2f', mean_mses);
        fprintf('\nMean Accuracy: %.2f%%\n\n', mean_accuracies);

        % Store the results
        results = [results; hiddenUnitsList(h), trainRatio, valRatio, testRatio, mean_mses, mean_accuracies];
    end
end

writematrix(results, 'results/report_experiment_configuration_1.csv')