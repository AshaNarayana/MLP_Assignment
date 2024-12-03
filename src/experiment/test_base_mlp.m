hiddenUnitsList = [50, 200, 500];
outputFuncs = {'logsig', 'softmax'};
costFuncs = {'mse', 'crossentropy'};
splits = [0.8, 0.1, 0.1; 0.4, 0.2, 0.4; 0.1, 0.1, 0.8];
numRuns = 3;

% Initialize results
results = [];

% Iterate over configurations
for h = 1:length(hiddenUnitsList)
    for f = 1:length(outputFuncs)
        for c = 1:length(costFuncs)
            for s = 1:size(splits, 1)
                % Get current split ratios
                trainRatio = splits(s, 1);
                valRatio = splits(s, 2);
                testRatio = splits(s, 3);

                % Call the base MLP function
                meanAccuracy = base_mlp(X_normalized, Y_onehot, ...
                                             hiddenUnitsList(h), ...
                                             outputFuncs{f}, ...
                                             costFuncs{c}, ...
                                             trainRatio, valRatio, testRatio, ...
                                             numRuns);

                % Store the results
                results = [results; hiddenUnitsList(h), f, c, s, meanAccuracy];

                % Display progress
                fprintf('Hidden Units: %d, OutputFunc: %s, CostFunc: %s, Split: [%0.1f/%0.1f/%0.1f], Mean Accuracy: %.2f%%\n', ...
                        hiddenUnitsList(h), outputFuncs{f}, costFuncs{c}, ...
                        splits(s, :), meanAccuracy);
            end
        end
    end
end
