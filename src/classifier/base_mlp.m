
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Base architecture MLP (function)  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function meanAccuracy = base_mlp(X, Y, hiddenUnits, outputFunc, costFunc, trainRatio, valRatio, testRatio, numRuns)
    % Randomly split the dataset
    [trainInd, valInd, testInd] = dividerand(size(X, 1), trainRatio, valRatio, testRatio);
    trainData = X(trainInd, :)';
    valData = X(valInd, :)';
    testData = X(testInd, :)';
    trainLabels = Y(:, trainInd);
    valLabels = Y(:, valInd);
    testLabels = Y(:, testInd);

    % Initialize accuracy tracking
    accuracies = zeros(1, numRuns);

    for run = 1:numRuns
        % Create and configure network
        net = patternnet(hiddenUnits);
        net.layers{1}.transferFcn = 'logsig'; % Hidden layer
        net.layers{2}.transferFcn = outputFunc; % Output layer
        net.performFcn = costFunc; % Cost function

        % Assign data splits
        net.divideFcn = 'divideind';
        net.divideParam.trainInd = trainInd;
        net.divideParam.valInd = valInd;
        net.divideParam.testInd = testInd;

        % Set training parameters
        net.trainParam.epochs = 2000;
        net.trainParam.lr = 0.05;
        net.trainParam.showWindow = true;  % Show training window
    
        % Train the network
        [net, ~] = train(net, trainData, trainLabels);

        % Validate the network
        valPredictions = net(valData);
        [~, valPredictedClasses] = max(valPredictions);
        [~, valActualClasses] = max(valLabels);
        valAccuracy = mean(valPredictedClasses == valActualClasses) * 100;

        % TODO Validation accuracy is very poort for this settings, we need to
        % fine tune it.
        fprintf('Validation Accuracy for run %d: %.2f%%\n', run, valAccuracy);

        % Test and calculate accuracy
        predictions = net(testData);
        [~, predictedClasses] = max(predictions);
        [~, actualClasses] = max(testLabels);
        accuracies(run) = mean(predictedClasses == actualClasses) * 100;
    end

    % Calculate mean accuracy
    meanAccuracy = mean(accuracies);
end