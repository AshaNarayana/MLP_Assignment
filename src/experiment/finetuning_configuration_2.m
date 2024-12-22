
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      Testing MLP parameters (2)     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


close all, clear all, clc 



% Data + preprocessing
data = load('caltech101_silhouettes_28.mat');
X = data.X / 255;
Y = full(ind2vec(data.Y(:)' + 1));
names = data.classnames;


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
hiddenLayer = 'logsig';
outputLayer = 'softmax';
costFunction = 'crossentropy';

hiddenUnitsList = [50, 200, 500];

trainingFList = {'traincgb', 'trainscg'};
lratesList = [0.5, 0.1, 0.01, 0.001];


% Initialize results
results = [];

for h = 1:length(hiddenUnitsList)
    for tr = 1:length(trainingFList)
        for lr = 1:length(lratesList)
            
            accuracies = zeros(1, 5); % Initialize accuracy tracking

            for run = 1:5
                network = feedforwardnet(hiddenUnitsList(h));
                network.layers{1}.transferFcn = hiddenLayer;
                network.layers{2}.transferFcn = outputLayer; 
                network.performFcn = costFunction; 
                
                network.outputs{:}.processFcns = {};
                
                % Set training parameters
                network.trainFcn = trainingFList{tr};
                network.trainParam.lr = lratesList(lr); % less than 1.0 and greater than 10^-6

                network.trainParam.epochs = 500; 
                network.trainParam.showWindow = true;  % Show training window
                
                
                % Train the network
                [network, ~] = train(network, trainData, trainLabels);
                
                
                % Validate the network
                valPredictions = network(valData);
                [~, valPredictedClasses] = max(valPredictions);
                [~, valActualClasses] = max(valLabels);
                valAccuracy = mean(valPredictedClasses == valActualClasses) * 100;
                fprintf('Validation Accuracy: %.2f%%\n', valAccuracy);
                
                
                % Test the network
                testPredictions = network(testData);
                [~, testPredictedClasses] = max(testPredictions);
                [~, testActualClasses] = max(testLabels);
                
                accuracy = mean(testPredictedClasses == testActualClasses) * 100;
                fprintf('Test Accuracy: %.2f%%\n', accuracy);
                accuracies(run) = accuracy;
            end
            

            % Store the results
            results = [results; hiddenUnitsList(h),  trainingFList(tr), lratesList(lr), mean(accuracies)];

            % Display progress
            fprintf('Hidden Units: %d, Training Function: %s, LR: %s, Mean Accuracy: %.2f%%\n\n', ...
                    hiddenUnitsList(h), trainingFList{tr}, lratesList(lr), mean(accuracies));
        end
    end
end

writecell(results, 'results/report_finetuning_configuration_2.csv')