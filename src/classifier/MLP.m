
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Base architecture for all MLPs   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef MLP
    properties
        hiddenUnits
        hiddenLayer
        outputLayer
        costFunction

        network
        valAccuracy
        valMmserror
        accuracy
        mserror
    end
    methods
        function self = MLP(hiddenUnits, hiddenLayer, outputLayer, costFunction)
            self.hiddenUnits = hiddenUnits;
            self.hiddenLayer = hiddenLayer;
            self.outputLayer = outputLayer;
            self.costFunction = costFunction;

            self.network = feedforwardnet(hiddenUnits);
            self.network.layers{1}.transferFcn = hiddenLayer;
            self.network.layers{2}.transferFcn = outputLayer; 
            self.network.performFcn = costFunction; 

            self.network.outputs{end}.processFcns = {}; 
        end

        function self = train(self, trainData, trainLabels, valData, valLabels)

            % Default parameters for MLP
            self.network.trainParam.epochs = 500; 
            self.network.trainParam.showWindow = true;  % Show training window

            % Train the network
            [self.network, ~] = train(self.network, trainData, trainLabels);

            % Validate the network
            valPredictions = self.network(valData);
            [~, valPredictedClasses] = max(valPredictions);
            [~, valActualClasses] = max(valLabels);

            self.valAccuracy = mean(valPredictedClasses == valActualClasses) * 100;
            self.valMmserror = mse(valPredictedClasses, valActualClasses);
            fprintf('Validation MSE: %.2f \n', self.valMmserror);
            fprintf('Validation Accuracy: %.2f%%\n', self.valAccuracy);
        end

        function self = test(self, testData, testLabels)
            predictions = self.network(testData);
            [~, predictedClasses] = max(predictions);
            [~, actualClasses] = max(testLabels);

            self.accuracy = mean(predictedClasses == actualClasses) * 100;
            self.mserror = mse(predictedClasses, actualClasses);
            fprintf('MSE: %.2f \n', self.mserror);
            fprintf('Test Accuracy: %.2f%%\n',self.accuracy);
        end
    end
end



% Using patternnet
%{
classdef MLP_A
    properties
        hiddenUnits
        hiddenLayer
        outputLayer
        costFunction

        network
    end
    methods
        function self = MLP(hiddenUnits, hiddenLayer, outputLayer, costFunction)
            self.hiddenUnits = hiddenUnits;
            self.hiddenLayer = hiddenLayer;
            self.outputLayer = outputLayer;
            self.costFunction = costFunction;

            self.network = patternnet(hiddenUnits);
            self.network.layers{1}.transferFcn = hiddenLayer;
            self.network.layers{2}.transferFcn = outputLayer; 
            self.network.performFcn = costFunction; 
        end

        function self = train(self, trainData, trainLabels, valData, valLabels)

            % Set training parameters
            self.network.trainParam.epochs = 2000;
            self.network.trainParam.lr = 0.01;
            self.network.trainParam.showWindow = true;  % Show training window

            % Train the network
            [self.network, ~] = train(self.network, trainData, trainLabels);

            % Validate the network
            valPredictions = self.network(valData);
            [~, valPredictedClasses] = max(valPredictions);
            [~, valActualClasses] = max(valLabels);
            valAccuracy = mean(valPredictedClasses == valActualClasses) * 100;

            fprintf('Validation Accuracy: %.2f%%\n', valAccuracy);
        end

        function accuracy = test(self, testData, testLabels)
            predictions = self.network(testData);
            [~, predictedClasses] = max(predictions);
            [~, actualClasses] = max(testLabels);

            accuracy = mean(predictedClasses == actualClasses) * 100;
            fprintf('Test Accuracy: %.2f%%\n', accuracy);
        end
    end
end
%}