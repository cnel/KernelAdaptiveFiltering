function [expansionCoefficient,weightVector,biasTerm,learningCurve,dictionaryIndex,netSizeDiagram] = ...
    sparseKLMS1s(trainInput,trainTarget,typeKernel,paramKernel,...
    stepSizeFeatureVector,stepSizeWeightVector,stepSizeBias,toleranceDistance,tolerancePredictError,flagLearningCurve)
%Function sparseKLMS1s:   kernel least mean square with novel criteria
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Input:
%trainInput:        input signal inputDimension*trainSize, inputDimension is the input dimension and trainSize is the number of
%                   training data
%trainTarget:       desired signal for training trainSize*1
%testInput:         testing input, inputDimension*testSize, testSize is the number of the test data
%testTarget:        desired signal for testing testSize*1
%typeKernel:        'Gauss', 'Poly'
%paramKernel:       h (kernel size) for Gauss and p (order) for poly
%stepSizeFeatureVector:     learning rate for kernel part
%stepSizeWeightVector:      learning rate for linear part, set to zero to disable
%stepSizeBias:              learning rate for bias term, set to zero to disable
%flagLearningCurve:         control if calculating the learning curve
%toleranceDistance:         tolerance for the closeness of the new data to the dictionary
%tolerancePredictError:     tolerance for the apriori error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Output:
%expansionCoefficient:      onsisting of coefficients of the kernel
%                           expansion with the stepSizeFeatureVector
%weightVector:              the linear coefficients
%biasTerm:                  the bias term
%learningCurve:             trainSize*1 used for learning curve
%dictionaryIndex:           index of bases used in the kernel expansion in
%                               the training set
%netSizeDiagram:            network size over iteration


% memeory initialization
trainSize = length(trainTarget);

if flagLearningCurve
    learningCurve = zeros(trainSize,1);
    learningCurve(1) = mean(trainTarget.^2); 
else
    learningCurve =[];
end
netSizeDiagram = zeros(trainSize,1);

% n=1 init
predictionError = trainTarget(1);
expansionCoefficient = stepSizeFeatureVector*predictionError;
weightVector = stepSizeWeightVector*predictionError*trainInput(:,1);
biasTerm = stepSizeBias*predictionError;

% dictionary
dictionaryIndex = 1;
dictSize = 1;
netSizeDiagram(1) = 1;
% start
for n=2:trainSize
    % training
       
    % comparing the distance between trainInput(:,n) and the dictionary
    distance2dictionary = min(sum((trainInput(:,n)*ones(1,dictSize) - trainInput(:,dictionaryIndex)).^2));
    if (distance2dictionary < toleranceDistance)
        if flagLearningCurve, learningCurve(n) = learningCurve(n-1); end
        netSizeDiagram(n) = netSizeDiagram(n-1);
        continue;
    end
    networkOutput = expansionCoefficient*ker_eval(trainInput(:,n),trainInput(:,dictionaryIndex),typeKernel,paramKernel) + weightVector'*trainInput(:,n) + biasTerm;
    predictionError = trainTarget(n) - networkOutput;
    if (abs(predictionError) < tolerancePredictError)
        netSizeDiagram(n) = netSizeDiagram(n-1);
        continue;
        
    end
    % updating
    dictSize = dictSize + 1;
    dictionaryIndex(dictSize) = n;
    expansionCoefficient(dictSize) = stepSizeFeatureVector*predictionError;
    netSizeDiagram(n) = netSizeDiagram(n-1) + 1;
    
    weightVector = weightVector + stepSizeWeightVector*predictionError*trainInput(:,n);
    biasTerm = biasTerm + stepSizeBias*predictionError;
    
    if flagLearningCurve==1, learningCurve(n) = predictionError^2; end

end

return

