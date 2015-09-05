function [expansionCoefficient,dictionaryIndex,weightVector,biasTerm,learningCurve,netSizeDiagram] = ...
    sparseKAPA2s(K,trainInput,trainTarget,paramRegularization,typeKernel,paramKernel,stepSizeFeatureVector,stepSizeWeightVector,stepSizeBias,toleranceDistance,tolerancePredictError,flagLearningCurve)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function KAPA
%Kernel affine projection algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Input:
%trainInput:    input signal inputDimension*trainSize, inputDimension is the input dimension and 
%               trainSize is the number of training data
%trainTarget:   desired signal for training trainSize*1
%
%typeKernel:    'Gauss', 'Poly'
%paramKernel:   h (kernel size) for Gauss and p (order) for poly
%
%paramRegularization: regularization parameter in the cost function
%
%stepSizeFeatureVector:     learning rate for kernel part
%stepSizeWeightVector:      learning rate for linear part, set to zero to disable
%stepSizeBias:              learning rate for bias term, set to zero to disable
%
%toleranceDistance:         tolerance for the closeness of the new data to the dictionary
%tolerancePredictError:     tolerance for the apriori error
%
%flagLearningCurve:    control if calculating the learning curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Output:
%expansionCoefficient:       coefficients of the kernel expansion
%dictionaryIndex:            dictionary stores all the bases centers
%weightVector:      the linear weight vector
%biasTerm:          the bias term
%learningCurve:     trainSize*1 used for learning curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Notes: Since bases are by default all the training data, it is skipped
%       here.


% memeory initialization
[inputDimension,trainSize] = size(trainInput);

networkOutput = zeros(K,1);
weightVector = zeros(inputDimension,1);
biasTerm = 0;

% keep the first K input in the dictionary blindly
expansionCoefficient = stepSizeFeatureVector*trainTarget(1);
% dictionary
dictionaryIndex = 1;
dictSize = 1;

if flagLearningCurve
    learningCurve = zeros(trainSize,1);
    learningCurve(1) = mean(trainTarget.^2);
else
    learningCurve = [];
end

netSizeDiagram = zeros(trainSize,1);
netSizeDiagram(1) = 1;

% start training
for n = 2:trainSize
    % filtering
    
    % comparing the distance between trainInput(:,n) and the dictionary
    distance2dictionary = min(sum((trainInput(:,n)*ones(1,dictSize) - trainInput(:,dictionaryIndex)).^2));
    if (distance2dictionary < toleranceDistance)
        if flagLearningCurve, learningCurve(n) = learningCurve(n-1); end
        netSizeDiagram(n) = netSizeDiagram(n-1);
        continue;
    end
    networkOutput(K) = expansionCoefficient*...
        ker_eval(trainInput(:,n),trainInput(:,dictionaryIndex),typeKernel,paramKernel) + weightVector'*trainInput(:,n) + biasTerm;
    predictionError = trainTarget(n) - networkOutput(K);
    if (abs(predictionError) < tolerancePredictError)
        netSizeDiagram(n) = netSizeDiagram(n-1);
        continue;
    end
    % updating
    weightVector = weightVector + stepSizeWeightVector*trainInput(:,n)*predictionError;
    biasTerm = biasTerm + stepSizeBias*predictionError;
    % updating
    dictSize = dictSize + 1;
    dictionaryIndex(dictSize) = n;
    netSizeDiagram(n) = netSizeDiagram(n-1) + 1;

    if dictSize < K
        expansionCoefficient(dictSize) = stepSizeFeatureVector*predictionError;
        
    else
        for kk = 1:K-1
            networkOutput(K-kk) = expansionCoefficient*...
                ker_eval(trainInput(:,dictionaryIndex(dictSize-kk)),trainInput(:,dictionaryIndex(1:end-1)),typeKernel,paramKernel)...
                + weightVector'*trainInput(:,dictionaryIndex(dictSize-kk)) + biasTerm;
        end
        aprioriErr = trainTarget(dictionaryIndex(end-K+1:end)) - networkOutput;
        expansionCoefficient(dictSize) = 0;
        expansionCoefficient(dictSize-K+1:dictSize) = expansionCoefficient(dictSize-K+1:dictSize) + ...
            stepSizeFeatureVector*aprioriErr'*inv(paramRegularization*eye(K) + gramMatrix(trainInput(:,dictionaryIndex(end-K+1:end)),typeKernel,paramKernel));
    end
    
    if flagLearningCurve, learningCurve(n) = predictionError^2; end

end

return

