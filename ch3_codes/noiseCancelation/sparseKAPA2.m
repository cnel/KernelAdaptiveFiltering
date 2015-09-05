function [predictionError,expansionCoefficient,dictionary,weightVector,biasTerm,learningCurve] = ...
    sparseKAPA2(K,trainInput,trainTarget,feedbackDimension,paramRegularization,typeKernel,paramKernel,stepSizeFeatureVector,stepSizeWeightVector,stepSizeBias,toleranceDistance,tolerancePredictError)
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
weightVector = zeros(inputDimension+feedbackDimension,1);
biasTerm = 0;

input = [trainInput(:,1);zeros(feedbackDimension,1)];
% keep the first K input in the dictionary blindly
expansionCoefficient = stepSizeFeatureVector*trainTarget(1);
% dictionary
predictionError = zeros(trainSize,1);
dictionary = input;
dictionaryIndex = 1;
dictSize = 1;

learningCurve = zeros(trainSize,1);
learningCurve(1) = trainTarget(1)^2;
toleranceDistance = toleranceDistance^2;

% start training
for n = 2:trainSize
    % filtering
    % there is problem wiht feedbackDimension>1 here%%%%%%%%%%%%%%%%%%%%%%
    input = [trainInput(:,n);networkOutput(K:-1:K-feedbackDimension+1)];
 
    
    % comparing the distance between trainInput(:,n) and the dictionary
    distance2dictionary = min(sum((input*ones(1,dictSize) - dictionary).^2));
    networkOutput(K) = expansionCoefficient*...
        ker_eval(input,dictionary,typeKernel,paramKernel) + weightVector'*input + biasTerm;
    predictionError(n) = trainTarget(n) - networkOutput(K);

    if (distance2dictionary < toleranceDistance)
        learningCurve(n) = learningCurve(n-1);
        continue;
    end
    if (abs(predictionError(n)) < tolerancePredictError)
        learningCurve(n) = predictionError(n)^2;        
        continue;
    end
    % updating
    weightVector = weightVector + stepSizeWeightVector*input*predictionError(n);
    biasTerm = biasTerm + stepSizeBias*predictionError(n);
    % updating
    dictSize = dictSize + 1;
    dictionary(:,dictSize) = input;
    dictionaryIndex(dictSize) = n;
    
    if dictSize < K
        expansionCoefficient(dictSize) = stepSizeFeatureVector*predictionError(n);
    else
        for kk = 1:K-1
            networkOutput(K-kk) = expansionCoefficient*...
                ker_eval(dictionary(:,dictSize-kk),dictionary(:,1:end-1),typeKernel,paramKernel)...
                + weightVector'*dictionary(:,dictSize-kk) + biasTerm;
        end
        aprioriErr = trainTarget(dictionaryIndex(end-K+1:end)) - networkOutput;
        expansionCoefficient(dictSize) = 0;
        expansionCoefficient(dictSize-K+1:dictSize) = expansionCoefficient(dictSize-K+1:dictSize) + ...
            stepSizeFeatureVector*aprioriErr'*inv(paramRegularization*eye(K) + gramMatrix(dictionary(:,end-K+1:end),typeKernel,paramKernel));
    end
    
    learningCurve(n) = predictionError(n)^2;

end

return

