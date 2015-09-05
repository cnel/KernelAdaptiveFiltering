function [predictionError,expansionCoefficient,dictionary,weightVector,biasTerm,learningCurve] = ...
    sparseKLMS1(trainInput,trainTarget,feedbackDimension,typeKernel,paramKernel,...
    stepSizeFeatureVector,stepSizeWeightVector,stepSizeBias,toleranceDistance,tolerancePredictError)
%Function sparseKLMS1:   kernel least mean square with novel criteria
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Input:
%trainInput:        input signal inputDimension*trainSize, inputDimension is the input dimension and trainSize is the number of
%                   training data
%trainTarget:       desired signal for training trainSize*1
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
%dictionary:                bases used in the kernel expansion in
%                               the training set
%weightVector:              the linear coefficients
%biasTerm:                  the bias term
%learningCurve:             trainSize*1 used for learning curve

% memeory initialization
trainSize = length(trainTarget);

learningCurve = zeros(trainSize,1);
learningCurve(1) = trainTarget(1)^2; 

input = [trainInput(:,1);zeros(feedbackDimension,1)];
predictionError = size(trainSize,1);
% n=1 init
predictionError(1) = trainTarget(1);
expansionCoefficient = stepSizeFeatureVector*predictionError(1);
weightVector = stepSizeWeightVector*predictionError(1)*input;
biasTerm = stepSizeBias*predictionError(1);
networkOutput = zeros(trainSize,1);
toleranceDistance = toleranceDistance^2;

% dictionary
dictionary = input;
dictSize = 1;

% start
for n=2:trainSize
    % training
       
    % comparing the distance between trainInput(:,n) and the dictionary
    if feedbackDimension > n
        input = [trainInput(:,n);networkOutput(n-1:-1:1);zeros(feedbackDimension-n+1,1)];
    else
        input = [trainInput(:,n);networkOutput(n-1:-1:n-feedbackDimension)];
    end
    
    distance2dictionary = min(sum((input*ones(1,dictSize) - dictionary).^2));
    networkOutput(n) = expansionCoefficient*ker_eval(input,dictionary,typeKernel,paramKernel) + weightVector'*input + biasTerm;
    predictionError(n) = trainTarget(n) - networkOutput(n);
    
    if (distance2dictionary < toleranceDistance)
        learningCurve(n) = learningCurve(n-1);
        continue;
    end
    if (abs(predictionError(n)) < tolerancePredictError)
        learningCurve(n) = learningCurve(n-1);
        continue;
    end
    % updating
    dictSize = dictSize + 1;
    dictionary(:,dictSize) = input;
    expansionCoefficient(dictSize) = stepSizeFeatureVector*predictionError(n);
    
    weightVector = weightVector + stepSizeWeightVector*predictionError(n)*input;
    biasTerm = biasTerm + stepSizeBias*predictionError(n);
    
    learningCurve(n) = predictionError(n)^2;
    
end

return

