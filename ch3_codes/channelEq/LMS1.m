function [aprioriErr,weightVector,biasTerm,learningCurve]= ...
    LMS1(trainInput,trainTarget,testInput,testTarget,stepSizeWeightVector,stepSizeBias,flagLearningCurve)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function LMS1:
%Normal least mean square algorithms
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Input:
%trainInput:    input signal inputDimension*trainSize, inputDimension is the input dimension and trainSize 
%               is the number of training data
%trainTarget:   desired signal for training trainSize*1
%
%testInput:     testing input for calculating the learning curve, 
%               inputDimension*testSize, testSize is the number of test data
%testTarget:    desired signal for testing testSize*1
%
%stepSizeWeightVector:  learning rate for weight part, set to zero to disable
%stepSizeBias:          learning rate for bias term, set to zero to disable
%
%flagLearningCurve: A FLAG to indicate if learning curve is needed
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Output:
%aprioriErr:        apriori error 
%weightVector:      the linear coefficients
%biasTerm:          the bias term
%learningCurve:     trainSize*1 used for learning curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Notes: none.

% memeory initialization
[inputDimension,trainSize] = size(trainInput);

if flagLearningCurve
    learningCurve = zeros(trainSize,1);
else
    learningCurve = [];
end

weightVector = zeros(inputDimension,1);
biasTerm = 0;
aprioriErr = zeros(trainSize,1);

% training
for n = 1:trainSize
    networkOutput = weightVector'*trainInput(:,n) + biasTerm;
    aprioriErr(n) = trainTarget(n) - networkOutput;
    weightVector = weightVector + stepSizeWeightVector*aprioriErr(n)*trainInput(:,n);
    biasTerm = biasTerm + stepSizeBias*aprioriErr(n);
    if flagLearningCurve
        % testing
        err = testTarget -(testInput'*weightVector + biasTerm);
        learningCurve(n) = mean(err.^2);
    end
end

return
