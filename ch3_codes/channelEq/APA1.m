function [weightVector,biasTerm,learningCurve]= ...
    APA1(K,trainInput,trainTarget,testInput,testTarget,stepSizeWeightVector,stepSizeBias,flagLearningCurve)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function LMS1:
%Normal least mean square algorithms
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Input:
%K:             the K most recent observations are used to estimate the
%               gradient and hessian
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
%weightVector:      the linear coefficients
%biasTerm:          the bias term
%learningCurve:     trainSize*1 used for learning curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Notes: none.

% memeory initialization
[inputDimension,trainSize] = size(trainInput);

if flagLearningCurve
    learningCurve = zeros(trainSize,1);
    learningCurve(1:K) = mean(testTarget.^2)*ones(K,1);
else
    learningCurve = [];
end

weightVector = zeros(inputDimension,1);
biasTerm = 0;

% training
for n = 1:trainSize-K+1
    networkOutput = trainInput(:,n:n+K-1)'*weightVector + biasTerm;
    aprioriErr = trainTarget(n:n+K-1) - networkOutput;
    weightVector = weightVector + stepSizeWeightVector*trainInput(:,n:n+K-1)*aprioriErr;
    biasTerm = biasTerm + stepSizeBias*sum(aprioriErr);
    if flagLearningCurve
        % testing
        err = testTarget -(testInput'*weightVector + biasTerm);
        learningCurve(n+K-1) = mean(err.^2);
    end
end

return
