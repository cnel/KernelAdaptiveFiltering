function [weightVector,biasTerm,learningCurve]= ...
    LMS1(trainInput,trainTarget,stepSizeWeightVector,stepSizeBias)
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
%stepSizeWeightVector:  learning rate for weight part, set to zero to disable
%stepSizeBias:          learning rate for bias term, set to zero to disable
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

learningCurve = zeros(trainSize,1);

weightVector = zeros(inputDimension,1);
biasTerm = 0;

% training
for n = 1:trainSize
    networkOutput = weightVector'*trainInput(:,n) + biasTerm;
    aprioriErr = trainTarget(n) - networkOutput;
    weightVector = weightVector + stepSizeWeightVector*aprioriErr*trainInput(:,n);
    biasTerm = biasTerm + stepSizeBias*aprioriErr;
    learningCurve(n) = aprioriErr^2;
end

return
