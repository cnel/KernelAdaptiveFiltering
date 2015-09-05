function [weightVector,biasTerm,learningCurve]= ...
    LMS2(trainInput,trainTarget,regularizationFactor,stepSizeWeightVector,stepSizeBias)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function LMS2:
%Normalized least mean square algorithms
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Input:
%trainInput:    input signal inputDimension*trainSize, inputDimension is the input dimension and trainSize 
%               is the number of training data
%trainTarget:   desired signal for training trainSize*1
%
%
%regularizationFactor: regularization factor in Newton's recursion
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

learningCurve = zeros(trainSize,1);

weightVector = zeros(inputDimension,1);
biasTerm = 0;

% training
for n = 1:trainSize
    networkOutput = weightVector'*trainInput(:,n) + biasTerm;
    aprioriErr = trainTarget(n) - networkOutput;
    weightVector = weightVector + stepSizeWeightVector*aprioriErr*trainInput(:,n)/(sum(trainInput(:,n).^2) + regularizationFactor);
    biasTerm = biasTerm + stepSizeBias*aprioriErr;
    learningCurve(n) = aprioriErr^2;
end

return
