function [aprioriErr,weightVector,biasTerm,learningCurve]= ...
    LMS2(trainInput,trainTarget,feedbackDimension,regularizationFactor,stepSizeWeightVector,stepSizeBias)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function LMS2:
%Normalized least mean square algorithms
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Input:
%trainInput:    input signal inputDimension*trainSize
%               BUT you have to add the feedback dimension onto it to form
%               the real regressors!!
%               data format according to the problem, say the input may
%               accept feedback from the output in adaptive noise
%               cancellation!!!
%feedbackDimension: the number of output feedback as the input
%trainTarget:   desired signal for training trainSize*1
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
trainInput(:,trainSize+1) = zeros(inputDimension,1);

learningCurve = zeros(trainSize,1);

weightVector = zeros(inputDimension+feedbackDimension,1);
biasTerm = 0;

input = [trainInput(:,1);zeros(feedbackDimension,1)];
networkOutput = zeros(trainSize,1);
aprioriErr = zeros(trainSize,1);
% training
for n = 1:trainSize
    networkOutput(n) = weightVector'*input + biasTerm;
    aprioriErr(n) = trainTarget(n) - networkOutput(n);
    weightVector = weightVector + stepSizeWeightVector*aprioriErr(n)*input/(sum(input.^2) + regularizationFactor);
    biasTerm = biasTerm + stepSizeBias*aprioriErr(n);
    learningCurve(n) = aprioriErr(n)^2;
    if feedbackDimension > n
        input = [trainInput(:,n+1);networkOutput(n:-1:1);zeros(feedbackDimension-n,1)];
    else
        input = [trainInput(:,n+1);networkOutput(n:-1:n-feedbackDimension+1)];
    end
end

return
