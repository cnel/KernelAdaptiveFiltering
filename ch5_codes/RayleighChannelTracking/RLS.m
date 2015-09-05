function [weightVector,learningCurve]= ...
    RLS(trainInput,trainTarget,pInitial,forgettingFactor)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function RLS:
%recursive least square exponentially weighted
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Input:
%trainInput:    input signal inputDimension*trainSize, inputDimension is the input dimension and trainSize 
%               is the number of training data
%trainTarget:   desired signal for training trainSize*1
%
%pInitial:  the initial value for the P matrix, for example diagonal matrix
%
%forgettingFactor: the exponentially weighted value, very close to 1
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

iForgettingFactor = 1/forgettingFactor;
P = pInitial;

% training
for n = 1:trainSize
    u = trainInput(:,n);
    iGamma = 1+iForgettingFactor*u'*P*u;
    gamma = 1/iGamma;
    gain = iForgettingFactor*P*u*gamma;
    error = trainTarget(n) - u'*weightVector;
    weightVector = weightVector + gain*error;
    P = iForgettingFactor*P - gain*gain'*iGamma;

    learningCurve(n) = error^2;
    
end

return
