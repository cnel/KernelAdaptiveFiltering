function [weightVector,learningCurve]= ...
    EX_RLS(trainInput,trainTarget,pInitial,forgettingFactor,alphaParameter,qFactor)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function EX_RLS:
%extended recursive least squares
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Input:
%trainInput:    input signal inputDimension*trainSize, inputDimension is the input dimension and trainSize 
%               is the number of training data
%trainTarget:   desired signal for training trainSize*1
%
%
%pInitial:  the initial value for the P matrix, for example diagonal matrix
%
%forgettingFactor: the exponentially weighted value, very close to 1
%
%alphaParameter: alpha parameter is a scalar state transition factor,
%                   usually close to 1
%
%qFactor: modeling the variation in the state, providing a tradeoff between
%           the measurement noise ans modelind variantion, usually small
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
P_matrix = pInitial;

% training
for n = 1:trainSize
    u = trainInput(:,n);
    iGamma = 1+iForgettingFactor*u'*P_matrix*u;
    gamma = 1/iGamma;
    gain = alphaParameter*iForgettingFactor*P_matrix*u*gamma;
    error = trainTarget(n) - u'*weightVector;
    weightVector = alphaParameter*weightVector + gain*error;
    P_matrix = iForgettingFactor*alphaParameter^2*P_matrix - gain*gain'*iGamma + qFactor;
    learningCurve(n) = error^2;
    
end

return
