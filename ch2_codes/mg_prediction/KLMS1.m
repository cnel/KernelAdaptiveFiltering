function [expansionCoefficient,weightVector,biasTerm,learningCurve] = ...
    KLMS1(trainInput,trainTarget,typeKernel,paramKernel,stepSizeFeatureVector,stepSizeWeightVector,stepSizeBias)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Input:
%trainInput:    input signal inputDimension*trainSize, inputDimension is the input dimension and 
%               trainSize is the number of training data
%trainTarget:   desired signal for training trainSize*1
%
%typeKernel:    'Gauss', 'Poly'
%paramKernel:   h (kernel size) for Gauss and p (order) for poly
%
%stepSizeFeatureVector:     learning rate for kernel part
%stepSizeWeightVector:      learning rate for linear part, set to zero to disable
%stepSizeBias:              learning rate for bias term, set to zero to disable
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Output:
%expansionCoefficient:        consisting of coefficients of the kernel expansion
%weightVector:      the linear weight vector
%biasTerm:          the bias term
%learningCurve:     trainSize*1 used for learning curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Notes: none.


% memeory initialization
trainSize = length(trainTarget);
expansionCoefficient = zeros(trainSize,1);
learningCurve = zeros(trainSize,1);

% n=1 init
aprioriErr = trainTarget(1);
weightVector = stepSizeWeightVector*aprioriErr*trainInput(:,1);
biasTerm = stepSizeBias*aprioriErr;
expansionCoefficient(1) = stepSizeFeatureVector*aprioriErr;
learningCurve(1) = aprioriErr^2;

% start
for n = 2:trainSize
    % training
    % filtering
    ii = 1:n-1;
    networkOutput = expansionCoefficient(ii)'*ker_eval(trainInput(:,n),trainInput(:,ii),typeKernel,paramKernel) + weightVector'*trainInput(:,n) + biasTerm;
    aprioriErr = trainTarget(n) - networkOutput; 
    % updating
    weightVector = weightVector + stepSizeWeightVector*aprioriErr*trainInput(:,n);
    biasTerm = biasTerm + stepSizeBias*aprioriErr;
    expansionCoefficient(n) = stepSizeFeatureVector*aprioriErr;
    
   learningCurve(n) = aprioriErr^2;
end

return

