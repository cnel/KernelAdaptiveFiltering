function [expansionCoefficient,weightVector,biasTerm,learningCurve] = ...
    KLMS1_LC(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeFeatureVector,stepSizeWeightVector,stepSizeBias,flagLearningCurve)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Input:
%trainInput:    input signal inputDimension*trainSize, inputDimension is the input dimension and 
%               trainSize is the number of training data
%trainTarget:   desired signal for training trainSize*1
%
%testInput:     testing input, inputDimension*testSize, testSize is the number of the test data
%testTarget:    desired signal for testing testSize*1
%
%typeKernel:    'Gauss', 'Poly'
%paramKernel:   h (kernel size) for Gauss and p (order) for poly
%
%stepSizeFeatureVector:     learning rate for kernel part
%stepSizeWeightVector:      learning rate for linear part, set to zero to disable
%stepSizeBias:              learning rate for bias term, set to zero to disable
%
%flagLearningCurve:    control if calculating the learning curve
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
testSize = length(testTarget);

expansionCoefficient = zeros(trainSize,1);

if flagLearningCurve
    learningCurve = zeros(trainSize,1);
    learningCurve(1) = mean(testTarget.^2);
else
    learningCurve = [];
end

% n=1 init
aprioriErr = trainTarget(1);
weightVector = stepSizeWeightVector*aprioriErr*trainInput(:,1);
biasTerm = stepSizeBias*aprioriErr;
expansionCoefficient(1) = stepSizeFeatureVector*aprioriErr;
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
    
    if flagLearningCurve
        % testing
        y_te = zeros(testSize,1);
        for jj = 1:testSize
            ii = 1:n;
            y_te(jj) = expansionCoefficient(ii)'*ker_eval(testInput(:,jj),trainInput(:,ii),typeKernel,paramKernel) + weightVector'*testInput(:,jj) + biasTerm;
        end
        err = testTarget - y_te;
        learningCurve(n) = mean(err.^2);
    end
end

return

