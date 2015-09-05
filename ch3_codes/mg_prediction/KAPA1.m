function [expansionCoefficient,weightVector,biasTerm,learningCurve] = ...
    KAPA1(K,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeFeatureVector,stepSizeWeightVector,stepSizeBias,flagLearningCurve)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function KAPA
%Kernel affine projection algorithm
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
%paramRegularization: regularization parameter in the cost function
%
%stepSizeFeatureVector:     learning rate for kernel part
%stepSizeWeightVector:      learning rate for linear part, set to zero to disable
%stepSizeBias:              learning rate for bias term, set to zero to disable
%
%flagLearningCurve:    control if calculating the learning curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Output:
%baseDictionary:            dictionary stores all the bases centers
%expansionCoefficient:      coefficients of the kernel expansion
%weightVector:      the linear weight vector
%biasTerm:          the bias term
%learningCurve:     trainSize*1 used for learning curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Notes: Since bases are by default all the training data, it is skipped
%       here.


% memeory initialization
[inputDimension,trainSize] = size(trainInput);
testSize = length(testTarget);

expansionCoefficient = zeros(trainSize,1);
networkOutput = zeros(K,1);
weightVector = zeros(inputDimension,1);
biasTerm = 0;

if flagLearningCurve
    learningCurve = zeros(trainSize,1);
    learningCurve(1) = mean(testTarget.^2);
else
    learningCurve = [];
end

expansionCoefficient(1) = stepSizeFeatureVector*trainTarget(1);
% start training
for n = 2:K
    ii = 1:n-1;
    networkOutput(K) = expansionCoefficient(ii)'*...
        ker_eval(trainInput(:,n),trainInput(:,ii),typeKernel,paramKernel) + weightVector'*trainInput(:,n) + biasTerm;
    predictionError = trainTarget(n) - networkOutput(K);
    
    % updating
    weightVector = weightVector + stepSizeWeightVector*trainInput(:,n)*predictionError;
    biasTerm = biasTerm + stepSizeBias*predictionError;
    % updating
    expansionCoefficient(n) = stepSizeFeatureVector*predictionError;
    if flagLearningCurve
        % testing
        y_te = zeros(testSize,1);
        for jj = 1:testSize
            ii = 1:n;
            y_te(jj) = expansionCoefficient(ii)'*...
                ker_eval(testInput(:,jj),trainInput(:,ii),typeKernel,paramKernel) + weightVector'*testInput(:,jj) + biasTerm;
        end
        err = testTarget - y_te;
        learningCurve(n) = mean(err.^2);
    end
end

for n = K+1:trainSize
    % filtering
    ii = 1:n-1;
    for kk = 1:K
        networkOutput(kk) = expansionCoefficient(ii)'*...
            ker_eval(trainInput(:,n+kk-K),trainInput(:,ii),typeKernel,paramKernel) + weightVector'*trainInput(:,n+kk-K) + biasTerm;
    end
    aprioriErr = trainTarget(n-K+1:n) - networkOutput; 
    
    % updating
    weightVector = weightVector + stepSizeWeightVector*trainInput(:,n-K+1:n)*aprioriErr;
    biasTerm = biasTerm + stepSizeBias*sum(aprioriErr);

    expansionCoefficient(n-K+1:n) = expansionCoefficient(n-K+1:n) + stepSizeFeatureVector*aprioriErr;
     
    if flagLearningCurve
        % testing
        y_te = zeros(testSize,1);
        for jj = 1:testSize
            ii = 1:n;
            y_te(jj) = expansionCoefficient(ii)'*...
                ker_eval(testInput(:,jj),trainInput(:,ii),typeKernel,paramKernel) + weightVector'*testInput(:,jj) + biasTerm;
        end
        err = testTarget - y_te;
        learningCurve(n) = mean(err.^2);
    end
end

return

