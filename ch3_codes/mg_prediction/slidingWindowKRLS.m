function [expansionCoefficient,learningCurve] = ...
    slidingWindowKRLS(K,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,paramRegularization,flagLearningCurve)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function sliding window kernel recursive least squares
%
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
%paramRegularization: regularization parameter in cost function
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

expansionCoefficient = zeros(K,1);

if flagLearningCurve
    learningCurve = zeros(trainSize,1);
    learningCurve(1) = mean(testTarget.^2);
else
    learningCurve = [];
end

Q_matrix = 1/(paramRegularization + ker_eval(trainInput(:,1),trainInput(:,1),typeKernel,paramKernel));
expansionCoefficient(1) = Q_matrix*trainTarget(1);
% start training
for n = 2:K
    ii = 1:n-1;
    k_vector = ker_eval(trainInput(:,n),trainInput(:,ii),typeKernel,paramKernel);
    f_vector = Q_matrix*k_vector;
    s = 1/(paramRegularization+ ker_eval(trainInput(:,n),trainInput(:,n),typeKernel,paramKernel) - k_vector'*f_vector);
    Q_tmp = zeros(n,n);
    Q_tmp(ii,ii) = Q_matrix + f_vector*f_vector'*s;
    Q_tmp(ii,n) = -f_vector*s;
    Q_tmp(n,ii) = Q_tmp(ii,n)';
    Q_tmp(n,n) = s;
    Q_matrix = Q_tmp;
    
    error = trainTarget(n) - k_vector'*expansionCoefficient(ii);
    
    % updating
    expansionCoefficient(n) = s*error;
    expansionCoefficient(ii) = expansionCoefficient(ii) - f_vector*expansionCoefficient(n);
    
    if flagLearningCurve
        % testing
        y_te = zeros(testSize,1);
        for jj = 1:testSize
            ii = 1:n;
            y_te(jj) = expansionCoefficient(ii)'*...
                ker_eval(testInput(:,jj),trainInput(:,ii),typeKernel,paramKernel);
        end
        err = testTarget - y_te;
        learningCurve(n) = mean(err.^2);
    end
end

% start training
for n = K+1:trainSize
     % updating
    
    expansionCoefficient = inv(paramRegularization*eye(K) + gramMatrix(trainInput(:,n-K+1:n),typeKernel,paramKernel))*trainTarget(n-K+1:n);
     
    if flagLearningCurve
        % testing
        y_te = zeros(testSize,1);
        for jj = 1:testSize
            ii = n-K+1:n;
            y_te(jj) = expansionCoefficient'*...
                ker_eval(testInput(:,jj),trainInput(:,ii),typeKernel,paramKernel);
        end
        err = testTarget - y_te;
        learningCurve(n) = mean(err.^2);
    end
end

return

