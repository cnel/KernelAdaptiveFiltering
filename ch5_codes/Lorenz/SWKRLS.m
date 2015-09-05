function [expansionCoefficient,learningCurve] = ...
    SWKRLS(K,trainInput,trainTarget,typeKernel,paramKernel,paramRegularization)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function sliding window kernel recursive least squares
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Input:
%trainInput:    input signal inputDimension*trainSize, inputDimension is the input dimension and 
%               trainSize is the number of training data
%trainTarget:   desired signal for training trainSize*1
%
%typeKernel:    'Gauss', 'Poly'
%paramKernel:   h (kernel size) for Gauss and p (order) for poly
%
%paramRegularization: regularization parameter in cost function
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Output:
%expansionCoefficient:      coefficients of the kernel expansion
%learningCurve:     trainSize*1 used for learning curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Notes: Since bases are by default all the training data, it is skipped
%       here.

% memeory initialization
[inputDimension,trainSize] = size(trainInput);

expansionCoefficient = zeros(K,1);

learningCurve = zeros(trainSize,1);
learningCurve(1) = trainTarget(1)^2;

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
    
    learningCurve(n) = error^2;
end

% start training
for n = K+1:trainSize
    
    k_vector = ker_eval(trainInput(:,n),trainInput(:,n-K:n-1),typeKernel,paramKernel);
    error = trainTarget(n) - k_vector'*expansionCoefficient;
    
    % updating
    expansionCoefficient = inv(paramRegularization*eye(K) + gramMatrix(trainInput(:,n-K+1:n),typeKernel,paramKernel))*trainTarget(n-K+1:n);
  
    learningCurve(n) = error^2;
end

return

