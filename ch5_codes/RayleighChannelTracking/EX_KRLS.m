function [expansionCoefficient,learningCurve] = ...
    EX_KRLS(trainInput,trainTarget,typeKernel,paramKernel,alphaParameter,regularizationFactor,forgettingFactor,qFactor)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function KRLS
%extended Kernel recursive least squares
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
%alphaParameter: alpha parameter in the state-space model
%regularizationFactor: regularization parameter in Newton's recursion
%
%forgettingFactor: expoentially weighted value
%
%flagLearningCurve:    control if calculating the learning curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Output:
%baseDictionary:            dictionary stores all the bases centers
%expansionCoefficient:      coefficients of the kernel expansion
%learningCurve:     trainSize*1 used for learning curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Notes: Since bases are by default all the training data, it is skipped
%       here.


% memeory initialization
[inputDimension,trainSize] = size(trainInput);

expansionCoefficient = zeros(trainSize,1);

learningCurve = zeros(trainSize,1);

Q_matrix = 0;
roe = 1/(forgettingFactor*regularizationFactor);

% start training
for n = 1:trainSize
    ii = 1:n-1;
    k_vector = ker_eval(trainInput(:,n),trainInput(:,ii),typeKernel,paramKernel);
    f_vector = Q_matrix*k_vector;
    r_e = (forgettingFactor^(n)+ roe*ker_eval(trainInput(:,n),trainInput(:,n),typeKernel,paramKernel) - k_vector'*f_vector);
    s = 1/r_e;
    
    error = trainTarget(n) - k_vector'*expansionCoefficient(ii);    
    % updating
    expansionCoefficient(n) = alphaParameter*roe*s*error;
    expansionCoefficient(ii) = alphaParameter*(expansionCoefficient(ii) - f_vector*s*error);
    
    Q_tmp = zeros(n,n);
    Q_tmp(ii,ii) = Q_matrix + f_vector*f_vector'*s;
    Q_tmp(ii,n) = -roe*f_vector*s;
    Q_tmp(n,ii) = Q_tmp(ii,n)';
    Q_tmp(n,n) = roe^2*s;
    Q_matrix = alphaParameter^2*Q_tmp;
    
    roe = alphaParameter^2*roe + forgettingFactor^(n)*qFactor;
    
    learningCurve(n) = error^2;

end

return