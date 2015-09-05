function [expansionCoefficient,dictionaryIndex,learningCurve,ALD] = ...
    EX_KRLS_ALD(trainInput,trainTarget,typeKernel,paramKernel,alphaParameter,regularizationFactor,forgettingFactor,qFactor,threshold)
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
learningCurve = zeros(trainSize,1);
ALD = zeros(trainSize,1);

% n = 1
Q_matrix = alphaParameter^2/((forgettingFactor^2*regularizationFactor+ker_eval(trainInput(:,1),trainInput(:,1),typeKernel,paramKernel))*(alphaParameter^2+forgettingFactor^2*regularizationFactor*qFactor));
roe_r = regularizationFactor*forgettingFactor/(alphaParameter^2 + forgettingFactor*qFactor);

expansionCoefficient = alphaParameter*trainTarget(1)/(regularizationFactor*forgettingFactor^2+ker_eval(trainInput(:,1),trainInput(:,1),typeKernel,paramKernel));

dictionaryIndex = 1;
dictSize = 1;

ALD(1) = 0;
learningCurve(1) = trainTarget(1)^1;

% start training
for n = 2:trainSize

    k_vector = ker_eval(trainInput(:,n),trainInput(:,dictionaryIndex),typeKernel,paramKernel);
    f_vector = Q_matrix*k_vector;
    r_e = (forgettingFactor^(dictSize)*roe_r+ ker_eval(trainInput(:,n),trainInput(:,n),typeKernel,paramKernel) - k_vector'*f_vector);
    
    ALD(n) = r_e;
    
    if (r_e >threshold)

        s = 1/r_e;
        error = trainTarget(n) - k_vector'*expansionCoefficient';    
        
        % updating coefficients
        expansionCoefficient = alphaParameter*(expansionCoefficient - f_vector'*s*error);
        dictSize = dictSize + 1;
        expansionCoefficient(dictSize) = alphaParameter*s*error;
        dictionaryIndex(dictSize) = n;

        %update Q_matrix and roe
        Q_tmp = zeros(dictSize,dictSize);
        Q_tmp(1:dictSize-1,1:dictSize-1) = Q_matrix + f_vector*f_vector'*s;
        Q_tmp(1:dictSize-1,dictSize) = -f_vector*s;
        Q_tmp(dictSize,1:dictSize-1) = Q_tmp(1:dictSize-1,dictSize)';
        Q_tmp(dictSize,dictSize) = s;
       
        ratio = 1/(1+forgettingFactor^(dictSize)*qFactor*alphaParameter^(-2)*roe_r);
        Q_matrix = ratio*Q_tmp;
        
        roe_r = roe_r/(alphaParameter^2 + forgettingFactor^(dictSize)*qFactor*roe_r);
   
        learningCurve(n) = error^2;
    else
        learningCurve(n) = learningCurve(n-1);
    end

end

return