function [expansionCoefficient] = ...
    regularizationNetwork(trainInput,trainTarget,typeKernel,paramKernel,regularizationParameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function regularizationNetwork
%regularized RBF
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Input:
%trainInput:    input signal inputDimension*trainSize, inputDimension is the input dimension and 
%               trainSize is the number of training data
%trainTarget:   desired signal for training trainSize*1
%
%typeKernel:    'Gauss', 'Poly'
%paramKernel:   h (kernel size) for Gauss and p (order) for poly
%
%regularizationFactor: regularization parameter
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Output:
%baseDictionary:            dictionary stores all the bases centers
%expansionCoefficient:      coefficients of the kernel expansion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Notes: Since bases are by default all the training data, it is skipped
%       here.


% memeory initialization
[inputDimension,trainSize] = size(trainInput);
gramMatrix = zeros(trainSize,trainSize);

for i = 1:trainSize-1
    j = i+1:trainSize;
    gramMatrix(i,j) = ker_eval(trainInput(:,i),trainInput(:,j),typeKernel,paramKernel);
    gramMatrix(i,i) = ker_eval(trainInput(:,i),trainInput(:,i),typeKernel,paramKernel);
    gramMatrix(j,i) = gramMatrix(i,j)';
end
G_lam = gramMatrix + regularizationParameter*eye(trainSize);
expansionCoefficient = inv(G_lam)*trainTarget;
return