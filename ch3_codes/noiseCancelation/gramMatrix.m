function G = gramMatrix(data,typeKernel,paramKernel)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function gramMatrix
%Calculate the gram matrix of data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%inputs:
%data:  inputDimension*dataSize, the output matrix will be
%       dataSize-by-dataSize
%typeKernel:    'Gauss','Poly'
%paramKernel:   parameter used in kernel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%outputs:
%G:     GramMatrix of data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Notes: none.

[inputDimension,dataSize] = size(data);
G = zeros(dataSize,dataSize);

for ii = 1:dataSize
    jj = ii:dataSize;
    G(ii,jj) = ker_eval(data(:,ii),data(:,jj),typeKernel,paramKernel);
    G(jj,ii) = G(ii,jj);
end
return