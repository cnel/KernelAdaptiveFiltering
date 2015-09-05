%Copyright
%Weifeng Liu CNEL
%July 4 2008
%
%Description:
%Prediction result
%
%Usage:
%Ch5, CO2 concentration forecasting, figure 5-10
%
%ouside functions called
%ker_eval

close all, clear all
clc
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Data Formatting
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Input signal
load CO2_data.mat date ave_missing ave_interpolate

dataSize = length(date);

% Data size for training and testing
trainSize = ceil(dataSize*0.7);
testSize = dataSize - trainSize;

%Input training signal with data embedding
trainInput = date(1:trainSize)';

%Input test data with embedding
testInput = date(1+trainSize:end)';

%Desired training signal
trainTarget = ave_interpolate(1:trainSize);

testTarget =  ave_interpolate(1+trainSize:end);
%%
%Kernel parameters
typeKernel = 'CO2';
paramKernel = 0.2;

flagLearningCurve = 0;

disp('Please wait')
%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%              AOGR
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
th1 = -.03061;
th2 = 100;
regularizationFactorAogr = 0.00;
forgettingFactorAogr = 1;
[expansionCoefficientAogr,dictionaryIndexAogr,learningCurveAogr,CI] = ...
    AOGR(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactorAogr,forgettingFactorAogr,1,th1,th2,[],flagLearningCurve);

%%
%prediction result
output_KRLS_test = zeros(testSize,1);
for jj = 1:testSize
     output_KRLS_test(jj) = expansionCoefficientAogr*...
                ker_eval(testInput(:,jj),trainInput(:,dictionaryIndexAogr),typeKernel,paramKernel);
end

%prediction variance
variance_KRLS_test = zeros(testSize,1);
Q_matrix = inv(gramMatrix(trainInput(:,dictionaryIndexAogr),typeKernel,paramKernel) + regularizationFactorAogr*eye(length(dictionaryIndexAogr)));

for jj = 1:testSize
    k_vector = ker_eval(testInput(:,jj),trainInput(:,dictionaryIndexAogr),typeKernel,paramKernel);
    f_vector = Q_matrix*k_vector;
    variance_KRLS_test(jj) = regularizationFactorAogr + ker_eval(testInput(:,jj),testInput(:,jj),typeKernel,paramKernel) -...
        k_vector'*f_vector;
end

%%
lineWid = 2;
figure;
% plot(trainInput, trainTarget, 'b');
% hold on
plot(testInput, testTarget, 'b--','LineWidth',lineWid);
hold on
plot(testInput, output_KRLS_test, 'r','LineWidth',lineWid);
plot(testInput, output_KRLS_test + 3*sqrt(variance_KRLS_test), 'g--');
plot(testInput, output_KRLS_test - 3*sqrt(variance_KRLS_test), 'g--');
grid
hold off

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
%legend('Conditional Information')
xlabel('year'),ylabel('CO2 concentration')
axis tight

