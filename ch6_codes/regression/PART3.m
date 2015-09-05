%Copyright
%Weifeng Liu CNEL
%July 4 2008
%
%Description:
%Evaluation surprise in nonlinear regression
%ALD and surprise for adnormality detection and removal
%
%Usage:
%Ch5, nonlinear regression, figures 5-5, 5-6, 5-7
%
%ouside functions called
%AOGR1 and AOGR2, nlG

close all, clear all
clc
%% Data Formatting
%addpath('H:\APrograms\toolBoxKAPA_old');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Data Formatting
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Input signal
inputSignal = randn(500,1);
% figure(1),plot(inputSignal)

% Synthesize the linear filter
linearFilter = 1;

% time delay (embedding) length
inputDimension = length(linearFilter);

%Nonlinearity parameter
typeNonlinear = 2;
paramNonlinear = 2;

%Kernel parameters
typeKernel = 'Gauss';
paramKernel = 0.2;

% Data size for training and testing
trainSize = 200;
testSize = 100;

%Input training signal with data embedding
trainInput = zeros(inputDimension,trainSize);
for k = 1:trainSize
    trainInput(:,k) = inputSignal(k:k+inputDimension-1);
end

%Input test data with embedding
testInput = zeros(inputDimension,testSize);
for k = 1:testSize
    testInput(:,k) = inputSignal(k+trainSize:k+inputDimension-1+trainSize);
end

%Desired training signal
trainTarget = zeros(trainSize,1);
for ii=1:trainSize
    trainTarget(ii) = inputSignal(ii:ii+inputDimension-1)'*linearFilter;
end
%Pass through the nonlinearity
trainTarget = nlG(trainTarget,paramNonlinear,typeNonlinear);


%%
%outlier added at 100, 110, 120, 130, 140,150,160 by changing its sign
for ii = 1:15
    trainTarget(40+ii*10) = -trainTarget(40+ii*10);
end
%%
%Desired training signal
testTarget = zeros(testSize,1);
for ii=1:testSize
    testTarget(ii) = inputSignal(ii+trainSize:ii+inputDimension-1+trainSize)'*linearFilter;
end
%Pass through the nonlinearity
testTarget = nlG(testTarget,paramNonlinear,typeNonlinear);

%%
flagLearningCurve = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Algorithms Debugging
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

th1Aogr1 = -3.14;
th1Aogr2 = -3.3754;
th2Aogr1 = 200;
th2Aogr2 = 10^6;
regularizationFactorAogr = 0.001;
forgettingFactorAogr = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%              AOGR 1
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[expansionCoefficientAogr1,dictionaryIndexAogr1,learningCurveAogr1,CI_Aogr1] = ...
    AOGR1(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactorAogr,forgettingFactorAogr,th1Aogr1,th2Aogr1,flagLearningCurve);



% =========end of AOGR1================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%              AOGR 2
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[expansionCoefficientAogr2,dictionaryIndexAogr2,learningCurveAogr2,CI_Aogr2] = ...
    AOGR2(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactorAogr,forgettingFactorAogr,th1Aogr2,th2Aogr2,flagLearningCurve);


% =========end of AOGR2================

%%
lineWid = 2;

figure
plot(learningCurveAogr1,'-b','LineWidth', lineWid);
hold on
plot(learningCurveAogr2,'--r','LineWidth', lineWid);
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
legend('KRLS-SC','KRLS-ALD')
xlabel('iteration'),ylabel('testing MSE')
grid on


%%
figure
subplot(2,1,1);
plot(CI_Aogr1,'-b','LineWidth', lineWid)
axis([0, 200, 0, 200])
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('iteration'),ylabel('SC')
grid on

subplot(2,1,2);
plot(CI_Aogr2,'--r','LineWidth', lineWid);

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
%legend('ALD')
xlabel('iteration'),ylabel('ALD')
grid on

ii = 50:10:190;
figure
plot(trainInput, trainTarget, 'ob');
hold on
plot(trainInput(ii), trainTarget(ii), 'ow');
plot(trainInput(ii), trainTarget(ii), 'xr');
hold off
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
%legend('normal data','outlier')
xlabel('input'),ylabel('output')
grid on

%plot(trainInput, output_KRLS, '.r');