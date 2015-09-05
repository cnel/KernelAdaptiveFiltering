%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Copyright Weifeng Liu 
%CNEL
%July 1, 2008
%
%description:
%evaluate the effectiveness of novelty criterion for KLMS in Mackey Glass time series short term prediction
%growth curve
%
%Usage:
%ch2, chaotic time seriese prediction, figure 2-8
%
%Outside functions called:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all, clear all
clc
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Data Formatting
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load MK30   %MK30 5000*1

varNoise = 0.0001;
inputDimension = 10; 

% Data size for training and testing
trainSize = 4500;
testSize = 200;

inputSignal = MK30 + sqrt(varNoise)*randn(size(MK30));
% inputSignal = inputSignal - mean(inputSignal);

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

% One step ahead prediction
predictionHorizon = 1;

% Desired training signal
trainTarget = zeros(trainSize,1);
for ii=1:trainSize
    trainTarget(ii) = inputSignal(ii+inputDimension+predictionHorizon-1);
end

% Desired training signal
testTarget = zeros(testSize,1);
for ii=1:testSize
    testTarget(ii) = inputSignal(ii+inputDimension+trainSize+predictionHorizon-1);
end


%Kernel parameters
typeKernel = 'Gauss';
paramKernel = 1;
%%

disp('Growth curves are generating...')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%        sparse KLMS1 1
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=========sparseKLMS1===================
stepSizeSklms1 = .1;
stepSizeWeightSklms1 = 0;
stepSizeBiasSklms1 = 0;
flagLearningCurve = 1;
tolDistance = 0.05;
tolError = 0.1;

[expansionCoefficientSklms1,weightVectorSklms1,biasTermSklms1,learningCurveSklms1,dictIndexSklms1] = ...
    sparseKLMS1(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeSklms1,stepSizeWeightSklms1,stepSizeBiasSklms1,tolDistance,tolError,flagLearningCurve);
%=========end of sparse KLMS 1================

growthCurve = zeros(trainSize,1);
for kk=1:length(dictIndexSklms1)-1
    growthCurve(dictIndexSklms1(kk):dictIndexSklms1(kk+1)-1) = kk;
end
growthCurve(dictIndexSklms1(kk+1):end) = (kk+1)*ones(trainSize-dictIndexSklms1(kk+1)+1,1);


interval = 100;
halfInterval = interval/2;

growthRateCurve = zeros(trainSize,1);
for kk=1:trainSize - halfInterval
    growthRateCurve(kk) = (growthCurve(kk+halfInterval) - growthCurve(1))/(kk+halfInterval-1);
end

for kk = halfInterval+1 : trainSize- halfInterval
    growthRateCurve(kk) = (growthCurve(kk+halfInterval) - growthCurve(kk-halfInterval))/interval;
end

for kk=trainSize - halfInterval+1 : trainSize
    growthRateCurve(kk) = (growthCurve(trainSize) - growthCurve(kk-halfInterval))/(trainSize - kk + halfInterval);
end

%%
figure

plot(growthCurve,'b','LineWidth',2)
legend('growth curve')
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('iteration'),ylabel('network size')
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
grid on
figure

plot(growthRateCurve,'b','LineWidth',2)
legend('growth rate')
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('iteration'),ylabel('average growth rate')
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
grid on



figure

plot(learningCurveSklms1,'b','LineWidth',2)
legend('KLMS-NC')
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('iteration'),ylabel('testing MSE')
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
grid on

%%
figure

plot(learningCurveSklms1,'b','LineWidth',2)

legend('KLMS-NC')

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
set(gca,'YScale','log')
xlabel('iteration'),ylabel('testing MSE')
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
grid on
%%

disp('====================================')

disp('>>SKLMS1')
mseMean = mean(learningCurveSklms1(end-100:end));
mseStd = std(learningCurveSklms1(end-100:end));
disp([num2str(mseMean),'+/-',num2str(mseStd)]);
disp(length(expansionCoefficientSklms1));

disp('====================================')


