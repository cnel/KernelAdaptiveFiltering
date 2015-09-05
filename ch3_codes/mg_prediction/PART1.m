%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Copyright Weifeng Liu 
%CNEL
%July 1, 2008
%
%description:
%compare the performance of LMS KLMS KAPA KRLS in Mackey Glass time series short term prediction
%learning curve
%
%Usage:
%ch3, chaotic time seriese prediction
%
%Outside functions called:
%LMS1 KAPA1 KAPA2, KLMS1, KRLS, SW-KRLS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all, clear all
clc

%% Data Formatting
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Data Formatting
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load MK30   %MK30 5000*1

MK30 = MK30(1000:5000);
varNoise = 0.001;
inputDimension = 7; 

% Data size for training and testing
trainSize = 500;
testSize = 100;

inputSignal = MK30 + sqrt(varNoise)*randn(size(MK30));
inputSignal = inputSignal - mean(inputSignal);

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

disp('learning curves are generating. Please wait...')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%             LMS 1
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stepSizeWeightLms1 = .04;
stepSizeBiasLms1 = .04;
flagLearningCurve = 1;
[aprioriErrLms1,weightVectorLms1,biasTermLms1,learningCurveLms1]= ...
    LMS1(trainInput,trainTarget,testInput,testTarget,stepSizeWeightLms1,stepSizeBiasLms1,flagLearningCurve);
%=========end of Linear LMS 1================


K = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%               KAPA 1
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stepSizeKapa1 = .04;
stepSizeWeightKapa1 = 0;
stepSizeBiasKapa1 = 0;
% flagLearningCurve = 1;
% K = 10;
[expansionCoeffKapa1,weightVectorKapa1,biasTermKapa1,learningCurveKapa1] = ...
    KAPA1(K,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeKapa1,stepSizeWeightKapa1,stepSizeBiasKapa1,flagLearningCurve);
%=========end of KAPA 1================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%           Normalized KAPA 2
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
regularizationFactorKapa2 = .1;
stepSizeKapa2 = .04;
stepSizeWeightKapa2 = 0;
stepSizeBiasKapa2 = 0;
% flagLearningCurve = 1;
% K = 5;

[expansionCoeffKapa2,weightVectorKapa2,biasTermKapa2,learningCurveKapa2] = ...
    KAPA2(K,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactorKapa2,stepSizeKapa2,stepSizeWeightKapa2,stepSizeBiasKapa2,flagLearningCurve);
%=========end of Normalized KAPA 2================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%               KLMS 1
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stepSizeKlms1 = .2;
stepSizeWeightKlms1 = 0;
stepSizeBiasKlms1 = 0;
% flagLearningCurve = 1;

[expansionCoefficientKlms1,weightVectorKlms1,biasTermKlms1,learningCurveKlms1] = ...
    KLMS1(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeKlms1,stepSizeWeightKlms1,stepSizeBiasKlms1,flagLearningCurve);
%=========end of KLMS 1================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%        Sliding window KRLS
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
paramRegularizationSw = 0.01;
Ksw = 50;

[expansionCoefficientSwkrls,learningCurveSwkrls] = ...
    slidingWindowKRLS(Ksw,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,paramRegularizationSw,flagLearningCurve);

%=========end of sliding window KRLS================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%              KRLS
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

regularizationFactorKrls = 0.1;
forgettingFactorKrls = 1;

% flagLearningCurve = 1;
[expansionCoefficientKrls,learningCurveKrls] = ...
    KRLS(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactorKrls,forgettingFactorKrls,flagLearningCurve);

% =========end of KRLS================

%%
figure(1),
plot(learningCurveLms1,'k:','LineWidth',1)
hold on
plot(learningCurveKlms1,'k-','LineWidth',3)
plot(learningCurveKapa1,'k--','LineWidth',2)
plot(learningCurveKapa2,'k-.','LineWidth',2)
plot(learningCurveSwkrls,'k-','LineWidth',1)
plot(learningCurveKrls,'k:','LineWidth',3)

legend('LMS','KLMS','KAPA-1','KAPA-2','SW-KRLS','KRLS')

hold off
xlabel('iteration'),ylabel('MSE')
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
grid off
set(gca, 'YScale','log')

%%
disp('====================================')
disp('>>LMS1')
mseMean = mean(learningCurveLms1(end-100:end));
mseStd = std(learningCurveLms1(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('>>KLMS1')
mseMean = mean(learningCurveKlms1(end-100:end));
mseStd = std(learningCurveKlms1(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('>>KAPA1')
mseMean = mean(learningCurveKapa1(end-100:end));
mseStd = std(learningCurveKapa1(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('>>KAPA2')
mseMean = mean(learningCurveKapa2(end-100:end));
mseStd = std(learningCurveKapa2(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('>>SW-KRLS')
mseMean = mean(learningCurveSwkrls(end-100:end));
mseStd = std(learningCurveSwkrls(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('>>KRLS')
mseMean = mean(learningCurveKrls(end-100:end));
mseStd = std(learningCurveKrls(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('====================================')

