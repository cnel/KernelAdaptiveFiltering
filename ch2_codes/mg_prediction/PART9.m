%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Copyright Weifeng Liu 
%CNEL
%July 1, 2008
%
%description:
%evaluate the effectiveness of novelty criterion for KLMS in Mackey Glass time series short term prediction
%Learning curve
%
%Usage:
%ch2, chaotic time seriese prediction
%
%Outside functions called:
%KLMS1_LC, sparseKLMS1
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
trainSize = 1000;
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

disp('Learning curves are generating')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%               KLMS 1
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stepSizeKlms1 = .1;
stepSizeWeightKlms1 = 0;
stepSizeBiasKlms1 = 0;
flagLearningCurve = 1;

[expansionCoefficientKlms1,weightVectorKlms1,biasTermKlms1,learningCurveKlms1] = ...
    KLMS1_LC(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeKlms1,stepSizeWeightKlms1,stepSizeBiasKlms1,flagLearningCurve);
%=========end of KLMS 1================


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
tolDistance = 0.1;
tolError = 0.05;

[expansionCoefficientSklms1_1,weightVectorSklms1,biasTermSklms1,learningCurveSklms1_1,dictIndexSklms1_1] = ...
    sparseKLMS1(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeSklms1,stepSizeWeightSklms1,stepSizeBiasSklms1,tolDistance,tolError,flagLearningCurve);
%=========end of sparse KLMS 1================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%        sparse KLMS1 2
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=========sparseKLMS1===================
stepSizeSklms1 = .1;
stepSizeWeightSklms1 = 0;
stepSizeBiasSklms1 = 0;
flagLearningCurve = 1;
tolDistance = 0.05;
tolError = 0.05;

[expansionCoefficientSklms1_2,weightVectorSklms1,biasTermSklms1,learningCurveSklms1_2,dictIndexSklms1_2] = ...
    sparseKLMS1(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeSklms1,stepSizeWeightSklms1,stepSizeBiasSklms1,tolDistance,tolError,flagLearningCurve);
%=========end of sparse KLMS 1================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%        sparse KLMS1 3
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=========sparseKLMS1===================
stepSizeSklms1 = .1;
stepSizeWeightSklms1 = 0;
stepSizeBiasSklms1 = 0;
flagLearningCurve = 1;
tolDistance = 0.2;
tolError = 0.05;

[expansionCoefficientSklms1_3,weightVectorSklms1,biasTermSklms1,learningCurveSklms1_3,dictIndexSklms1_3] = ...
    sparseKLMS1(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeSklms1,stepSizeWeightSklms1,stepSizeBiasSklms1,tolDistance,tolError,flagLearningCurve);
%=========end of sparse KLMS 1================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%        sparse KLMS1 4
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=========sparseKLMS1===================
stepSizeSklms1 = .1;
stepSizeWeightSklms1 = 0;
stepSizeBiasSklms1 = 0;
flagLearningCurve = 1;
tolDistance = 0.05;
tolError = 0.1;

[expansionCoefficientSklms1_4,weightVectorSklms1,biasTermSklms1,learningCurveSklms1_4,dictIndexSklms1_4] = ...
    sparseKLMS1(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeSklms1,stepSizeWeightSklms1,stepSizeBiasSklms1,tolDistance,tolError,flagLearningCurve);
%=========end of sparse KLMS 1================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%        sparse KLMS1 5
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=========sparseKLMS1===================
stepSizeSklms1 = .1;
stepSizeWeightSklms1 = 0;
stepSizeBiasSklms1 = 0;
flagLearningCurve = 1;
tolDistance = 0.05;
tolError = 0.02;

[expansionCoefficientSklms1_5,weightVectorSklms1,biasTermSklms1,learningCurveSklms1_5,dictIndexSklms1_5] = ...
    sparseKLMS1(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeSklms1,stepSizeWeightSklms1,stepSizeBiasSklms1,tolDistance,tolError,flagLearningCurve);
%=========end of sparse KLMS 1================


%%
figure

plot(learningCurveKlms1,'k-','LineWidth',2),hold on
plot(learningCurveSklms1_1,'k--','LineWidth',2)
legend('KLMS','KLMS-NC')
hold off

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
hold off
grid on

xlabel('iteration'),ylabel('testing MSE')

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
hold off
grid on

%%
figure

plot(learningCurveKlms1,'k-','LineWidth',2),hold on
plot(learningCurveSklms1_1,'k--','LineWidth',2)


legend('KLMS','KLMS-NC')

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
hold off
set(gca,'YScale','log')
xlabel('iteration'),ylabel('testing MSE')
grid on
%%

disp('====================================')

disp('>>KLMS1')
mseMean = mean(learningCurveKlms1(end-100:end));
mseStd = std(learningCurveKlms1(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('>>SKLMS1-1')
mseMean = mean(learningCurveSklms1_1(end-100:end));
mseStd = std(learningCurveSklms1_1(end-100:end));
disp([num2str(mseMean),'+/-',num2str(mseStd)]);
disp(length(expansionCoefficientSklms1_1));

disp('>>SKLMS1-2')
mseMean = mean(learningCurveSklms1_2(end-100:end));
mseStd = std(learningCurveSklms1_2(end-100:end));
disp([num2str(mseMean),'+/-',num2str(mseStd)]);
disp(length(expansionCoefficientSklms1_2));

disp('>>SKLMS1-3')
mseMean = mean(learningCurveSklms1_3(end-100:end));
mseStd = std(learningCurveSklms1_3(end-100:end));
disp([num2str(mseMean),'+/-',num2str(mseStd)]);
disp(length(expansionCoefficientSklms1_3));

disp('>>SKLMS1-4')
mseMean = mean(learningCurveSklms1_4(end-100:end));
mseStd = std(learningCurveSklms1_4(end-100:end));
disp([num2str(mseMean),'+/-',num2str(mseStd)]);
disp(length(expansionCoefficientSklms1_4));

disp('>>SKLMS1-5')
mseMean = mean(learningCurveSklms1_5(end-100:end));
mseStd = std(learningCurveSklms1_5(end-100:end));
disp([num2str(mseMean),'+/-',num2str(mseStd)]);
disp(length(expansionCoefficientSklms1_5));

disp('====================================')


