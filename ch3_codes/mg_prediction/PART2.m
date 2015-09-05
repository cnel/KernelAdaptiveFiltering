%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Copyright Weifeng Liu 
%CNEL
%July 1, 2008
%
%description:
%test the novelty criterion for KLMS KAPA in Mackey Glass time series short term prediction
%KLMS1, sparseKLMS1 vs KAPA1-2, sparseKAPA1-2
%Learning curve
%
%Usage:
%ch3, chaotic time seriese prediction, figure 3-2 and table 3-4
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
inputDimension = 7; 

% Data size for training and testing
trainSize = 1500;
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
%               KAPA 1
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stepSizeKapa1 = .05;
stepSizeWeightKapa1 = 0;
stepSizeBiasKapa1 = 0;
flagLearningCurve = 1;
K = 10;
[expansionCoeffKapa1,weightVectorKapa1,biasTermKapa1,learningCurveKapa1] = ...
    KAPA1(K,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeKapa1,stepSizeWeightKapa1,stepSizeBiasKapa1,flagLearningCurve);
%=========end of KAPA 1================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%           Normalized KAPA 2
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
regularizationFactorKapa2 = 0.1;
stepSizeKapa2 = .05;
stepSizeWeightKapa2 = 0;
stepSizeBiasKapa2 = 0;
% flagLearningCurve = 1;
% K = 5;

[expansionCoeffKapa2,weightVectorKapa2,biasTermKapa2,learningCurveKapa2] = ...
    KAPA2(K,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactorKapa2,stepSizeKapa2,stepSizeWeightKapa2,stepSizeBiasKapa2,flagLearningCurve);
%=========end of Normalized KAPA 2================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%        sparseKAPA1
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stepSizeSkapa1 = .05;
stepSizeWeightSkapa1 = 0;
stepSizeBiasSkapa1 = 0;
flagLearningCurve = 1;
tolDistance = 0.1;
tolError = 0.05;

[expansionCoefficientSkapa1,dictIndexSkapa1,weightVectorSkapa1,biasTermSkapa1,learningCurveSkapa1] = ...
    sparseKAPA1(K,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeSkapa1,stepSizeWeightSkapa1,stepSizeBiasSkapa1,tolDistance,tolError,flagLearningCurve);
%=========end of sparse KAPA 1================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%        sparseKAPA2
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stepSizeSkapa2 = .05;
stepSizeWeightSkapa2 = 0;
stepSizeBiasSkapa2 = 0;
paramRegularization = .1;
flagLearningCurve = 1;
tolDistance = 0.1;
tolError = 0.05;

[expansionCoefficientSkapa2,dictIndexSkapa2,weightVectorSkapa2,biasTermSkapa2,learningCurveSkapa2] = ...
    sparseKAPA2(K,trainInput,trainTarget,testInput,testTarget,paramRegularization,typeKernel,paramKernel,stepSizeSkapa2,stepSizeWeightSkapa2,stepSizeBiasSkapa2,tolDistance,tolError,flagLearningCurve);
%=========end of sparse KAPA 2================


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %
% %               KLMS 1
% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% stepSizeKlms1 = .1;
% stepSizeWeightKlms1 = 0;
% stepSizeBiasKlms1 = 0;
% % flagLearningCurve = 1;
% 
% [expansionCoefficientKlms1,weightVectorKlms1,biasTermKlms1,learningCurveKlms1] = ...
%     KLMS1(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeKlms1,stepSizeWeightKlms1,stepSizeBiasKlms1,flagLearningCurve);
% %=========end of KLMS 1================
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %
% %        sparse KLMS1
% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %=========sparseKLMS1===================
% stepSizeSklms1 = .1;
% stepSizeWeightSklms1 = 0;
% stepSizeBiasSklms1 = 0;
% flagLearningCurve = 1;
% tolDistance = 0.1;
% tolError = 0.05;
% 
% [expansionCoefficientSklms1,weightVectorSklms1,biasTermSklms1,learningCurveSklms1,dictIndexSklms1] = ...
%     sparseKLMS1(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeSklms1,stepSizeWeightSklms1,stepSizeBiasSklms1,tolDistance,tolError,flagLearningCurve);
% %=========end of sparse KLMS 1================


%%
figure

% subplot(311)
% plot(learningCurveKlms1,'k-'),hold on
% plot(learningCurveSklms1,'k--')
% legend('KLMS','KLMS-NC')
% hold off

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
hold off
grid on

subplot(312)
plot(learningCurveKapa1,'c-'),hold on
plot(learningCurveSkapa1,'c.')
legend('KAPA-1','KAPA-1-NC')

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
hold off
grid on

subplot(313)
plot(learningCurveKapa2,'c:')
hold on
plot(learningCurveSkapa2,'c--')
legend('KAPA-2','KAPA-2-NC')

xlabel('iteration'),ylabel('testing MSE')

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
hold off
grid on

%%
figure

%plot(learningCurveKlms1,'k-','LineWidth',1),hold on
%plot(learningCurveSklms1,'k:','LineWidth',1)

plot(learningCurveKapa1,'b-','LineWidth',1),hold on
plot(learningCurveSkapa1,'b:','LineWidth',1)

plot(learningCurveKapa2,'r-','LineWidth',3)
plot(learningCurveSkapa2,'r:','LineWidth',3)

legend('KAPA-1','KAPA-1-NC','KAPA-2','KAPA-2-NC')

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
hold off
set(gca,'YScale','log')
xlabel('iteration'),ylabel('testing MSE')
%%

disp('====================================')

% disp('>>KLMS1')
% mseMean = mean(learningCurveKlms1(end-100:end));
% mseStd = std(learningCurveKlms1(end-100:end));
% 
% disp([num2str(mseMean),'+/-',num2str(mseStd)]);

% disp('>>SKLMS1')
% mseMean = mean(learningCurveSklms1(end-100:end));
% mseStd = std(learningCurveSklms1(end-100:end));
% disp([num2str(mseMean),'+/-',num2str(mseStd)]);
% disp(length(expansionCoefficientSklms1));

disp('>>KAPA1')
mseMean = mean(learningCurveKapa1(end-100:end));
mseStd = std(learningCurveKapa1(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('>>SKAPA1')
mseMean = mean(learningCurveSkapa1(end-100:end));
mseStd = std(learningCurveSkapa1(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);
disp(length(expansionCoefficientSkapa1));

disp('>>KAPA2')
mseMean = mean(learningCurveKapa2(end-100:end));
mseStd = std(learningCurveKapa2(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('>>SKAPA2')
mseMean = mean(learningCurveSkapa2(end-100:end));
mseStd = std(learningCurveSkapa2(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);
disp(length(expansionCoefficientSkapa2));

disp('====================================')


