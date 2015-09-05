%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Copyright Weifeng Liu 
%CNEL
%July 1, 2008
%
%description:
%compare the performance of LMS2 KLMS KAPA2 in noise cancelation
%ensemble learning curves
%
%Usage:
%ch3, noise cancelation, figure 3-4 and table 3-6
%
%Outside functions called or toolboxes used:
%LMS2, sparseKLMS1, sparseKAPA2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


close all, clear all
clc

%% Data Formatting
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Data Formatting
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

varNoise = 0;
inputDimension = 3; 

% Data size for training and testing
trainSize = 1000;
dataSize = trainSize;
referenceNoise = zeros(dataSize,1);
trainInput = zeros(inputDimension,trainSize);
%Kernel parameters
typeKernel = 'Gauss';
paramKernel = 1;
%%
K= 10;
feedbackDimension = 1;
stepSizeWeightLms2 = .2;
stepSizeBiasLms2 = .2;
paramRegularizationLms2 = 0.005;

stepSizeFeatureVectorSklms1 = 0.5;
stepSizeWeightVectorSklms1 = 0.1;
stepSizeBiasSklms1 = 0.1;

toleranceDistance = 0.15;
tolerancePredictError = 0.01;

paramRegularizationSkapa2 = 0.005;
stepSizeFeatureVectorSkapa2 = 0.2;
stepSizeWeightVectorSkapa2 = 0.05;
stepSizeBiasSkapa2 = 0.05;

ensembleLearningCurveLms2 = zeros(trainSize,1);
ensembleLearningCurveSklms1 = zeros(trainSize,1);
ensembleLearningCurveSkapa2 = zeros(trainSize,1);

%%

MC = 200;
networkSizeSklms1 = zeros(MC,1);
networkSizeSkapa1 = zeros(MC,1);
disp([num2str(MC),' Monte Carlo simulations'])
disp('ensemble learning curves are generating...')
for mc = 1:MC
    disp(mc)
    
    noiseSource = rand(dataSize,1)-0.5;

    referenceNoise(1) = noiseSource(1);
    referenceNoise(2) = noiseSource(2)-0.2*referenceNoise(1)-referenceNoise(1)*noiseSource(1)+0.1*noiseSource(1);
    for ii = 3:dataSize
        referenceNoise(ii) = noiseSource(ii) - 0.2*referenceNoise(ii-1) - referenceNoise(ii-1)*noiseSource(ii-1) + 0.1*noiseSource(ii-1) + 0.4*referenceNoise(ii-2);
    end

    %Input training signal with data embedding
    for k = 1:trainSize
        if k < inputDimension
            trainInput(:,k) = [referenceNoise(k:-1:1);zeros(inputDimension-k,1)];
        else
            trainInput(:,k) = referenceNoise(k:-1:k-inputDimension+1);
        end
    end

    % Desired training signal
    trainTarget = zeros(trainSize,1);
    for ii=1:trainSize
        trainTarget(ii) = noiseSource(ii);
    end
%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %             LMS 2
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [tmp,weightVectorLms2,biasTermLms2,learningCurveLms2]= ...
        LMS2(trainInput,trainTarget,feedbackDimension,paramRegularizationLms2,stepSizeWeightLms2,stepSizeBiasLms2);
    %=========end of Linear LMS 2================

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %               SKLMS 1
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [tmp,expansionCoefficientSklms1,dictionarySklms1,weightVectorSklms1,biasTermSklms1,learningCurveSklms1] = ...
        sparseKLMS1(trainInput,trainTarget,feedbackDimension,typeKernel,paramKernel,...
        stepSizeFeatureVectorSklms1,stepSizeWeightVectorSklms1,stepSizeBiasSklms1,toleranceDistance,tolerancePredictError);

	networkSizeSklms1(mc) = length(expansionCoefficientSklms1);

		
    %=========end of KLMS 1================

    % 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %           Normalized KAPA 2
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [tmp,expansionCoefficientSkapa2,dictionarySkapa2,weightVectorSkapa2,biasTermSkapa2,learningCurveSkapa2] = ...
        sparseKAPA2(K,trainInput,trainTarget,feedbackDimension,paramRegularizationSkapa2,typeKernel,paramKernel,stepSizeFeatureVectorSkapa2,stepSizeWeightVectorSkapa2,stepSizeBiasSkapa2,toleranceDistance,tolerancePredictError);
%%
    %=========end of Normalized KAPA 2================
    ensembleLearningCurveLms2 = ensembleLearningCurveLms2 + learningCurveLms2;
    ensembleLearningCurveSklms1 = ensembleLearningCurveSklms1 + learningCurveSklms1;
    ensembleLearningCurveSkapa2 = ensembleLearningCurveSkapa2 + learningCurveSkapa2;
	
	networkSizeSkapa2(mc) = length(expansionCoefficientSkapa2);
% 
end%mc


%%

figure(10),
lineWid = 2;
plot(ensembleLearningCurveLms2/MC,'k-','LineWidth', lineWid)
hold on
plot(ensembleLearningCurveSklms1/MC,'k--','LineWidth', lineWid);
plot(ensembleLearningCurveSkapa2/MC,'k:','LineWidth', lineWid)
% 
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
legend('NLMS','KLMS-NC','KAPA-2-NC')
hold off
xlabel('iteration'),ylabel('MSE')
grid on


disp('====================================')
disp('<<LMS2')

nrLms2 = 10*log10(ensembleLearningCurveLms2(end-100:end)/MC/mean(trainTarget.^2));

mseMean = mean(nrLms2);

stdMean = std(nrLms2);

disp(num2str(mseMean));
disp(num2str(stdMean));

disp('<<SKLMS1')

nrSklms1 = 10*log10(ensembleLearningCurveSklms1(end-100:end)/MC/mean(trainTarget.^2));

mseMean = mean(nrSklms1);

stdMean = std(nrSklms1);

disp(num2str(mseMean));
disp(num2str(stdMean));

netMean = mean(networkSizeSklms1);
netStd = std(networkSizeSklms1);

disp([num2str(netMean),'+\-',num2str(netStd)])

disp('<<SKAPA2')
nrSkapa2 = 10*log10(ensembleLearningCurveSkapa2(end-100:end)/MC/mean(trainTarget.^2));

mseMean = mean(nrSkapa2);

stdMean = std(nrSkapa2);

disp(num2str(mseMean));
disp(num2str(stdMean));

netMean = mean(networkSizeSkapa2);
netStd = std(networkSizeSkapa2);

disp([num2str(netMean),'+\-',num2str(netStd)])
disp('====================================')