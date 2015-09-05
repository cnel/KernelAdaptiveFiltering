% rayleigh channel tracking
% Weifeng Liu 
% Sep. 2007.
%
% Description:
% compare extended KRLS and KRLS, KLMS in rayleigh channel tracking
% learning curves
%
% Usage
% Ch 5
%
% Outside functions used
% KRLS, KLMS and EX_KRLS


close all
clear all
%clc

var_n = 1e-3;
sqn = sqrt(var_n);

sampleInterval = 0.8e-6; % sampling frequency is 1MHz
numberSymbol = 3000;
dopplerFrequency = 100; % Doppler frequency
trainSize = numberSymbol;

epsilon = 1e-3;

channelLength = 5;
channel  = zeros(channelLength,trainSize);



alpha = bessel(0,2*pi*dopplerFrequency*sampleInterval); 
q = 1-alpha*alpha;


ensembleLearningCurveKlms = zeros(trainSize,1);
ensembleLearningCurveKrls = zeros(trainSize,1);
ensembleLearningCurveExkrls = zeros(trainSize,1);
% time delay (embedding) length
inputDimension = channelLength;

%Nonlinearity parameter
typeNonlinear = 1;
paramNonlinear = 2;

%Kernel parameters
typeKernel = 'Gauss';
paramKernel = .05;

alphaParameterExkrls = 0.99999;
regularizationFactorExkrls = 0.01;
forgettingFactorExkrls = .995;
qFactorExkrls = 1e-4;
aldThExkrls = 0.03;

regularizationFactorKrls = 0.01;
forgettingFactorKrls = 1;
aldThKrls = log(aldThExkrls)/2;

flagLearningCurve = 1;
toleranceDistance = 0.7;
tolerancePredictError = 0.2;

stepSizeKlms1 = .15;
stepSizeWeightKlms1 = 0;
stepSizeBiasKlms1 = 0;

L = 10;

disp([num2str(L),' Monte Carlo simulations. Please wait...'])

for k = 1:L
    disp(k);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %       Data Formatting
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for i=1:channelLength;
        channel(i,:) = rayleigh(sampleInterval,numberSymbol,dopplerFrequency);
    end

    channel = real(channel);

     % Input signal
    inputSignal = randn(1,trainSize + channelLength);
    noise = sqn*randn(1,trainSize); 


    %Input training signal with data embedding
    trainInput = zeros(inputDimension,trainSize);
    for kk = 1:trainSize
        trainInput(:,kk) = inputSignal(kk:kk+inputDimension-1);
    end

    %Desired training signal
    trainTarget = zeros(trainSize,1);
    for ii=1:trainSize
        trainTarget(ii) = trainInput(:,ii)'*channel(:,ii);
    end

    trainTarget = trainTarget + noise';
    %Pass through the nonlinearity
    trainTarget = nlG(trainTarget,paramNonlinear,typeNonlinear);

    %=========sparse Kernel LMS 1===================

    [expansionCoefficientKlms1,weightVectorKlms1,biasTermKlms1,learningCurveKlms,dictionaryIndexKlms1,netSizeKlms1] = ...
        sparseKLMS1s(trainInput,trainTarget,typeKernel,paramKernel,...
        stepSizeKlms1,stepSizeWeightKlms1,stepSizeBiasKlms1,toleranceDistance,tolerancePredictError,flagLearningCurve);
    %=========end of sparse Kernel LMS================

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %              KRLS
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [expansionCoefficientKrls,dictionaryIndexKrls,learningCurveKrls, aldKrls] = ...
        KRLS_ALD(trainInput,trainTarget,typeKernel,paramKernel,regularizationFactorKrls,forgettingFactorKrls, aldThKrls);

    % =========end of KRLS================

    ensembleLearningCurveKlms = ensembleLearningCurveKlms + learningCurveKlms;
    ensembleLearningCurveKrls = ensembleLearningCurveKrls + learningCurveKrls;

 
end

ensembleLearningCurveKlms = 10*log10(ensembleLearningCurveKlms/L);
ensembleLearningCurveKrls = 10*log10(ensembleLearningCurveKrls/L);

mse_Klms = mean(ensembleLearningCurveKlms(end-100:end));
mse_Krls = mean(ensembleLearningCurveKrls(end-100:end));

disp([num2str(mse_Klms),' ',num2str(mse_Krls),' '])

figure
plot(1:trainSize,ensembleLearningCurveKlms,'b:','LineWidth',1)
hold on
plot(1:trainSize,ensembleLearningCurveKrls,'r-','LineWidth',1)


xlabel('iteration')
ylabel('dB')
grid
axis tight
legend('KLMS-NC','KRLS-ALD')

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('iteration'),ylabel('testing MSE (dB)')

disp('====================================')

mseMean = mean(ensembleLearningCurveKlms(end-100:end));
mseStd = std(ensembleLearningCurveKlms(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

mseMean = mean(ensembleLearningCurveKrls(end-100:end));
mseStd = std(ensembleLearningCurveKrls(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('====================================')

