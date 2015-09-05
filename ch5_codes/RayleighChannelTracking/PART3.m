% rayleigh channel tracking
% Weifeng Liu 
% Sep. 2007.
%
% Description:
% ALD in extended KRLS in rayleigh channel tracking
% complexity reduction
%
% Usage
% Ch 4, figure 4-2
%
% Outside functions used
% EX_KRLS


close all
clear all
clc

var_n = 1e-3;
sqn = sqrt(var_n);

sampleInterval = 0.8e-6; % sampling frequency is 1.25MHz
numberSymbol = 500;
dopplerFrequency = 100; % Doppler frequency
trainSize = numberSymbol;

epsilon = 1e-3;

channelLength = 5;
channel  = zeros(channelLength,trainSize);

for i=1:channelLength;
 channel(i,:) = rayleigh(sampleInterval,numberSymbol,dopplerFrequency);
end

channel = real(channel);

alpha = bessel(0,2*pi*dopplerFrequency*sampleInterval); 
q = 1-alpha*alpha;

% time delay (embedding) length
inputDimension = channelLength;

%Nonlinearity parameter
typeNonlinear = 1;
paramNonlinear = 2;

%Kernel parameters
typeKernel = 'Gauss';
paramKernel = .1;


numThreshold = 10;

disp([num2str(numThreshold), ' thredholds'])

threshold_v = linspace(0,.05,numThreshold);

mseMean = zeros(1,numThreshold);
mseStd = zeros(1,numThreshold);

netSize = zeros(1,numThreshold);

for kkk = 1:numThreshold
    threshold = threshold_v(kkk);

    disp(['threshold = ', num2str(threshold)])
    
    L = 50;
    
    netSizeMC = zeros(L,1);
    ensembleLearningCurveExkrls = zeros(trainSize,1);

    disp([num2str(L),' Monte Carlo simulations. Please wait...'])

    for k = 1:L
        disp(k);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        %       Data Formatting
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        %              Ex-KRLS
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        alphaParameterExkrls = 0.999998;
        regularizationFactorExkrls = 0.01;
        forgettingFactorExkrls = .995;
        qFactorExkrls = 0.0001;
        % flagLearningCurve = 1;

        [expansionCoefficientExkrls,dictionaryIndex,learningCurveExkrls,ALD] = ...
            EX_KRLS_ALD_2(trainInput,trainTarget,typeKernel,paramKernel,alphaParameterExkrls,regularizationFactorExkrls,forgettingFactorExkrls,qFactorExkrls,threshold);
        %=========end of Ex_KRLS================

        ensembleLearningCurveExkrls = ensembleLearningCurveExkrls + learningCurveExkrls;
        netSizeMC(k) = length(dictionaryIndex);
    end

    ensembleLearningCurveExkrls = 10*log10(ensembleLearningCurveExkrls/L);

    mse_Exkrls = mean(ensembleLearningCurveExkrls(end-100:end));

    netSize(kkk) = mean(netSizeMC);
    
    disp('====================================')

    mseMean(kkk) = mean(ensembleLearningCurveExkrls(end-100:end));
    mseStd(kkk) = std(ensembleLearningCurveExkrls(end-100:end));

    disp([num2str(mseMean(kkk)),'+/-',num2str(mseStd(kkk))]);
    disp('====================================')
end

errorbar(threshold_v,mseMean,mseStd)
set(gca, 'FontSize', 14);
% set(gca,'LineWidth',3)
set(gca, 'FontName', 'Arial');
xlabel('ALD threshold'),ylabel('MSE (dB)')
grid on
axis tight

figure
plot(netSize,mseMean,'LineWidth',3)
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
%legend('Conditional Information')
xlabel('network size'),ylabel('MSE (dB)')
grid on
