% Lorenz model tracking
% Weifeng Liu 
% Jul. 2008.
%
% Description:
% kernel size in EX-KRLS in Lorenz signal modeling
% kernel size
%
% Usage
% Ch 4
%
% Outside functions used
% EX-KRLS


close all
clear all
clc

load lorenz.mat
%lorenz2 50000*1 double
lorenz2 = lorenz2 - mean(lorenz2);
lorenz2 = lorenz2/std(lorenz2);

trainSize = 500;
inputDimension = 5;

predictionHorizon = 10;

%Kernel parameters
typeKernel = 'Gauss';
paramKernel_v = logspace(-1,1,10);
%paramKernel = 1;

L = 20;

for h=1:length(paramKernel_v)
	
	paramKernel = paramKernel_v(h);
    disp([num2str(paramKernel),' -> paramKernel'])
	disp([num2str(L),' Monte Carlo simulations. Please wait...'])
    ensembleLearningCurveExkrls = zeros(trainSize,1);
    
    for k = 1:L
        disp(k);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        %       Data Formatting
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Input training signal with data embedding
        inputSignal = lorenz2(k*trainSize:k*trainSize+trainSize+inputDimension+predictionHorizon+1);

        trainInput = zeros(inputDimension,trainSize);
        for kk = 1:trainSize
            trainInput(:,kk) = inputSignal(kk:kk+inputDimension-1);
        end

        % Desired training signal
        trainTarget = zeros(trainSize,1);
        for ii=1:trainSize
            trainTarget(ii) = inputSignal(ii+inputDimension+predictionHorizon-1);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        %              Ex-KRLS
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        alphaParameterExkrls = 1;
        regularizationFactorExkrls = 0.001;
        forgettingFactorExkrls = 0.99;
        qFactorExkrls = 0.01;
        % flagLearningCurve = 1;

        [expansionCoefficientExkrls,learningCurveExkrls] = ...
            EX_KRLS(trainInput,trainTarget,typeKernel,paramKernel,alphaParameterExkrls,regularizationFactorExkrls,forgettingFactorExkrls,qFactorExkrls);
        %=========end of Ex_KRLS================

        ensembleLearningCurveExkrls = ensembleLearningCurveExkrls + learningCurveExkrls;

    end
    signalPower = std(inputSignal)^2;

    %%
    ensembleLearningCurveExkrls_dB = 10*log10(ensembleLearningCurveExkrls/L/signalPower);

    disp('====================================')

    mseMean(h) = mean(ensembleLearningCurveExkrls_dB(end-100:end));
    mseStd(h) = std(ensembleLearningCurveExkrls_dB(end-100:end));
    disp([num2str(mseMean(h)),'+/-',num2str(mseStd(h))]);

    disp('====================================') 
end


%%
errorbar(paramKernel_v,mseMean,mseStd)
set(gca, 'FontSize', 14);

set(gca, 'FontName', 'Arial');
xlabel('kernel parameter'),ylabel('MSE (dB)')
grid on
axis tight

