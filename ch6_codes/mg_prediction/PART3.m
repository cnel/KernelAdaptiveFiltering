%Copyright
%Weifeng Liu CNEL
%July 4 2008
%
%Description:
%m-g time series prediction
%KLMS-SC vs. KLMS-CC for redundency removal
%
%Usage:
%Ch6, m-g time series, redundancy removal, figures
%
%ouside functions called

close all, clear all
clc

%% Data Formatting
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Data Formatting
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load MK30   %MK30 5000*1

varNoise = 0;
inputDimension = 7; 

% Data size for training and testing
trainSize = 500;
testSize = 100;

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
numTh1 = 50;
dicSize_KLMSSC = zeros(numTh1,1);
testMSE_KLMSSC = zeros(numTh1,1);
dicSize_KLMSCC = zeros(numTh1,1);
testMSE_KLMSCC = zeros(numTh1,1);

th_KLMSSC = linspace(-2.5,0.5,numTh1);
th_KLMSCC = linspace(0.1,1.1,numTh1);


flagLearningCurve = 0;

stepSizeScklms = .4;
stepSizeWeightScklms = 0;
stepSizeBiasScklms = 0;
regularizationFactor = 0.001;
th1 = 1000; 
   

disp(['Please wait...',num2str(numTh1),' times loops'])
for ii = 1:numTh1
	disp(ii)
	th_KLMSSC_ii = th_KLMSSC(ii);
	th_KLMSCC_ii = th_KLMSCC(ii);
	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	%             KLMS-SC
    %
    % function [expansionCoefficient,weightVector,biasTerm,learningCurve,dictionaryIndex] = ...
    % SCKLMS(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,...
    % stepSizeFeatureVector,stepSizeWeightVector,stepSizeBias,th1,th2,reguarization, flagLearningCurve)
	%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	[expansionCoefficient_KLMSSC,weightVector,biasTerm,learningCurve_KLMSSC, dictionaryIndex_KLMSSC] = ...
		SCKLMS(trainInput,trainTarget,testInput,testTarget, typeKernel,paramKernel,stepSizeScklms,stepSizeWeightScklms, stepSizeBiasScklms, th1, th_KLMSSC_ii, regularizationFactor, flagLearningCurve);

	y_te = zeros(testSize,1);
	for jj = 1:testSize
		 y_te(jj) = expansionCoefficient_KLMSSC*...
			ker_eval(testInput(:,jj),trainInput(:,dictionaryIndex_KLMSSC),typeKernel,paramKernel);
	end
	testMSE_KLMSSC(ii) = mean((testTarget - y_te).^2);
	dicSize_KLMSSC(ii) = length(dictionaryIndex_KLMSSC);


	% =========end of KLMS-SC================

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	%              KLMS-CC
	%
	%
    % function [expansionCoefficient,weightVector,biasTerm,learningCurve,dictionaryIndex] = ...
    % CCKLMS1(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,...
    % stepSizeFeatureVector,stepSizeWeightVector,stepSizeBias,toleranceCoherence,flagLearningCurve)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
	[expansionCoefficient_KLMSCC,weightVector,biasTerm,learningCurve_KLMSCC,dictionaryIndex_KLMSCC] = ...
		CCKLMS1(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeScklms,stepSizeWeightScklms, stepSizeBiasScklms,th_KLMSCC_ii,flagLearningCurve);

	y_te = zeros(testSize,1);
	for jj = 1:testSize
		 y_te(jj) = expansionCoefficient_KLMSCC*...
			ker_eval(testInput(:,jj),trainInput(:,dictionaryIndex_KLMSCC),typeKernel,paramKernel);
	end
	testMSE_KLMSCC(ii) = mean((testTarget - y_te).^2);
	dicSize_KLMSCC(ii) = length(dictionaryIndex_KLMSCC);


	% =========end of KLMS-CC================

end %th1

%%
lineWid = 2;
figure
subplot(2,1,1)
plot(th_KLMSSC,10*log10(testMSE_KLMSSC),'LineWidth', lineWid)

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('SC Threshold (T_2)'),ylabel('Testing MSE (dB)')
grid on

subplot(2,1,2)
plot(th_KLMSSC,dicSize_KLMSSC,'LineWidth', lineWid)

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');

xlabel('SC Threshold (T_2)'),ylabel('# centers')
grid on

%%
figure
subplot(2,1,1)
plot(th_KLMSCC,10*log10(testMSE_KLMSCC),'LineWidth', lineWid)

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('CC Threshold'),ylabel('Testing MSE (dB)')
grid on

subplot(2,1,2)
plot(th_KLMSCC,dicSize_KLMSCC,'LineWidth', lineWid)

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');

xlabel('CC Threshold'),ylabel('# centers')
grid on
%
%%
figure
plot(dicSize_KLMSSC,10*log10(testMSE_KLMSSC),'b-','LineWidth', lineWid)
hold on
plot(dicSize_KLMSCC,10*log10(testMSE_KLMSCC),'r--','LineWidth', lineWid)
hold off
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('final network size'),ylabel('testing MSE (dB)')
grid on
legend('KLMS-SC','KLMS-CC')
