%Copyright
%Weifeng Liu CNEL
%July 4 2008
%
%Description:
%Evaluation the conditional information in m-g time series prediction
%ALD and Conditional info for redundency removal
%
%Usage:
%Ch5, m-g time series, redundancy removal, figures
%
%ouside functions called
%AOGR1 and AOGR2

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
testSize = 50;

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
numTh1 = 30;
dicSizeAogr1 = zeros(numTh1,1);
testMSEAogr1 = zeros(numTh1,1);
dicSizeAogr2 = zeros(numTh1,1);
testMSEAogr2 = zeros(numTh1,1);

th1Aogr1 = linspace(-3.8,0.1,numTh1);
th1Aogr2 = linspace(-3.8,0.1,numTh1);
th2Aogr1 = 10^7;
th2Aogr2 = 10^7;

regularizationFactorAogr = 0.001;
forgettingFactorAogr = 1;
flagLearningCurve = 0;

disp(['Please wait...',num2str(numTh1),' times loops'])
for ii = 1:numTh1
	disp(ii)
	th1Aogr1_ii = th1Aogr1(ii);
	th1Aogr2_ii = th1Aogr2(ii);
	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	%              AOGR 1 (no input apriori)
	%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	[expansionCoefficientAogr1,dictionaryIndexAogr1,learningCurveAogr1,CI_Aogr1] = ...
		AOGR_CI(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactorAogr,forgettingFactorAogr,th1Aogr1_ii,th2Aogr1,[],flagLearningCurve);

	y_te = zeros(testSize,1);
	for jj = 1:testSize
		 y_te(jj) = expansionCoefficientAogr1*...
			ker_eval(testInput(:,jj),trainInput(:,dictionaryIndexAogr1),typeKernel,paramKernel);
	end
	testMSEAogr1(ii) = mean((testTarget - y_te).^2);
	dicSizeAogr1(ii) = length(dictionaryIndexAogr1);


	% =========end of AOGR1================

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	%              AOGR 2 (ALD)
	%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


	[expansionCoefficientAogr2,dictionaryIndexAogr2,learningCurveAogr2,CI_Aogr2] = ...
		AOGR_ALD(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactorAogr,forgettingFactorAogr,th1Aogr2_ii,th2Aogr2,flagLearningCurve);

	y_te = zeros(testSize,1);
	for jj = 1:testSize
		 y_te(jj) = expansionCoefficientAogr2*...
			ker_eval(testInput(:,jj),trainInput(:,dictionaryIndexAogr2),typeKernel,paramKernel);
	end
	testMSEAogr2(ii) = mean((testTarget - y_te).^2);
	dicSizeAogr2(ii) = length(dictionaryIndexAogr2);


	% =========end of AOGR2================

end %th1

%%
lineWid = 2;
figure
subplot(2,1,1)
plot(th1Aogr1,10*log10(testMSEAogr1),'LineWidth', lineWid)

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('SC Threshold (T_2)'),ylabel('Testing MSE (dB)')
grid on

subplot(2,1,2)
plot(th1Aogr1,dicSizeAogr1,'LineWidth', lineWid)

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');

xlabel('SC Threshold (T_2)'),ylabel('# centers')
grid on

%%
figure
subplot(2,1,1)
plot(th1Aogr2,10*log10(testMSEAogr2),'LineWidth', lineWid)

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('ALD Threshold (T_2)'),ylabel('Testing MSE (dB)')
grid on

subplot(2,1,2)
plot(th1Aogr2,dicSizeAogr2,'LineWidth', lineWid)

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');

xlabel('ALD Threshold (T_2)'),ylabel('# centers')
grid on
%
%%
figure
plot(dicSizeAogr1,10*log10(testMSEAogr1),'b-','LineWidth', lineWid)
hold on
plot(dicSizeAogr2,10*log10(testMSEAogr2),'r--','LineWidth', lineWid)
hold off
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('final network size'),ylabel('testing MSE (dB)')
grid on
legend('KRLS-SC','KRLS-ALD')
