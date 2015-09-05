%Copyright
%Weifeng Liu CNEL
%July 4 2008
%
%Description:
%Data analysis of CO2 data
%
%Usage:
%Ch5, CO2 concentration forecasting, figure 5-10
%
%ouside functions called
%ker_eval

close all, clear all
clc
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Data Formatting
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Input signal
load CO2_data.mat date ave_missing ave_interpolate

dataSize = length(date);

% Data size for training and testing
trainSize = ceil(dataSize*0.7);
testSize = dataSize - trainSize;

%Input training signal with data embedding
trainInput = date(1:trainSize)';

%Input test data with embedding
testInput = date(1+trainSize:end)';

%Desired training signal
trainTarget = ave_interpolate(1:trainSize);

testTarget =  ave_interpolate(1+trainSize:end);
%%
%Kernel parameters
typeKernel = 'CO2';
paramKernel = 0.2;

%%
numTh1 = 50;
dicSizeAogr1 = zeros(numTh1,1);
testMSEAogr1 = zeros(numTh1,1);
dicSizeScklms = zeros(numTh1,1);
testMSEScklms = zeros(numTh1,1);

th1Aogr1 = linspace(-1.5,3,numTh1);

th2Aogr1 = 10^7;


regularizationFactorAogr = 0;
forgettingFactorAogr = 1;
flagLearningCurve = 0;

stepSizeScklms = .01;
stepSizeWeightScklms = 0;
stepSizeBiasScklms = 0;
regularizationFactor = 0.001;
th1 = 10^9; 
    
disp(['Please wait...',num2str(numTh1),' times loops'])
for ii = 1:numTh1
	disp(ii)
	th1Aogr1_ii = th1Aogr1(ii);
	
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

%     	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	%
% 	%              SCKLMS (no input apriori)
% 	%
% 	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% 	th2 = th1Aogr1_ii;
%     
% 	[expansionCoefficientScklms,weightVectorScklms,biasTermScklms,learningCurveScklms,dictIndexScklms] = ...
% 		SCKLMS(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeScklms,stepSizeWeightScklms,stepSizeBiasScklms,th1,th2,regularizationFactor,flagLearningCurve);
% 
%     
% 	y_te = zeros(testSize,1);
% 	for jj = 1:testSize
% 		 y_te(jj) = expansionCoefficientScklms*...
% 			ker_eval(testInput(:,jj),trainInput(:,dictIndexScklms),typeKernel,paramKernel);
% 	end
% 	testMSEScklms(ii) = mean((testTarget - y_te).^2);
% 	dicSizeScklms(ii) = length(dictIndexScklms);

	% =========end of AOGR1================
    
end %th1

%%
lineWid = 2;
figure
subplot(2,1,1)
plot(th1Aogr1,10*log10(testMSEAogr1),'LineWidth', lineWid)

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
ylabel('testing MSE (dB)')
grid on

subplot(2,1,2)
plot(th1Aogr1,dicSizeAogr1,'LineWidth', lineWid)

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');

xlabel('redundancy threshold of SC-KRLS (T_2)'),ylabel('network size')
grid on

% figure
% subplot(2,1,1)
% plot(th1Aogr1,10*log10(testMSEScklms),'LineWidth', lineWid)
% 
% set(gca, 'FontSize', 14);
% set(gca, 'FontName', 'Arial');
% ylabel('testing MSE (dB)')
% grid on
% 
% subplot(2,1,2)
% plot(th1Aogr1,dicSizeScklms,'LineWidth', lineWid)
% 
% set(gca, 'FontSize', 14);
% set(gca, 'FontName', 'Arial');
% 
% xlabel('redundancy threshold of SC-KLMS (T_2)'),ylabel('network size')
% grid on

%%
figure
plot(dicSizeAogr1,10*log10(testMSEAogr1),'b-','LineWidth', lineWid)
hold on
%plot(dicSizeScklms,10*log10(testMSEScklms),'r.','LineWidth', lineWid)
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('final network size'),ylabel('testing MSE (dB)')
grid on
%legend('SC-KRLS','SC-KLMS')