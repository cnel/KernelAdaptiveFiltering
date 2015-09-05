%Copyright
%Weifeng Liu CNEL
%July 4 2008
%
%Description:
%Compare SKLMS1, resource allocating network and AOGR in m-g time series prediction
%
%Usage:
%Ch5, m-g time series, figures
%
%ouside functions called
%sparseKLMS1, RAN and AOGR

close all, clear all
clc

%% Data Formatting
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Data Formatting
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load MK30   %MK30 5000*1

varNoise = 0.004;
inputDimension = 7; 

% Data size for training and testing
trainSize = 500;
testSize = 50;

MC = 50;
disp([num2str(MC),' monte carlo simulations'])

netSizeRan = zeros(MC,1);
netSizeSklms1 = zeros(MC,1);
netSizeAogr = zeros(MC,1);
netSizeScklms = zeros(MC,1);

enLearningCurveRan = zeros(trainSize,1);
enLearningCurveSklms1 = zeros(trainSize,1);
enLearningCurveScklms = zeros(trainSize,1);
enLearningCurveAogr = zeros(trainSize,1);
enLearningCurveLms1 = zeros(trainSize,1);

for mc = 1:MC
	disp(mc);

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

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	%             LMS 1
	%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	stepSizeWeightLms1 = .01;
	stepSizeBiasLms1 = .01;
	flagLearningCurve = 1;
	[aprioriErrLms1,weightVectorLms1,biasTermLms1,learningCurveLms1]= ...
		LMS1(trainInput,trainTarget,testInput,testTarget,stepSizeWeightLms1,stepSizeBiasLms1,flagLearningCurve);
	enLearningCurveLms1 = enLearningCurveLms1 + learningCurveLms1;
	
	%=========end of Linear LMS 1================
	%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	%        sparse KLMS1
	%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%=========sparseKLMS1===================
	stepSizeSklms1 = .5;
	stepSizeWeightSklms1 = 0;
	stepSizeBiasSklms1 = 0;
	flagLearningCurve = 1;
	tolDistance = 0.01;
	tolError = 0.1;

	[expansionCoefficientSklms1,weightVectorSklms1,biasTermSklms1,learningCurveSklms1,dictIndexSklms1] = ...
		sparseKLMS1(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeSklms1,stepSizeWeightSklms1,stepSizeBiasSklms1,tolDistance,tolError,flagLearningCurve);
	
	netSizeSklms1(mc) = length(expansionCoefficientSklms1);
	enLearningCurveSklms1 = enLearningCurveSklms1 + learningCurveSklms1;

	%=========end of sparse KLMS 1================


	%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	%        SC-KLMS
	%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%=========sparseKLMS1===================
	stepSizeScklms = .5;
	stepSizeWeightScklms = 0;
	stepSizeBiasScklms = 0;
	flagLearningCurve = 1;
	regularizationFactor = 0.01;
	th1 = 200; 
	th2 = -1;
    
	[expansionCoefficientScklms,weightVectorScklms,biasTermScklms,learningCurveScklms,dictIndexScklms] = ...
		SCKLMS(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeScklms,stepSizeWeightScklms,stepSizeBiasScklms,th1,th2,regularizationFactor,flagLearningCurve);
	
	netSizeScklms(mc) = length(expansionCoefficientScklms);
	enLearningCurveScklms = enLearningCurveScklms + learningCurveScklms;

	%=========end of SC-KLMS ================
    
	%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	%        RAN
	%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	stepSizeRan = .1;
	flagLearningCurve = 1;

	tolError = 0.05;
	delta_max = 0.5;
	delta_min = 0.05;
	tau = 45;
	overlapFactor = 0.87;

	[expansionCoefficientRan,centerSetRan, biasTermRan,learningCurveRan] = ...
		RAN(trainInput,trainTarget,testInput,testTarget,delta_max, delta_min,...
		tau, overlapFactor, stepSizeRan,tolError,flagLearningCurve);

	netSizeRan(mc) = length(expansionCoefficientRan);
	enLearningCurveRan = enLearningCurveRan + learningCurveRan;
	%=========end of RAN================
	%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	%        KRLS-CI
	%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	regularizationFactor = 0.01;
	th1 = -1; 
	th2 = 200;
	[expansionCoefficientCi,dictionaryIndexCi,learningCurveCi,CI] = ...
		AOGR(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactor,1,2,th1,th2,[],flagLearningCurve);

	netSizeAogr(mc) = length(expansionCoefficientCi);
	enLearningCurveAogr = learningCurveCi + learningCurveCi;
	
%%%%%%%%%end of AOGR====================
end
%%
lineWid = 2;
figure(10),

plot(enLearningCurveLms1/MC,'r-.','LineWidth', 0.5)
hold on

plot(enLearningCurveRan/MC,'b--','LineWidth', lineWid)
plot(enLearningCurveSklms1/MC,'b:','LineWidth', lineWid)
plot(enLearningCurveScklms/MC,'g-','LineWidth', lineWid)
plot(enLearningCurveAogr/MC,'g-.','LineWidth', lineWid)

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('iteration'),ylabel('testing MSE')


legend('LMS','RAN','KLMS-NC','KLMS-SC','KRLS-SC')

hold off
xlabel('iteration'),ylabel('testing MSE')

set(gca,'YScale','log')

%%
disp('====================================')

disp('>>RAN')
mseMean = mean(netSizeRan);
mseStd = std(netSizeRan);

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('>>NC-KLMS')
mseMean = mean(netSizeSklms1);
mseStd = std(netSizeSklms1);
disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('>>SC-KLMS')
mseMean = mean(netSizeScklms);
mseStd = std(netSizeScklms);
disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('>>AOGR')
mseMean = mean(netSizeAogr);
mseStd = std(netSizeAogr);

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('====================================')