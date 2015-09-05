%Copyright
%Weifeng Liu CNEL
%July 4 2008
%
%Description:
%Evaluation surprise in nonlinear regression
%ALD and surprise for redundency removal
%
%Usage:
%Ch5, nonlinear regression, redundancy removal, figures 5-2, 5-3, 5-4, 5-5
%
%ouside functions called
%AOGR1 and AOGR2, nlG
close all, clear all
clc

%% Data Formatting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Data Formatting
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Synthesize the linear filter
linearFilter = 1;

% time delay (embedding) length
inputDimension = length(linearFilter);

%Nonlinearity parameter
typeNonlinear = 2;
paramNonlinear = 2;

%Kernel parameters
typeKernel = 'Gauss';
paramKernel = 0.2;


% Data size for training and testing
trainSize = 200;
testSize = 100;

flagLearningCurve = 0;
numTh1 = 50;

testMSEAogr1_mc = zeros(numTh1,1);
dicSizeAogr1_mc = zeros(numTh1,1);
testMSEAogr2_mc = zeros(numTh1,1);
dicSizeAogr2_mc = zeros(numTh1,1);

MC = 100;

disp(['Please wait...', num2str(MC),' Monte Carlo simultions']);
for mc = 1:MC
	disp(mc)
	% Input signal
	inputSignal = randn(500,1);
	% figure(1),plot(inputSignal)

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

	%Desired training signal
	trainTarget = zeros(trainSize,1);
	for ii=1:trainSize
		trainTarget(ii) = inputSignal(ii:ii+inputDimension-1)'*linearFilter;
	end
	%Pass through the nonlinearity
	trainTarget = nlG(trainTarget,paramNonlinear,typeNonlinear);

	%Desired training signal
	testTarget = zeros(testSize,1);
	for ii=1:testSize
		testTarget(ii) = inputSignal(ii+trainSize:ii+inputDimension-1+trainSize)'*linearFilter;
	end
	%Pass through the nonlinearity
	testTarget = nlG(testTarget,paramNonlinear,typeNonlinear);

	%%

	dicSizeAogr1 = zeros(numTh1,1);
	testMSEAogr1 = zeros(numTh1,1);
	dicSizeAogr2 = zeros(numTh1,1);
	testMSEAogr2 = zeros(numTh1,1);

	th1Aogr1 = linspace(-5,5,numTh1);
	th1Aogr2 = linspace(-3.8,0.1,numTh1);
	th2Aogr1 = 10^7;
	th2Aogr2 = 10^7;

	regularizationFactorAogr = 0.001;
	forgettingFactorAogr = 1;

	for ii = 1:numTh1
		disp([num2str(mc), num2str(ii)])
		
		th1Aogr1_ii = th1Aogr1(ii);
		th1Aogr2_ii = th1Aogr2(ii);
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%
		%              AOGR 1
		%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


		[expansionCoefficientAogr1,dictionaryIndexAogr1,learningCurveAogr1,CI_Aogr1] = ...
			AOGR1(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactorAogr,forgettingFactorAogr,th1Aogr1_ii,th2Aogr1,flagLearningCurve);

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
		%              AOGR 2
		%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


		[expansionCoefficientAogr2,dictionaryIndexAogr2,learningCurveAogr2,CI_Aogr2] = ...
			AOGR2(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactorAogr,forgettingFactorAogr,th1Aogr2_ii,th2Aogr2,flagLearningCurve);

		y_te = zeros(testSize,1);
		for jj = 1:testSize
			 y_te(jj) = expansionCoefficientAogr2*...
				ker_eval(testInput(:,jj),trainInput(:,dictionaryIndexAogr2),typeKernel,paramKernel);
		end
		testMSEAogr2(ii) = mean((testTarget - y_te).^2);
		dicSizeAogr2(ii) = length(dictionaryIndexAogr2);


		% =========end of AOGR2================

	end %th1

	testMSEAogr1_mc = testMSEAogr1_mc + testMSEAogr1;
	dicSizeAogr1_mc = dicSizeAogr1_mc + dicSizeAogr1;

	testMSEAogr2_mc = testMSEAogr2_mc + testMSEAogr2;
	dicSizeAogr2_mc = dicSizeAogr2_mc + dicSizeAogr2;


end %mc
%%
lineWid = 2;
figure
subplot(2,1,1)
plot(th1Aogr1,testMSEAogr1_mc/MC,'LineWidth', lineWid)

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
%xlabel('redundancy threshold (T_2)'),
ylabel('testing MSE')
grid on

subplot(2,1,2)
plot(th1Aogr1,dicSizeAogr1_mc/MC,'LineWidth', lineWid)

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');

xlabel('T_2 in KRLS-SC'),ylabel('final network size')
grid on

%%
figure
subplot(2,1,1)
plot(th1Aogr2,testMSEAogr2_mc/MC,'LineWidth', lineWid)

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
ylabel('testing MSE')
grid on

subplot(2,1,2)
plot(th1Aogr2,dicSizeAogr2_mc/MC,'LineWidth', lineWid)

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');

xlabel('T_2 in KRLS-SC'),ylabel('final network size')
grid on
%
%%
figure
plot(dicSizeAogr1_mc/MC,testMSEAogr1_mc/MC,'b-','LineWidth', lineWid)
hold on
plot(dicSizeAogr2_mc/MC,testMSEAogr2_mc/MC,'r--','LineWidth', lineWid)
hold off
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('final network size'),ylabel('testing MSE')
grid on
legend('KRLS-SC','KRLS-ALD')
