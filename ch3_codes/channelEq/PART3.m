%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Copyright Weifeng Liu 
%CNEL
%July 1, 2008
%
%description:
%compare the performance of LMS1 APA1 SKLMS1 SKAPA1 SKAPA2 in channel
%equalization
%Abrupt change and reconvergence
%
%Usage:
%ch3, channel equalization, figures 3-9, 3-10
%
%Outside functions called or toolboxes used:
%LMS1s, APA1s, sparseKLMS1s, sparseKAPA1s, sparseKAPA2s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all,
close all
clc

%======filter config=======
%time delay (embedding) length
inputDimension = 3;
equalizationDelay = 0;
%Kernel parameters
typeKernel = 'Gauss';
paramKernel = .1;

%======end of config=======
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Parameters
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

stepSizeWeightLms1 = .02;
stepSizeBiasLms1 = .02;
%paramRegularizationLms1 = 5;

flagLearningCurve = 1;

K = 10;
stepSizeWeightApa1 = .02/K;
stepSizeBiasApa1 = .002;

stepSizeKapa1 = .2/K;
stepSizeWeightKapa1 = 0;
stepSizeBiasKapa1 = 0;

toleranceDistance = 0.05;
tolerancePredictError = 0.01;

stepSizeKapa2 = .5/K;
stepSizeWeightKapa2 = 0;
stepSizeBiasKapa2 = 0;
paramRegularizationKapa2 = 0.01;

stepSizeKlms1 = .15;
stepSizeWeightKlms1 = 0;
stepSizeBiasKlms1 = 0;

%data size
trainSize = 1500;
testSize = 100;

change_time = 500;
noise_std = 0.1;


%======end of data===========
%%    
MC = 100;
learningCurveLms1_en = zeros(trainSize,1);
learningCurveApa1_en = zeros(trainSize,1);
learningCurveKapa1_en = zeros(trainSize,1);
learningCurveKapa2_en = zeros(trainSize,1);
learningCurveKlms1_en = zeros(trainSize,1);

disp([num2str(MC), '  Monte Carlo simulations. Please wait...'])
for mc = 1:MC
    disp(mc)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	%       Data Formatting
	%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%=========data===============
	% Generate binary data
	u = sign(randn(1,trainSize*1.5));
	z = filter([1 0.5],1,u);
	% Channel noise
	ns = noise_std*randn(1,length(z));
	% Ouput of the nonlinear channel
	y = z - 0.9*z.^2 + ns;
    y(change_time:end) = 0.9*z(change_time:end).^2 - z(change_time:end) + ns(change_time:end);

	%data embedding
	trainInput = zeros(inputDimension,trainSize);
	for k=1:trainSize
		trainInput(:,k) = y(k:k+inputDimension-1)';
	end
	% Desired signal
	trainTarget = zeros(trainSize,1);
	for ii=1:trainSize
		trainTarget(ii) = u(equalizationDelay+ii);
	end % Generate binary data
	
	
    %======end of data===========

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %               LMS 1
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [aprioriErrLms1,weightVectorLms1,biasTermLms1,learningCurveLms1]= ...
        LMS1s(trainInput,trainTarget,stepSizeWeightLms1,stepSizeBiasLms1,flagLearningCurve);
    learningCurveLms1_en = learningCurveLms1_en +learningCurveLms1;
    %=========end of LMS 1================

    %%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %             APA 1
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [weightVectorApa1,biasTermApa1,learningCurveApa1]= ...
        APA1s(K,trainInput,trainTarget,stepSizeWeightApa1,stepSizeBiasApa1,flagLearningCurve);
    learningCurveApa1_en = learningCurveApa1_en + learningCurveApa1;
    %=========end of Linear APA 1================

    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %        sparse KAPA 1
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [expansionCoeffKapa1,dictionaryIndexKapa1,weightVectorKapa1,biasTermKapa1,learningCurveKapa1,netSizeKapa1] = ...
        sparseKAPA1s(K,trainInput,trainTarget,typeKernel,paramKernel,stepSizeKapa1,stepSizeWeightKapa1,stepSizeBiasKapa1,toleranceDistance,tolerancePredictError,flagLearningCurve);
    learningCurveKapa1_en = learningCurveKapa1_en + learningCurveKapa1;
    %=========end of KAPA 1================

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %        sparse KAPA 2
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [expansionCoeffKapa2,dictionaryIndexKapa2,weightVectorKapa2,biasTermKapa2,learningCurveKapa2,netSizeKapa2] = ...
        sparseKAPA2s(K,trainInput,trainTarget,paramRegularizationKapa2,typeKernel,paramKernel,stepSizeKapa2,stepSizeWeightKapa2,stepSizeBiasKapa2,toleranceDistance,tolerancePredictError,flagLearningCurve);
    learningCurveKapa2_en = learningCurveKapa2_en + learningCurveKapa2;
    %=========end of KAPA 1================
    %%

    %=========sparse Kernel LMS 1===================

    [expansionCoefficientKlms1,weightVectorKlms1,biasTermKlms1,learningCurveKlms1,dictionaryIndexKlms1,netSizeKlms1] = ...
        sparseKLMS1s(trainInput,trainTarget,typeKernel,paramKernel,...
        stepSizeKlms1,stepSizeWeightKlms1,stepSizeBiasKlms1,toleranceDistance,tolerancePredictError,flagLearningCurve);
    learningCurveKlms1_en = learningCurveKlms1_en + learningCurveKlms1;
    %=========end of sparse Kernel LMS================
    %%
end%mc

%%
lineWid = 3;
if flagLearningCurve
	figure,
%	plot(learningCurveLms1_en/MC,'c:','LineWidth', lineWid-2)
%	hold on
	plot(learningCurveApa1_en/MC,'c--','LineWidth', lineWid-2)
    hold on
	plot(learningCurveKlms1_en/MC,'b-','LineWidth', lineWid)
%	plot(learningCurveKapa1_en/MC,'g:','LineWidth', lineWid)
	plot(learningCurveKapa2_en/MC,'r-.','LineWidth', lineWid)
	hold off
	legend('APA-1','KLMS-NC','KAPA-2-NC')
    set(gca, 'FontSize', 14);
    set(gca, 'FontName', 'Arial');
    xlabel('iteration'),ylabel('MSE')
    grid on
end

figure,
plot(netSizeKlms1,'b-','LineWidth', lineWid)
hold on
%plot(netSizeKapa1,'g:','LineWidth', lineWid)
plot(netSizeKapa2,'r--','LineWidth', lineWid)
hold off
legend('KLMS-NC','KAPA-2-NC')
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('iteration'),ylabel('network size')
grid on
