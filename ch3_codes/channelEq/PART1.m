%Copyright
%Weifeng Liu, CNEL
%July 4, 2008
%
%Description:
%compare LMS2, APA1, SKLMS1, SKAPA1 and SKAPA2 in nonlinear channel equalization
%learning curves, bit error rate and network size
%
%Usage:
%Ch3, nonlinear channel equalization, figures 3-6, 3-7
%
%outside function called
%LMS1, APA1, SKLMS1, SKAPA1, SKAPA2

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

toleranceDistance = 0.26;
tolerancePredictError = 0.08;

stepSizeKapa2 = .5/K;
stepSizeWeightKapa2 = 0;
stepSizeBiasKapa2 = 0;
paramRegularizationKapa2 = 0.01;

stepSizeKlms1 = .15;
stepSizeWeightKlms1 = 0;
stepSizeBiasKlms1 = 0;

%data size
trainSize = 10000;
testSize = 100;

noise_std = 0.1;


%======end of data===========
%%    
MC = 50;
errNumLms1 = zeros(MC,1);
errNumApa1 = zeros(MC,1);
errNumSkapa1 = zeros(MC,1);
errNumSkapa2 = zeros(MC,1);
errNumSklms1 = zeros(MC,1);

ensembleLearningCurveLms1 = zeros(trainSize,1);
ensembleLearningCurveApa1 = zeros(trainSize,1);
ensembleLearningCurveKlms1 = zeros(trainSize,1);
ensembleLearningCurveKapa1 = zeros(trainSize,1);
ensembleLearningCurveKapa2 = zeros(trainSize,1);

disp([num2str(MC),' monte carlo simulation.'])

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
	
	
    u = sign(randn(1,testSize*2));
    z = filter([1 0.5],1,u);
    % Channel noise
    ns = noise_std*randn(1,length(z));
    % Ouput of the nonlinear channel
    y = z - 0.9*z.^2 + ns;

    %data embedding
    
    % Test data
    testInput = zeros(inputDimension,testSize);
    for k=1:testSize
        testInput(:,k) = y(k:k+inputDimension-1)';
    end

    testTarget = zeros(testSize,1);
    for ii=1:testSize
        testTarget(ii) = u(equalizationDelay+ii);
    end
    %======end of data===========

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %               LMS 1
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [aprioriErrLms1,weightVectorLms1,biasTermLms1,learningCurveLms1]= ...
        LMS1(trainInput,trainTarget,testInput,testTarget,stepSizeWeightLms1,stepSizeBiasLms1,flagLearningCurve);
    errNumLms1(mc) = sum(testTarget ~= sign(testInput'*weightVectorLms1 + biasTermLms1));
    
    ensembleLearningCurveLms1 = ensembleLearningCurveLms1 + learningCurveLms1;
    %=========end of LMS 1================

    %%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %             APA 1
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [weightVectorApa1,biasTermApa1,learningCurveApa1]= ...
        APA1(K,trainInput,trainTarget,testInput,testTarget,stepSizeWeightApa1,stepSizeBiasApa1,flagLearningCurve);
    errNumApa1(mc) = sum(testTarget ~= sign(testInput'*weightVectorApa1 + biasTermApa1));

    ensembleLearningCurveApa1 = ensembleLearningCurveApa1 + learningCurveApa1;
    %=========end of Linear APA 1================

    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %        sparse KAPA 1
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [expansionCoeffKapa1,dictionaryIndexKapa1,weightVectorKapa1,biasTermKapa1,learningCurveKapa1,netSizeKapa1] = ...
        sparseKAPA1(K,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeKapa1,stepSizeWeightKapa1,stepSizeBiasKapa1,toleranceDistance,tolerancePredictError,flagLearningCurve);
    y_te = zeros(testSize,1);
    for jj = 1:testSize
        y_te(jj) = expansionCoeffKapa1*...
            ker_eval(testInput(:,jj),trainInput(:,dictionaryIndexKapa1),typeKernel,paramKernel) + weightVectorKapa1'*testInput(:,jj) + biasTermKapa1;
    end
    errNumSkapa1(mc) = sum(testTarget ~= sign(y_te));
    
    ensembleLearningCurveKapa1 = ensembleLearningCurveKapa1 + learningCurveKapa1;
    %=========end of KAPA 1================

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %        sparse KAPA 2
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [expansionCoeffKapa2,dictionaryIndexKapa2,weightVectorKapa2,biasTermKapa2,learningCurveKapa2,netSizeKapa2] = ...
        sparseKAPA2(K,trainInput,trainTarget,testInput,testTarget,paramRegularizationKapa2,typeKernel,paramKernel,stepSizeKapa2,stepSizeWeightKapa2,stepSizeBiasKapa2,toleranceDistance,tolerancePredictError,flagLearningCurve);
%     y_te = zeros(testSize,1);
    for jj = 1:testSize
        y_te(jj) = expansionCoeffKapa2*...
            ker_eval(testInput(:,jj),trainInput(:,dictionaryIndexKapa2),typeKernel,paramKernel) + weightVectorKapa2'*testInput(:,jj) + biasTermKapa2;
    end
    errNumSkapa2(mc) = sum(testTarget ~= sign(y_te));
    
    ensembleLearningCurveKapa2 = ensembleLearningCurveKapa2 + learningCurveKapa2;

    %=========end of KAPA 1================


    %%

    %=========sparse Kernel LMS 1===================

    [expansionCoefficientKlms1,weightVectorKlms1,biasTermKlms1,learningCurveKlms1,dictionaryIndexKlms1,netSizeKlms1] = ...
        sparseKLMS1(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,...
        stepSizeKlms1,stepSizeWeightKlms1,stepSizeBiasKlms1,toleranceDistance,tolerancePredictError,flagLearningCurve);
%     y_te = zeros(testSize,1);
    for jj = 1:testSize
        y_te(jj) = expansionCoefficientKlms1*...
            ker_eval(testInput(:,jj),trainInput(:,dictionaryIndexKlms1),typeKernel,paramKernel) + weightVectorKlms1'*testInput(:,jj) + biasTermKlms1;
    end
    errNumSklms1(mc) = sum(testTarget ~= sign(y_te));
    
    ensembleLearningCurveKlms1 = ensembleLearningCurveKlms1 + learningCurveKlms1;

    %=========end of sparse Kernel LMS================
        
    %%
end%mc

%%

disp('====================================')

mseMean = mean(errNumLms1);
mseStd = std(errNumLms1);

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

mseMean = mean(errNumApa1);
mseStd = std(errNumApa1);

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

mseMean = mean(errNumSklms1);
mseStd = std(errNumSklms1);
disp([num2str(mseMean),'+/-',num2str(mseStd)]);


mseMean = mean(errNumSkapa1);
mseStd = std(errNumSkapa1);

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

mseMean = mean(errNumSkapa2);
mseStd = std(errNumSkapa2);

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('====================================')
%%
lineWid = 3;
if flagLearningCurve
	figure,
	plot(ensembleLearningCurveLms1/MC,'k-','LineWidth', lineWid-2)
	hold on
	plot(ensembleLearningCurveApa1/MC,'k-.','LineWidth', lineWid-2)
	plot(ensembleLearningCurveKlms1/MC,'k:','LineWidth', lineWid)
	plot(ensembleLearningCurveKapa1/MC,'k--','LineWidth', lineWid)
	plot(ensembleLearningCurveKapa2/MC,'k-','LineWidth', lineWid)
	hold off
	legend('LMS','APA-1','KLMS-NC','KAPA-1-NC','KAPA-2-NC')
    set(gca, 'FontSize', 14);
    set(gca, 'FontName', 'Arial');
    xlabel('iteration'),ylabel('MSE')
end



figure,
plot(netSizeKlms1,'k:','LineWidth', lineWid-2)
hold on
plot(netSizeKapa1,'k-','LineWidth', lineWid-1)
plot(netSizeKapa2,'k--','LineWidth', lineWid)
hold off
legend('KLMS-NC','KAPA-1-NC','KAPA-2-NC')
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('iteration'),ylabel('network size')
grid on
