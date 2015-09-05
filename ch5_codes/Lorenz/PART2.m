% Lorenz series modeling
% Weifeng Liu 
% Jul. 2008.
%
% Description:
% compare extended KRLS and KRLS in Lorenz signal modeling
% learning curves
%
% Usage
% Ch 4
%
% Outside functions used
% LMS2, RLS, EX_RLS, KRLS and EX_KRLS


close all
clear all
clc

load lorenz.mat
%lorenz2 50000*1 double
lorenz2 = lorenz2 - mean(lorenz2);
lorenz2 = lorenz2/std(lorenz2);

trainSize = 1000;
inputDimension = 5;
predictionHorizon = 10;

%Kernel parameters
typeKernel = 'Gauss';
paramKernel = 1;


L = 200;

disp([num2str(L),' Monte Carlo simulations. Please wait...'])

ensembleLearningCurveLms2 = zeros(trainSize,1);
ensembleLearningCurveRls = zeros(trainSize,1);
ensembleLearningCurveExrls = zeros(trainSize,1);
ensembleLearningCurveKrls = zeros(trainSize,1);
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
    %               Normalized LMS 2
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    regularizationFactorLms2 = 0.001;
    stepSizeWeightLms2 = .2;
    stepSizeBiasLms2 = 0;

    [weightVectorLms2,biasTermLms2,learningCurveLms2]= ...
        LMS2(trainInput,trainTarget,regularizationFactorLms2,stepSizeWeightLms2,stepSizeBiasLms2);

    %=========end of Normalized LMS 2================

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %               RLS
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	epsilon = 1e-3;
    pInitialRls = (1/epsilon)*eye(inputDimension);
    forgettingFactorRls = 1;

    [weightVectorRls,learningCurveRls]= ...
        RLS(trainInput,trainTarget,pInitialRls,forgettingFactorRls);

    %%=========end of RLS================
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %             Ex-RLS
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	epsilon = 1e-3;
    pInitialExrls = (1/epsilon)*eye(inputDimension);
    forgettingFactorExrls = 0.99;
    alphaParameterExrls = 1;
    qFactorExrls = 0.01;
    % flagLearningCurve = 1;
    [weightVectorExrls,learningCurveExrls]= ...
        EX_RLS(trainInput,trainTarget,pInitialExrls,forgettingFactorExrls,alphaParameterExrls,qFactorExrls);
    %=========end of ex-rls================

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %              KRLS
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    regularizationFactorKrls = 0.001;
    forgettingFactorKrls = 1;

    % flagLearningCurve = 1;
    [expansionCoefficientKrls,learningCurveKrls] = ...
        KRLS(trainInput,trainTarget,typeKernel,paramKernel,regularizationFactorKrls,forgettingFactorKrls);

    % =========end of KRLS================
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

    ensembleLearningCurveLms2 = ensembleLearningCurveLms2 + learningCurveLms2;
    ensembleLearningCurveRls = ensembleLearningCurveRls + learningCurveRls;
    ensembleLearningCurveExrls = ensembleLearningCurveExrls + learningCurveExrls;
    
    ensembleLearningCurveKrls = ensembleLearningCurveKrls + learningCurveKrls;
    ensembleLearningCurveExkrls = ensembleLearningCurveExkrls + learningCurveExkrls;
 
end

signalPower = std(inputSignal)^2;

%%
% figure
% % plot(1:trainSize,ensembleLearningCurveLms2/L,'b:',1:trainSize,ensembleLearningCurveExrls/L,'c-.',1:trainSize,ensembleLearningCurveKrls/L,'g--',1:trainSize,ensembleLearningCurveExkrls/L,'r-');
% 
% xlabel('iteration')
% ylabel('dB')
% grid
% axis tight
% legend('LMS2','ExRLS','KRLS','ExKRLS')
% 
% set(gca, 'FontSize', 14);
% set(gca, 'FontName', 'Arial');
% %legend('Conditional Information')
% xlabel('iteration'),ylabel('testing MSE (dB)')

%%
ensembleLearningCurveLms2_dB = 10*log10(ensembleLearningCurveLms2/L/signalPower);
ensembleLearningCurveRls_dB = 10*log10(ensembleLearningCurveRls/L/signalPower);
ensembleLearningCurveExrls_dB = 10*log10(ensembleLearningCurveExrls/L/signalPower);
ensembleLearningCurveKrls_dB = 10*log10(ensembleLearningCurveKrls/L/signalPower);
ensembleLearningCurveExkrls_dB = 10*log10(ensembleLearningCurveExkrls/L/signalPower);

mse_Lms2 = mean(ensembleLearningCurveLms2_dB(end-100:end));
mse_Rls = mean(ensembleLearningCurveRls_dB(end-100:end));
mse_Exrls = mean(ensembleLearningCurveExrls_dB(end-100:end));
mse_Krls = mean(ensembleLearningCurveKrls_dB(end-100:end));
mse_Exkrls = mean(ensembleLearningCurveExkrls_dB(end-100:end));
disp([num2str(mse_Lms2),' ',num2str(mse_Rls),' ',num2str(mse_Exrls),' ',num2str(mse_Krls),' ',num2str(mse_Exkrls),' '])

figure
plot(1:trainSize,ensembleLearningCurveLms2_dB,'b:',1:trainSize,ensembleLearningCurveExrls_dB,'c-.',1:trainSize,ensembleLearningCurveKrls_dB,'g--',1:trainSize,ensembleLearningCurveExkrls_dB,'r-');

xlabel('iteration')
ylabel('dB')
grid
axis tight
legend('LMS2','ExRLS','KRLS','ExKRLS')

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
%legend('Conditional Information')
xlabel('iteration'),ylabel('MSE (dB)')

%%

disp('====================================')

mseMean = mean(ensembleLearningCurveLms2_dB(end-100:end));
mseStd = std(ensembleLearningCurveLms2_dB(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);
% 
mseMean = mean(ensembleLearningCurveRls_dB(end-100:end));
mseStd = std(ensembleLearningCurveRls_dB(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

mseMean = mean(ensembleLearningCurveExrls_dB(end-100:end));
mseStd = std(ensembleLearningCurveExrls_dB(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

mseMean = mean(ensembleLearningCurveKrls_dB(end-100:end));
mseStd = std(ensembleLearningCurveKrls_dB(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

mseMean = mean(ensembleLearningCurveExkrls_dB(end-100:end));
mseStd = std(ensembleLearningCurveExkrls_dB(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);
disp('====================================')

% figure,plot(inputSignal)