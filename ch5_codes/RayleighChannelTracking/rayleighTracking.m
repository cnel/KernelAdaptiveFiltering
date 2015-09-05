% rayleigh channel tracking
% Weifeng Liu 
% Sep. 2007.
%
% Description:
% compare extended KRLS and KRLS in rayleigh channel tracking
% learning curves
%
% Usage
% Ch 4, figure 4-1
%
% Outside functions used
% LMS2, RLS, EX_RLS, KRLS and EX_KRLS


close all
clear all
clc

var_n = 1e-3;
sqn = sqrt(var_n);

sampleInterval = 0.8e-6; % sampling frequency is 1.25MHz
numberSymbol = 2000;
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


ensembleLearningCurveLms2 = zeros(trainSize,1);
ensembleLearningCurveRls = zeros(trainSize,1);
ensembleLearningCurveExrls = zeros(trainSize,1);
ensembleLearningCurveKrls = zeros(trainSize,1);
ensembleLearningCurveExkrls = zeros(trainSize,1);
% time delay (embedding) length
inputDimension = channelLength;

%Nonlinearity parameter
typeNonlinear = 1;
paramNonlinear = 2;

%Kernel parameters
typeKernel = 'Gauss';
paramKernel = .1;


L = 500;

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
    %               Normalized LMS 2
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    regularizationFactorLms2 = epsilon;
    stepSizeWeightLms2 = .25;
    stepSizeBiasLms2 = 0;

    [weightVectorLms2,biasTermLms2,learningCurveLms2]= ...
        LMS2(trainInput,trainTarget,regularizationFactorLms2,stepSizeWeightLms2,stepSizeBiasLms2);

    %=========end of Normalized LMS 2================

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %               RLS
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    pInitialRls = (1/epsilon)*eye(inputDimension);
    forgettingFactorRls = 1;

    [weightVectorRls,learningCurveRls]= ...
        RLS(trainInput,trainTarget,pInitialRls,forgettingFactorRls);

    %=========end of RLS================
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %             Ex-RLS
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    pInitialExrls = (1/epsilon)*eye(inputDimension);
    forgettingFactorExrls = .995;
    alphaParameterExrls = alpha;
    qFactorExrls = q;
    % flagLearningCurve = 1;
    [weightVectorExrls,learningCurveExrls]= ...
        EX_RLS(trainInput,trainTarget,pInitialExrls,forgettingFactorExrls,alphaParameterExrls,qFactorExrls);
    %=========end of ex-rls================

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %              KRLS
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    regularizationFactorKrls = 0.1;
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
    alphaParameterExkrls = 0.999998;
    regularizationFactorExkrls = 0.1;
    forgettingFactorExkrls = .995;
    qFactorExkrls = 0.0001;
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

ensembleLearningCurveLms2 = 10*log10(ensembleLearningCurveLms2/L);
ensembleLearningCurveRls = 10*log10(ensembleLearningCurveRls/L);
ensembleLearningCurveExrls = 10*log10(ensembleLearningCurveExrls/L);
ensembleLearningCurveKrls = 10*log10(ensembleLearningCurveKrls/L);
ensembleLearningCurveExkrls = 10*log10(ensembleLearningCurveExkrls/L);

mse_Lms2 = mean(ensembleLearningCurveLms2(end-100:end));
mse_Rls = mean(ensembleLearningCurveRls(end-100:end));
mse_Exrls = mean(ensembleLearningCurveExrls(end-100:end));
mse_Krls = mean(ensembleLearningCurveKrls(end-100:end));
mse_Exkrls = mean(ensembleLearningCurveExkrls(end-100:end));
disp([num2str(mse_Lms2),' ',num2str(mse_Rls),' ',num2str(mse_Exrls),' ',num2str(mse_Krls),' ',num2str(mse_Exkrls),' '])

plot(1:trainSize,ensembleLearningCurveLms2,'b-.',1:trainSize,ensembleLearningCurveExrls,'c-.',1:trainSize,ensembleLearningCurveKrls,'g-',1:trainSize,ensembleLearningCurveExkrls,'r-');

title('MSE')
xlabel('iteration')
ylabel('dB')
grid
axis tight
legend('LMS2','ExRLS','KRLS','ExKRLS')

disp('====================================')

mseMean = mean(ensembleLearningCurveLms2(end-100:end));
mseStd = std(ensembleLearningCurveLms2(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);
% 
mseMean = mean(ensembleLearningCurveRls(end-100:end));
mseStd = std(ensembleLearningCurveRls(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

mseMean = mean(ensembleLearningCurveExrls(end-100:end));
mseStd = std(ensembleLearningCurveExrls(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

mseMean = mean(ensembleLearningCurveKrls(end-100:end));
mseStd = std(ensembleLearningCurveKrls(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

mseMean = mean(ensembleLearningCurveExkrls(end-100:end));
mseStd = std(ensembleLearningCurveExkrls(end-100:end));

disp([num2str(mseMean),'+/-',num2str(mseStd)]);
disp('====================================')

