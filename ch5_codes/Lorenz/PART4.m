% Lorenz series modeling
% Weifeng Liu 
% Jul. 2008.
%
% Description:
% compare SW-KRLS and EX-KRLS in Lorenz signal modeling
% window length
%
% Usage
% Ch 4
%
% Outside functions used
% SW-KRLS


close all
clear all
clc

load lorenz.mat
%lorenz2 50000*1 double
lorenz2 = lorenz2 - mean(lorenz2);
lorenz2 = lorenz2/std(lorenz2);

trainSize = 500;
inputDimension = 5;
K_v = 10:20:400;
predictionHorizon = 10;

%Kernel parameters
typeKernel = 'Gauss';
paramKernel = 1;

L = 20;


for h=1:length(K_v)
	
	
	K = K_v(h);
    disp([num2str(K),' -> K'])
	disp([num2str(L),' Monte Carlo simulations. Please wait...'])

	ensembleLearningCurveSwkrls = zeros(trainSize,1);
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
		%              SW-KRLS
		%
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		regularizationFactorSwkrls = 0.0001;
		
        [expansionCoefficientSwkrls,learningCurveSwkrls] = ...
            SWKRLS(K,trainInput,trainTarget,typeKernel,paramKernel,regularizationFactorSwkrls);

		% =========end of SWKRLS================
		ensembleLearningCurveSwkrls = ensembleLearningCurveSwkrls + learningCurveSwkrls;


	end

	signalPower = std(inputSignal)^2;

	%%
	ensembleLearningCurveSwkrls_dB = 10*log10(ensembleLearningCurveSwkrls/L/signalPower);

	disp('====================================')

	mseMean = mean(ensembleLearningCurveSwkrls_dB(end-100:end));
	mseStd = std(ensembleLearningCurveSwkrls_dB(end-100:end));
	mseMeanSwkrls_h(h) = mseMean;
	mseStdSwkrls_h(h) = mseStd;
	disp([num2str(mseMean),'+/-',num2str(mseStd)]);
	
	disp('====================================')
end


%%
disp([num2str(L),' Monte Carlo simulations for EX-KRLS. Please wait...'])

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

mseMean = mean(ensembleLearningCurveExkrls_dB(end-100:end));
mseStd = std(ensembleLearningCurveExkrls_dB(end-100:end));
disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('====================================')

%%

figure
plot(K_v,mseMeanSwkrls_h,'k--','lineWidth',3);
hold on
plot(K_v,mseMean*ones(size(K_v)),'r-','lineWidth',3)

xlabel('iteration')
ylabel('dB')
grid
% axis tight

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
legend('SW-KRLS','EX-KRLS')
xlabel('window length'),ylabel('MSE (dB)')