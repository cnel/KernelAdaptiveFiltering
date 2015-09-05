%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Copyright Aaron Liu 
%CNEL
%July 1, 2008
%
%description:
%evaluate the performance of Leaky KLMS for Mackey Glass time series
%one step prediction
%Monte Carlo simulation with different regularization parameters
%
%Usage:
%ch2, m-g prediction
%
%Outside functions called:
%KLMS3 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all 
clear all 
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Data Formatting
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load MK30   %MK30 5000*1

MK30 = MK30(1000:5000);
varNoise = 0.001;
inputDimension = 10; 

% One step ahead prediction
predictionHorizon = 1;

%Kernel parameters
typeKernel = 'Gauss';

paramKernel = 1;

% Data size for training and testing
trainSize = 500;
testSize = 50;

regularizationParam_v = linspace(0,0.1,20);

mseMeanKlms3 = zeros(size(regularizationParam_v));
mseStdKlms3 = zeros(size(regularizationParam_v));

for kkk = 1:length(regularizationParam_v)

    %%
    stepSizeKlms3 = .1;

    paramRegularizationKlms3 = regularizationParam_v(kkk);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %       Monte Carlo
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    MC = 50;
    
    testMseKlms3 = zeros(MC,1);
    
    disp(['Regularization param = ',num2str(paramRegularizationKlms3)]);
    disp([num2str(MC), ' Monte Carlo simulations']);
    
    for mc = 1:MC

        disp(mc)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        %       data
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        inputSignal = MK30 + sqrt(varNoise)*randn(size(MK30));
        inputSignal = inputSignal - mean(inputSignal);

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


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        %               Leaky KLMS
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

          [expansionCoefficientKlms3,weightVectorKlms3,biasTermKlms3,learningCurveKlms3] = ...
              KLMS3(trainInput,trainTarget,[],[],typeKernel,paramKernel,paramRegularizationKlms3,stepSizeKlms3,0,0,0);
          y_te = zeros(testSize,1);

          for jj = 1:testSize

                y_te(jj) = expansionCoefficientKlms3'*...
                ker_eval(testInput(:,jj),trainInput,typeKernel,paramKernel);
          end
          err = testTarget - y_te;
          testMseKlms3(mc) = mean(err.^2);
        
        %=========end of KLMS3================

    end%mc

    disp('====================================')
    disp([num2str(paramRegularizationKlms3), ' regularization parameter'])
    disp('<<KLMS3')
    mseMeanKlms3(kkk) = mean(testMseKlms3);
    mseStdKlms3(kkk) = std(testMseKlms3);
    disp([num2str(mseMeanKlms3(kkk)),'+/-',num2str(mseStdKlms3(kkk))]);

    disp('====================================')

end
figure
errorbar(regularizationParam_v,mseMeanKlms3,mseStdKlms3,'LineWidth',2)
% set(gca,'XScale','log')
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('regularization parameter \lambda'),ylabel('Testing MSE')
grid on
axis tight

figure
plotyy(regularizationParam_v,mseMeanKlms3,regularizationParam_v,mseStdKlms3)


