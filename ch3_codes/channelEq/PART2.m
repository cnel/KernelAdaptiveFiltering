%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Copyright Weifeng Liu 
%CNEL
%July 1, 2008
%
%description:
%compare the performance of LMS1 APA1 SKLMS1 SKAPA1 SKAPA2 in channel
%equalization
%SNR vs Bit error rate
%
%Usage:
%ch3, channel equalization, figure 3-8
%
%Outside functions called or toolboxes used:
%LMS1, APA1, sparseKLMS1, sparseKAPA1, sparseKAPA2
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

flagLearningCurve = 0;

K = 10;
stepSizeWeightApa1 = .02/K;
stepSizeBiasApa1 = .002;

stepSizeKapa1 = .2/K;
stepSizeWeightKapa1 = 0;
stepSizeBiasKapa1 = 0;


stepSizeKapa2 = .5/K;
stepSizeWeightKapa2 = 0;
stepSizeBiasKapa2 = 0;
paramRegularizationKapa2 = 0.01;

stepSizeKlms1 = .15;
stepSizeWeightKlms1 = 0;
stepSizeBiasKlms1 = 0;

%data size
trainSize = 1000;
testSize = 100000;

noise_std_v =               [0.4  0.3 .15 .25 .35 .2  .1  .05 .01   .45 .6  1];
toleranceDistance_v =       [0.25 .2  .07 .13 .22 .1  .04 .01 .0005 .45 .8  1];
tolerancePredictError_v =   [0.12 .1  .08 .08 .09 .09 .03 .01 .001  .1  .1 .2];

numSNR = length(noise_std_v);

networksizeLms1 = zeros(numSNR,1);
networksizeApa1 = zeros(numSNR,1);
networksizeKlms1 = zeros(numSNR,1);
networksizeKapa1 = zeros(numSNR,1);
networksizeKapa2 = zeros(numSNR,1);

berLms1 = zeros(numSNR,1);
berApa1 = zeros(numSNR,1);
berKlms1 = zeros(numSNR,1);
berKapa1 = zeros(numSNR,1);
berKapa2 = zeros(numSNR,1);

networksizeLms1 = zeros(numSNR,1);
networksizeApa1 = zeros(numSNR,1);
networksizeKlms1 = zeros(numSNR,1);
networksizeKapa1 = zeros(numSNR,1);
networksizeKapa2 = zeros(numSNR,1);

 
MC = 10;
errNumLms1 = zeros(MC,1);
errNumApa1 = zeros(MC,1);
errNumSkapa1 = zeros(MC,1);
errNumSkapa2 = zeros(MC,1);
errNumSklms1 = zeros(MC,1);

for kkk = 1:length(noise_std_v);
    
    noise_std = noise_std_v(kkk);
    toleranceDistance = toleranceDistance_v(kkk);
    tolerancePredictError = tolerancePredictError_v(kkk);

    for mc = 1:MC

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
        
        length(expansionCoeffKapa1)
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

        length(expansionCoeffKapa2)
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
        
        length(expansionCoefficientKlms1)
        %=========end of sparse Kernel LMS================
        %%
    end%mc

    %%

    disp('====================================')

    disp('noise std')
    disp(num2str(noise_std))
    

%%%%%%%%%%%%%%
    mseMean = mean(errNumLms1);
    mseStd = std(errNumLms1);
    
    disp('>>LMS1')
    berLms1(kkk) = mseMean/testSize;

    disp([num2str(mseMean),'+/-',num2str(mseStd)]);
%%%%%%%%%%%%
    mseMean = mean(errNumApa1);
    mseStd = std(errNumApa1);

    disp('>>APA1')
    berApa1(kkk) = mseMean/testSize;
    disp([num2str(mseMean),'+/-',num2str(mseStd)]);

%%%%%%%%%%%%%

    mseMean = mean(errNumSklms1);
    mseStd = std(errNumSklms1);
    disp('>>SKLMS1')
    berKlms1(kkk) = mseMean/testSize;
    disp([num2str(mseMean),'+/-',num2str(mseStd)]);

%%%%%%%%%%%%%%
    mseMean = mean(errNumSkapa1);
    mseStd = std(errNumSkapa1);
    disp('>>SKAPA1')
    berKapa1(kkk) = mseMean/testSize;
    disp([num2str(mseMean),'+/-',num2str(mseStd)]);
%%%%%%%%%%%%%%%

    mseMean = mean(errNumSkapa2);
    mseStd = std(errNumSkapa2);
    disp('>>SKAPA2')
    berKapa2(kkk) = mseMean/testSize;
    disp([num2str(mseMean),'+/-',num2str(mseStd)]);

    disp('====================================')
end

sigma = 10*log10(1./(noise_std_v.^2)); %it is the SNR

[sigma,IX] = sort(sigma)
lineWid = 3;

figure
plot(sigma, berLms1(IX),'k:','LineWidth', lineWid-2)
hold on
plot(sigma, berApa1(IX),'k-','LineWidth', lineWid-2)
plot(sigma, berKlms1(IX),'k-','LineWidth', lineWid)
plot(sigma, berKapa1(IX),'k--','LineWidth', lineWid)
plot(sigma, berKapa2(IX),'k-.','LineWidth', lineWid)
hold off

legend('LMS','APA-1','KLMS-NC','KAPA-1-NC','KAPA-2-NC')
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('Normalized SNR (dB)'),ylabel('BER')
grid on
