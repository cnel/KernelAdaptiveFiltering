%Debugging AOGR

%Gaussian kernel
%nonlinear regression
%Learning curve

close all, clear all
%% Data Formatting
%addpath('H:\APrograms\toolBoxKAPA_old');
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

flagLearningCurve = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Algorithms Debugging
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%              KRLS
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

regularizationFactorKrls = 0.001;
forgettingFactorKrls = 1;

% flagLearningCurve = 1;
[expansionCoefficientKrls,learningCurveKrls] = ...
    KRLS_old(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactorKrls,forgettingFactorKrls,flagLearningCurve);

output_KRLS = zeros(trainSize,1);

for jj = 1:trainSize
     output_KRLS(jj) = expansionCoefficientKrls'*...
                ker_eval(trainInput(:,jj),trainInput,typeKernel,paramKernel);
end

%prediction result
output_KRLS_test = zeros(testSize,1);
for jj = 1:testSize
     output_KRLS_test(jj) = expansionCoefficientKrls'*...
                ker_eval(testInput(:,jj),trainInput,typeKernel,paramKernel);
end

%prediction variance
variance_KRLS_test = zeros(testSize,1);
Q_matrix = inv(gramMatrix(trainInput,typeKernel,paramKernel) + regularizationFactorKrls*eye(trainSize));

for jj = 1:testSize
    k_vector = ker_eval(testInput(:,jj),trainInput,typeKernel,paramKernel);
    f_vector = Q_matrix*k_vector;
    variance_KRLS_test(jj) = regularizationFactorKrls + ker_eval(testInput(:,jj),testInput(:,jj),typeKernel,paramKernel) -...
        k_vector'*f_vector;
end
% =========end of KRLS================


figure
semilogy(learningCurveKrls)
figure
plot(trainInput, trainTarget, 'o');
hold on
plot(trainInput, output_KRLS, '.r');

%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%              AOGR
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
th1 = 0.5;
th2 = 100;
regularizationFactorAogr = 0.001;
forgettingFactorAogr = 1;
[expansionCoefficientAogr,dictionaryIndexAogr,learningCurveAogr,CI_Aogr,CI2_Aogr,CI3_Aogr] = ...
    AOGR(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactorAogr,forgettingFactorAogr,th1,th2,flagLearningCurve);

lineWid = 2;

figure
subplot(4,1,1);
plot(CI_Aogr)
grid on
subplot(4,1,2);
plot(CI2_Aogr)
grid on
subplot(4,1,3);
plot(CI3_Aogr)
grid on
subplot(4,1,4);
plot(learningCurveAogr);
grid on

figure
subplot(2,1,1);
plot(CI2_Aogr,'LineWidth', lineWid)

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
%legend('Conditional Information')
xlabel('iteration'),ylabel('CI')
grid on


subplot(2,1,2);
semilogy(learningCurveAogr,'LineWidth', lineWid);

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
%legend('NLMS','SKLMS-1','SKAPA-2')
xlabel('iteration'),ylabel('MSE')
grid on

% =========end of AOGR================

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %
% %               Normalized APA 2
% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% regularizationFactorApa2 = 0.01;
% stepSizeWeightApa2 = .1;
% stepSizeBiasApa2 = .1;
% % flagLearningCurve = 1;
% % K =2;
% K = 10;
% [weightVectorApa2,biasTermApa2,learningCurveApa2]= ...
%     APA2(K,trainInput,trainTarget,testInput,testTarget,regularizationFactorApa2,stepSizeWeightApa2,stepSizeBiasApa2,flagLearningCurve);
% %=========end of Normalized APA 2================

% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %
% %              Leaky APA 3
% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% paramRegularizationApa3 = 0.01;
% stepSizeWeightApa3 = .02;
% stepSizeBiasApa3 = .02;
% % flagLearningCurve = 1;
% % K = 3;
% [weightVectorApa3,biasTermApa3,learningCurveApa3]= ...
%     APA3(K,trainInput,trainTarget,testInput,testTarget,paramRegularizationApa3,stepSizeWeightApa3,stepSizeBiasApa3,flagLearningCurve);
% %=========end of Leaky APA 3================
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %
% %              Leaky Normalized APA 4
% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% paramRegularizationApa4 = 0.01;
% stepSizeWeightApa4 = .1;
% stepSizeBiasApa4 = .1;
% % flagLearningCurve = 1;
% % K =5;
% [weightVectorApa4,biasTermApa4,learningCurveApa4]= ...
%     APA4(K,trainInput,trainTarget,testInput,testTarget,paramRegularizationApa4,stepSizeWeightApa4,stepSizeBiasApa4,flagLearningCurve);
% %=========end of Leaky Normalized APA 4================
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %
% %              Leaky Normalized APA 5
% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% paramRegularizationApa5 = 0.01;
% stepSizeWeightApa5 = .1;
% stepSizeBiasApa5 = .1;
% % flagLearningCurve = 1;
% % K =5;
% [weightVectorApa5,biasTermApa5,learningCurveApa5]= ...
%     APA5(K,trainInput,trainTarget,testInput,testTarget,paramRegularizationApa5,stepSizeWeightApa5,stepSizeBiasApa5,flagLearningCurve);
% %=========end of Leaky Normalized APA 4================
% %%
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %
% %               KAPA 1
% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% stepSizeKapa1 = .2;
% stepSizeWeightKapa1 = 0;
% stepSizeBiasKapa1 = 0;
% % flagLearningCurve = 1;
% % K = 10;
% [expansionCoeffKapa1,weightVectorKapa1,biasTermKapa1,learningCurveKapa1] = ...
%     KAPA1(K,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeKapa1,stepSizeWeightKapa1,stepSizeBiasKapa1,flagLearningCurve);
% %=========end of KAPA 1================
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %
% %           Normalized KAPA 2
% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% regularizationFactorKapa2 = 0.01;
% stepSizeKapa2 = .2;
% stepSizeWeightKapa2 = 0;
% stepSizeBiasKapa2 = 0;
% % flagLearningCurve = 1;
% % K = 5;
% 
% [expansionCoeffKapa2,weightVectorKapa2,biasTermKapa2,learningCurveKapa2] = ...
%     KAPA2(K,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactorKapa2,stepSizeKapa2,stepSizeWeightKapa2,stepSizeBiasKapa2,flagLearningCurve);
% %=========end of Normalized KAPA 2================
% % 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %
% %           leaky KAPA 3
% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% stepSizeKapa3 = .2;
% stepSizeWeightKapa3 = 0;
% stepSizeBiasKapa3 = 0;
% paramRegularizationKapa3 = 0.01;
% % flagLearningCurve = 1;
% % K = 5;
% 
% [expansionCoeffKapa3,weightVectorKapa3,biasTermKapa3,learningCurveKapa3] = ...
%     KAPA3(K,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,paramRegularizationKapa3,stepSizeKapa3,stepSizeWeightKapa3,stepSizeBiasKapa3,flagLearningCurve);
% %=========end of leaky KAPA 3================
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %
% %        leaky Newton's KAPA 4
% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% stepSizeKapa4 = 0.2;
% stepSizeWeightKapa4 = 0;
% stepSizeBiasKapa4 = 0;
% paramRegularizationKapa4 = 0.01;
% % flagLearningCurve = 1;
% % K = 5;
% [expansionCoeffKapa4,weightVectorKapa4,biasTermKapa4,learningCurveKapa4] = ...
%     KAPA4(K,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,paramRegularizationKapa4,stepSizeKapa4,stepSizeWeightKapa4,stepSizeBiasKapa4,flagLearningCurve);
% %=========end of leaky Newton's KAPA 4================
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %
% %        KAPA 5
% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% stepSizeKapa5 = 0.2;
% stepSizeWeightKapa5 = 0;
% stepSizeBiasKapa5 = 0;
% paramRegularizationKapa5 = 0.01;
% % toleranceDistance = 0.01;
% % tolerancePredictError = 0.1;
% % K = 5;
% [expansionCoeffKapa5,weightVectorKapa5,biasTermKapa5,learningCurveKapa5] = ...
%     KAPA5(K,trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,paramRegularizationKapa5,stepSizeKapa5,stepSizeWeightKapa5,stepSizeBiasKapa5,flagLearningCurve);
% 
% %=========end of KAPA 5================
% %%
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %
% %               KLMS 1
% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% stepSizeKlms1 = 1;
% stepSizeWeightKlms1 = 0;
% stepSizeBiasKlms1 = 0;
% % flagLearningCurve = 1;
% 
% [aprioriErrKlms1,weightVectorKlms1,biasTermKlms1,learningCurveKlms1] = ...
%     KLMS1(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,stepSizeKlms1,stepSizeWeightKlms1,stepSizeBiasKlms1,flagLearningCurve);
% %=========end of KLMS 1================
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %
% %           Normalized KLMS 2
% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% regularizationFactorKlms2 = 0.01;
% stepSizeKlms2 = 1.1;
% stepSizeWeightKlms2 = 0;
% stepSizeBiasKlms2 = 0;
% % flagLearningCurve = 1;
% 
% [aprioriErrKlms2,weightVectorKlms2,biasTermKlms2,learningCurveKlms2] = ...
%     KLMS2(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactorKlms2,stepSizeKlms2,stepSizeWeightKlms2,stepSizeBiasKlms2,flagLearningCurve);
% %=========end of Normalized KLMS 2================
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %
% %           leaky KLMS 3
% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% stepSizeKlms3 = 1;
% stepSizeWeightKlms3 = 0;
% stepSizeBiasKlms3 = 0;
% paramRegularizationKlms3 = 0.01;
% % flagLearningCurve = 1;
% 
% [expansionCoeffKlms3,weightVectorKlms3,biasTermKlms3,learningCurveKlms3] = ...
%     KLMS3(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,paramRegularizationKlms3,stepSizeKlms3,stepSizeWeightKlms3,stepSizeBiasKlms3,flagLearningCurve);
% %=========end of leaky KLMS 3================
% 
% %=========compact Kernel LMS 5===================
% stepSizeKlms5 = 1;
% stepSizeWeightKlms5 = 0;
% stepSizeBiasKlms5 = 0;
% flagLearningCurve = 1;
% toleranceDistance = 0.01;
% tolerancePredictError = 0.1;
% regularizationFactor = 1e-4;
% 
% [expansionCoefficientKlms5,weightVectorKlms5,biasTermKlms5,learningCurveKlms5,dictionaryIndexKlms5] = ...
%     sparseKLMS3(trainInput,trainTarget,testInput,testTarget,regularizationFactor,typeKernel,paramKernel,...
%     stepSizeKlms5,stepSizeWeightKlms5,stepSizeBiasKlms5,toleranceDistance,tolerancePredictError,flagLearningCurve);
% 
% %=========end of compact Kernel LMS================
% %%
% 
% %==========plot and test===================
% % figure(9)
% % plot(aprioriErrLms1,'r')
% % hold on
% % legend('L','K','leaky K')
% % hold off
% % xlabel('iteration'),ylabel('a priori error')

figure(10),
%plot(learningCurveLms1,'r-')
hold on
% plot(learningCurveApa2,'r.-')
%plot(learningCurveRls,'c:')
%plot(learningCurveExrls,'c--')
%plot(learningCurveExrls2,'c-')
plot(learningCurveKrls,'b.-')
plot(learningCurveAogr,'r-')
%plot(learningCurveExkrls,'b--')
% 
% plot(learningCurveKapa2,'g:')
%plot(learningCurveKlms1,'g-')
% plot(learningCurveApa3,'g.')
% plot(learningCurveApa4,'g--')
% plot(learningCurveApa5,'g.-')
% 
% plot(learningCurveKapa1,'c-')

% plot(learningCurveKapa3,'c.')
% plot(learningCurveKapa4,'c--')
% plot(learningCurveKapa5,'c.-')
% 

% plot(learningCurveKlms2,'b:')
% plot(learningCurveKlms3,'b.')
% plot(learningCurveKlms5,'b.-')
%legend('lms-1','Rls','Ex-rls','Ex-rls2','Krls','Ex-Krls','klms-1')
% hold off
xlabel('iteration'),ylabel('testing error')

