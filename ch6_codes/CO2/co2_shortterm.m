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
trainSize = dataSize;
testSize = dataSize - trainSize;

%Input training signal with data embedding
trainInput = date(1:trainSize)';

%Input test data with embedding
testIndex = dataSize-180:dataSize;

%Desired training signal
trainTarget = ave_interpolate(1:trainSize);

%%
%Kernel parameters
typeKernel = 'CO2';
paramKernel = 0.2;

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

regularizationFactor = 0.001;
forgettingFactor = 1;

expansionCoefficient = zeros(trainSize,1);
prediction_mean = zeros(trainSize,1);
prediction_variance = zeros(trainSize,1);

Q_matrix = 1/(forgettingFactor*regularizationFactor + ker_eval(trainInput(:,1),trainInput(:,1),typeKernel,paramKernel));
expansionCoefficient(1) = Q_matrix*trainTarget(1);
% start training
for n = 2:trainSize
    ii = 1:n-1;
    k_vector = ker_eval(trainInput(:,n),trainInput(:,ii),typeKernel,paramKernel);
    f_vector = Q_matrix*k_vector;
    s = 1/(regularizationFactor*forgettingFactor^(n)+ ker_eval(trainInput(:,n),trainInput(:,n),typeKernel,paramKernel) - k_vector'*f_vector);
    Q_tmp = zeros(n,n);
    Q_tmp(ii,ii) = Q_matrix + f_vector*f_vector'*s;
    Q_tmp(ii,n) = -f_vector*s;
    Q_tmp(n,ii) = Q_tmp(ii,n)';
    Q_tmp(n,n) = s;
    Q_matrix = Q_tmp;
    
    prediction_mean(n) = k_vector'*expansionCoefficient(ii);
    prediction_variance(n) = regularizationFactor + ker_eval(trainInput(:,n),trainInput(:,n),typeKernel,paramKernel) -...
        k_vector'*f_vector;
    
    error = trainTarget(n) - prediction_mean(n);
    
    % updating
    expansionCoefficient(n) = s*error;
    expansionCoefficient(ii) = expansionCoefficient(ii) - f_vector*expansionCoefficient(n);
  

end
% =========end of KRLS================

%%
figure;
plot(trainInput, trainTarget, 'b');
hold on
plot(trainInput(testIndex), prediction_mean(testIndex), 'r');
plot(trainInput(testIndex), prediction_mean(testIndex) + prediction_variance(testIndex)*5, 'g');
plot(trainInput(testIndex), prediction_mean(testIndex) - prediction_variance(testIndex)*5, 'g');
grid
hold off
%%

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


