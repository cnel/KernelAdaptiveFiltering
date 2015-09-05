%Copyright
%Weifeng Liu CNEL
%July 4 2008
%
%Description:
%Learning curves and effective data
%
%Usage:
%Ch5, CO2 concentration forecasting, figure 5-10
%
%ouside functions called
%ker_eval

close all, clear all
clc
%%
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

disp('learning curve is generating')
%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%              AOGR
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
th1 = -.03061;
th2 = 100;
regularizationFactorAogr = 0.00;
forgettingFactorAogr = 1;
[expansionCoefficientAogr,dictionaryIndexAogr,learningCurveAogr,CI] = ...
    AOGR(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactorAogr,forgettingFactorAogr,1,th1,th2,[],flagLearningCurve);

%%
lineWid = 2;

figure
subplot(4,1,1);
plot(CI(:,1))
grid on
subplot(4,1,2);
plot(CI(:,2))
grid on
subplot(4,1,3);
plot(CI(:,3))
grid on
subplot(4,1,4);
plot(learningCurveAogr);
grid on

figure
subplot(2,1,1);
plot(CI(:,2),'LineWidth', lineWid)

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
%legend('Conditional Information')
xlabel('iteration'),ylabel('surprise')
grid on


subplot(2,1,2);
semilogy(learningCurveAogr,'LineWidth', lineWid);

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('iteration'),ylabel('MSE')
grid on

figure
[AX,H1,H2] = plotyy(1:trainSize-50,CI(1:trainSize-50,1),1:trainSize-50,10*log10(learningCurveAogr(1:trainSize-50)));

hold on
plot(gca,dictionaryIndexAogr(1:end-6),CI(dictionaryIndexAogr(1:end-6),1),'ro')
grid on


set(get(AX(1),'Ylabel'),'String','surprise','FontSize', 14,'FontName', 'Arial')
set(get(AX(2),'Ylabel'),'String','testing MSE (dB)','FontSize', 14,'FontName', 'Arial')
set(get(AX(2),'Xlabel'),'String','iteration','FontSize', 14,'FontName', 'Arial')

set(H1,'LineStyle','x','MarkerSize', 8)
set(H2,'LineStyle','-','LineWidth', lineWid)
set(AX(1), 'FontSize', 14);
set(AX(1), 'FontName', 'Arial');
set(AX(2), 'FontSize', 14);
set(AX(2), 'FontName', 'Arial');

% =========end of AOGR================

