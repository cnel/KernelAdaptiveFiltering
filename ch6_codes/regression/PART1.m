%Copyright
%Weifeng Liu CNEL
%July 4 2008
%
%Description:
%Evaluation the surprise in nonlinear regression
%surprise vs learning curves
%
%Usage:
%Ch5, nonlinear regression, figure 5-1
%
%ouside functions called
%AOGR, nlG


close all, clear all
clc
%% Data Formatting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Data Formatting
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Input signal
inputSignal = randn(2000,1);
% figure(1),plot(inputSignal)

% Synthesize the linear filter
linearFilter = 1;

% time delay (embedding) length
inputDimension = length(linearFilter);

%Nonlinearity parameter
typeNonlinear = 2;
paramNonlinear = 2;

%Kernel parameters
typeKernel = 'Gauss';
paramKernel = 0.2;

% Data size for training and testing
trainSize = 200;
testSize = 100;

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

%Desired training signal
trainTarget = zeros(trainSize,1);
for ii=1:trainSize
    trainTarget(ii) = inputSignal(ii:ii+inputDimension-1)'*linearFilter;
end
%Pass through the nonlinearity
trainTarget = nlG(trainTarget,paramNonlinear,typeNonlinear);

%Desired training signal
testTarget = zeros(testSize,1);
for ii=1:testSize
    testTarget(ii) = inputSignal(ii+trainSize:ii+inputDimension-1+trainSize)'*linearFilter;
end
%Pass through the nonlinearity
testTarget = nlG(testTarget,paramNonlinear,typeNonlinear);

%%
flagLearningCurve = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%              AOGR
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
th1 = -1000000;
th2 = 1000000;
inputApriori = trainInput.^2/2;

regularizationFactorAogr = 0.001;
forgettingFactorAogr = 1;
criterion = 1;
[expansionCoefficientAogr,dictionaryIndexAogr,learningCurveAogr,CI_Aogr] = ...
    AOGR(trainInput,trainTarget,testInput,testTarget,typeKernel,paramKernel,regularizationFactorAogr,forgettingFactorAogr,criterion,th1,th2,inputApriori,flagLearningCurve);

lineWid = 2;

figure
[AX,H1,H2] = plotyy(1:trainSize,CI_Aogr(:,1),1:trainSize,10*log10(learningCurveAogr));

grid on

set(get(AX(1),'Ylabel'),'String','surprise','FontSize', 14,'FontName', 'Arial')
set(get(AX(2),'Ylabel'),'String','testing MSE (dB)','FontSize', 14,'FontName', 'Arial')
set(get(AX(2),'Xlabel'),'String','iteration','FontSize', 14,'FontName', 'Arial')

set(H1,'Marker','x','MarkerSize', 8)
set(H1,'LineStyle','none')
set(H2,'LineStyle','-','LineWidth', lineWid)
set(AX(1), 'FontSize', 14);
set(AX(1), 'FontName', 'Arial');
set(AX(2), 'FontSize', 14);
set(AX(2), 'FontName', 'Arial');


[B,IX] = sort(CI_Aogr(:,1),'descend');

figure
subplot(2,1,1)
plot(1:trainSize,trainInput,'o','MarkerSize', 8, 'MarkerEdgeColor','b');
hold on
plot(IX(1:30),trainInput(:,IX(1:30)),'o','MarkerSize', 8, 'MarkerEdgeColor','w');
plot(IX(1:30),trainInput(:,IX(1:30)),'x','MarkerSize', 8, 'MarkerEdgeColor','r');
set(gca,'FontSize', 14,'FontName', 'Arial')

ylabel('input')
grid on

subplot(2,1,2)
plot(1:trainSize,trainTarget,'o','MarkerSize', 8, 'MarkerEdgeColor','b');
hold on
plot(IX(1:30),trainTarget(IX(1:30)),'o','MarkerSize', 8, 'MarkerEdgeColor','w');
plot(IX(1:30),trainTarget(IX(1:30)),'x','MarkerSize', 8, 'MarkerEdgeColor','r');
set(gca,'FontSize', 14,'FontName', 'Arial')

grid on

xlabel('iteration')
ylabel('desired')

% plot in 2D
figure
plot(trainInput,trainTarget, 'o','MarkerSize', 8, 'MarkerEdgeColor','b');
hold on
plot(trainInput(:,IX(1:30)), trainTarget(IX(1:30)),'o','MarkerSize', 8, 'MarkerEdgeColor','w');
plot(trainInput(:,IX(1:30)), trainTarget(IX(1:30)),'x','MarkerSize', 8, 'MarkerEdgeColor','r');

set(gca,'FontSize', 14,'FontName', 'Arial')

grid on

xlabel('input')
ylabel('desired')

