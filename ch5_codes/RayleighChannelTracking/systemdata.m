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
numberSymbol = 1000;
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


L = 1;

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
end