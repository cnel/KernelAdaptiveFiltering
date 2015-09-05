% rayleigh channel tracking
% Weifeng Liu 
% Sep. 2007.
%
% Description:
% particle filter
%
% Usage
% Ch 4
%
% Outside functions used
% 


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

% time delay (embedding) length
inputDimension = channelLength;

%Nonlinearity parameter
typeNonlinear = 1;
paramNonlinear = 2;

% # of particles for channel estimation
xn=50;  

L = 100;

mse_perfectKnowledge = zeros(1,L);
mse_partialKnowledge = zeros(1,L);
mse_wrongKnowledge = zeros(1,L);

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
    %               Particle filter perfect knowledge
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%initial parameterxn,channel0
    Q=zeros(channelLength,1);  % noise variance of v in channel state updating c(k)=F*c(k-1)+v(k-1)
    [F,y_channel,e_channel]=ls([zeros(channelLength,1),channel(:,1:numberSymbol-1)],channel);
    for i=1:channelLength
        Q(i)=var(e_channel(i,:));
    end

    R=var(trainTarget'-nlG(sum(trainInput.*channel,1),paramNonlinear,typeNonlinear)); % noise variance of n in transfer function z(k)=nlG(u(k)*channel(k))+n(k)
    channel0=repmat(channel(:,1)',xn,1); % inital of channel0 in system

    %%%%%%%%%%%%%%%calculate channel estimation
    [channelExp,channelMle,Xrecord]=particlefilter(channel0,trainInput,trainTarget',Q,R,xn,F,paramNonlinear,typeNonlinear);

    %%%%%%%%%%%%%%%calculated output estimation
    trainOutputMle=nlG(sum(trainInput.*channelMle,1),paramNonlinear,typeNonlinear);
    
    mse_perfectKnowledge(k) = mse(trainTarget'-trainOutputMle);

    %=========end================

     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %               Particle filter partial knowledge
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % random init
    channel0=repmat(randn(1,channelLength),xn,1); % inital of channel0 in system

    %%%%%%%%%%%%%%%calculate channel estimation
    [channelExp,channelMle,Xrecord]=particlefilter(channel0,trainInput,trainTarget',Q,R,xn,F,paramNonlinear,typeNonlinear);

    %%%%%%%%%%%%%%%calculated output estimation
    trainOutputMle=nlG(sum(trainInput.*channelMle,1),paramNonlinear,typeNonlinear);
    
    mse_partialKnowledge(k) = mse(trainTarget'-trainOutputMle);

    %=========end================

     
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %               wrong filter partial knowledge
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Nonlinearity parameter
    %typeNonlinear = 1;
    %paramNonlinear = 2;
    
    typeNonlinear = 2;
    paramNonlinear = 0.1;

    channel0=repmat(randn(1,channelLength),xn,1); % inital of channel0 in system

    %%%%%%%%%%%%%%%calculate channel estimation
    [channelExp,channelMle,Xrecord]=particlefilter(channel0,trainInput,trainTarget',Q,R,xn,F,paramNonlinear,typeNonlinear);

    %%%%%%%%%%%%%%%calculated output estimation
    trainOutputMle=nlG(sum(trainInput.*channelMle,1),paramNonlinear,typeNonlinear);
    
    mse_wrongKnowledge1(k) = (mse(trainTarget'-trainOutputMle));
    
    typeNonlinear = 1;
    paramNonlinear = 2;

    %=========end================

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %               wrong filter partial knowledge
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Nonlinearity parameter
    %typeNonlinear = 1;
    %paramNonlinear = 2;
    
    typeNonlinear = 1;
    paramNonlinear = 1;

    channel0=repmat(randn(1,channelLength),xn,1); % inital of channel0 in system

    %%%%%%%%%%%%%%%calculate channel estimation
    [channelExp,channelMle,Xrecord]=particlefilter(channel0,trainInput,trainTarget',Q,R,xn,F,paramNonlinear,typeNonlinear);

    %%%%%%%%%%%%%%%calculated output estimation
    trainOutputMle=nlG(sum(trainInput.*channelMle,1),paramNonlinear,typeNonlinear);
    
    mse_wrongKnowledge2(k) = (mse(trainTarget'-trainOutputMle));
    
    typeNonlinear = 1;
    paramNonlinear = 2;

    %=========end================
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %               wrong filter partial knowledge
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Nonlinearity parameter
    %typeNonlinear = 1;
    %paramNonlinear = 2;
    
    typeNonlinear = 1;
    paramNonlinear = 3;

    channel0=repmat(randn(1,channelLength),xn,1); % inital of channel0 in system

    %%%%%%%%%%%%%%%calculate channel estimation
    [channelExp,channelMle,Xrecord]=particlefilter(channel0,trainInput,trainTarget',Q,R,xn,F,paramNonlinear,typeNonlinear);

    %%%%%%%%%%%%%%%calculated output estimation
    trainOutputMle=nlG(sum(trainInput.*channelMle,1),paramNonlinear,typeNonlinear);
    
    mse_wrongKnowledge3(k) = (mse(trainTarget'-trainOutputMle));
    
    typeNonlinear = 1;
    paramNonlinear = 2;

    %=========end================
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %               wrong filter partial knowledge
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Nonlinearity parameter
    %typeNonlinear = 1;
    %paramNonlinear = 2;
    
    typeNonlinear = 0;
    paramNonlinear = 1;
   
    channel0=repmat(randn(1,channelLength),xn,1); % inital of channel0 in system

    %%%%%%%%%%%%%%%calculate channel estimation
    [channelExp,channelMle,Xrecord]=particlefilter(channel0,trainInput,trainTarget',Q,R,xn,F,paramNonlinear,typeNonlinear);

    %%%%%%%%%%%%%%%calculated output estimation
    trainOutputMle=nlG(sum(trainInput.*channelMle,1),paramNonlinear,typeNonlinear);
    
    mse_wrongKnowledge4(k) = (mse(trainTarget'-trainOutputMle));
    
    typeNonlinear = 1;
    paramNonlinear = 2;

    %=========end================
    
    
end


disp('====================================')

disp('perfect knowledge')
mse_perfectKnowledge = 10*log10(mse_perfectKnowledge);

mseMean = mean(mse_perfectKnowledge);
mseStd = std(mse_perfectKnowledge);

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

mse_partialKnowledge = 10*log10(mse_partialKnowledge);
disp('partial knowledge')
mseMean = mean(mse_partialKnowledge);
mseStd = std(mse_partialKnowledge);

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

mse_wrongKnowledge1 = 10*log10(mse_wrongKnowledge1);
disp('wrong knowledge 1')
mseMean = mean(mse_wrongKnowledge1);
mseStd = std(mse_wrongKnowledge1);

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

mse_wrongKnowledge2 = 10*log10(mse_wrongKnowledge2);
disp('wrong knowledge 2')
mseMean = mean(mse_wrongKnowledge2);
mseStd = std(mse_wrongKnowledge2);

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

mse_wrongKnowledge3 = 10*log10(mse_wrongKnowledge3);
disp('wrong knowledge 3')
mseMean = mean(mse_wrongKnowledge3);
mseStd = std(mse_wrongKnowledge3);

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

mse_wrongKnowledge4 = 10*log10(mse_wrongKnowledge4);
disp('wrong knowledge 4')
mseMean = mean(mse_wrongKnowledge4);
mseStd = std(mse_wrongKnowledge4);

disp([num2str(mseMean),'+/-',num2str(mseStd)]);

disp('====================================')

