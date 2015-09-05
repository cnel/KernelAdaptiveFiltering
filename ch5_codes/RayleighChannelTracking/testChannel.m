%%%%%%%%%%%%%% initial desired data set
systemdata; %generate one Monte Carlo set of data: trainInput, TrainTarget, Channel, nlG
[D,T]=size(channel);

% wrong info
typeNonlinear = 1; % true 1 
paramNonlinear = 2; %true 2

%%%%%%%%%%%%%%%estimate the parameter: F,Q,R,
Q=zeros(D,1);  % noise variance of v in channel state updating c(k)=F*c(k-1)+v(k-1)
[F,y_channel,e_channel]=ls([zeros(D,1),channel(:,1:T-1)],channel);
for i=1:D
    Q(i)=var(e_channel(i,:));
end

R=var(trainTarget'-nlG(sum(trainInput.*channel,1),paramNonlinear,typeNonlinear)); % noise variance of n in transfer function z(k)=nlG(u(k)*channel(k))+n(k)


%no info
R = 0.0003;
F = eye(channelLength,channelLength);
Q = 0.0005*(zeros(channelLength,1)+1);


%%%%%%%%%%%%%%%initial parameterxn,channel0
xn=50;  % # of particles for channel estimation
channel0=repmat(channel(:,1)',xn,1); % inital of channel0 in system

%%%%%%%%%%%%%%%calculate channel estimation
[channelExp,channelMle,Xrecord]=particlefilter(channel0,trainInput,trainTarget',Q,R,xn,F,paramNonlinear,typeNonlinear);

%%%%%%%%%%%%%%%calculated output estimation
trainOutputExp=nlG(sum(trainInput.*channelExp,1),paramNonlinear,typeNonlinear);
trainOutputMle=nlG(sum(trainInput.*channelMle,1),paramNonlinear,typeNonlinear);

%%%%%%%%%%%%%%%draw the results
figure;
plot(trainTarget,'r:');
hold on;
plot(trainOutputMle,'g-.');% get the estimate state by maxlikehood (option 1)
plot(trainOutputExp);% get the estimate state by expectation (option 2)
legend('desired output', 'Channel by MAP','Channel by Exp');
err1=10*log10(mse(trainTarget'-trainOutputExp))
err2=10*log10(mse(trainTarget'-trainOutputMle))
