%Copyright
%Weifeng Liu CNEL
%July 4 2008
%
%Description:
%Evaluate the effectiveness of Maximum marginal likelihood
%
%Usage:
%Ch4, model selection
%

addpath('.\gpml-matlab\gpml')

close all
clear
clc

MC = 100

theta = zeros(3,MC)
n = 100;
covfunc = {'covSum', {'covSEiso','covNoise'}};
for mc = 1:MC
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % generate the data


    loghyper = [log(1.0); log(1.0); log(0.1)];
    x = 30*(rand(n,1)-0.5);
    y = chol(feval(covfunc{:}, loghyper, x))'*randn(n,1);        % Cholesky decomp.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % estimate the parameters and plot the regression result
    loghyper = [log(1.0); log(1.0); log(0.1)] + 0.05*randn(3,1);
    loghyper = minimize(loghyper, 'gpr', -100, covfunc, x, y);
    theta(:, mc) = exp(loghyper);
end

mean(theta(1,:))
std(theta(1,:))

mean(theta(2,:))
std(theta(2,:))

mean(theta(3,:))
std(theta(3,:))

