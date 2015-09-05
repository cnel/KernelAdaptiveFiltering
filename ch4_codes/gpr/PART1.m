%Copyright
%Weifeng Liu CNEL
%July 4 2008
%
%Description:
%Illustrate GPR
%
%Usage:
%Ch4, model selection
%

addpath('.\gpml-matlab\gpml')

close all
clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate the data
n = 100;
%rand('state',18);
%randn('state',20);
covfunc = {'covSum', {'covSEiso','covNoise'}};
loghyper = [log(1.0); log(1.0); log(0.1)];
x = 30*(rand(n,1)-0.5);
y = chol(feval(covfunc{:}, loghyper, x))'*randn(n,1);        % Cholesky decomp.

figure(1)
plot(x, y, 'k+', 'MarkerSize', 10);
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('x')
ylabel('y')

xstar = linspace(-15, 15, 201)';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% estimate the parameters and plot the regression result
loghyper = [log(1.0); log(1.0); log(0.1)] + 0.05*randn(3,1);
loghyper = minimize(loghyper, 'gpr', -100, covfunc, x, y);
disp(exp(loghyper))
[mu S2] = gpr(loghyper, covfunc, x, y, xstar);
S2 = S2 - exp(2*loghyper(3));
f = [mu+2*sqrt(S2);flipdim(mu-2*sqrt(S2),1)];
figure
fill([xstar; flipdim(xstar,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
hold on
plot(xstar,mu,'k-','LineWidth',2);
plot(x, y, 'k+', 'MarkerSize', 17);
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('x')
ylabel('f')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot the contour of marginal likelihood in terms of kernel size and noise
% variance
length = 40;
ell_vector = linspace(0.5, 1.2, length);
nv_vector = linspace(0.08,0.12,length);
nlml = zeros(length, length);

for ii = 1:length
    for jj = 1:length
        nlml(ii,jj) = gpr([log(ell_vector(ii)); log(1.0); log(nv_vector(jj))], covfunc, x, y);
    end
end

figure
contour(ell_vector, nv_vector, exp(-nlml), 10)
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
xlabel('kernel size')
ylabel('noise variance')
[c1, i1] = min(nlml);
[c2, i2] = min(c1);
hold on
plot(ell_vector(i2), nv_vector(i1(i2)),'k+','MarkerSize',10)