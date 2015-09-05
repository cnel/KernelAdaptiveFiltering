%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Copyright Weifeng Liu 
%CNEL
%July 1, 2008
%
%description:
%compare the performance of LMS and KLMS for Mackey Glass time series
%one step prediction
%Learning curves
%
%Usage:
%ch2, m-g prediction
%
%Outside functions called:
%none
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all,
close all
clc
%======filter config=======
%time delay (embedding) length
TD = 10;
%kernel parameter
a = 1;%fixed
%noise std
np =.04;
%data size
N_tr = 500;
N_te = 100;%
%%======end of config=======


disp('Learning curves are generating. Please wait...');

%======data formatting===========
load MK30   %MK30 5000*1
MK30 = MK30+np*randn(size(MK30));
MK30 = MK30 - mean(MK30);

%500 training data
train_set = MK30(1501:4500);

%100 testing data
test_set = MK30(4601:4900);

%data embedding
X = zeros(TD,N_tr);
for k=1:N_tr
    X(:,k) = train_set(k:k+TD-1)';
end
T = train_set(TD+1:TD+N_tr);

X_te = zeros(TD,N_te);
for k=1:N_te
    X_te(:,k) = test_set(k:k+TD-1)';
end
T_te = test_set(TD+1:TD+N_te);
%======end of data formatting===========

%
mse_te_l = zeros(N_tr,1);

%=========Linear LMS===================
%learning rate (step size)
lr_l = .2;%learning rate
w1 = zeros(1,TD);
e_l = zeros(N_tr,1);
for n=1:N_tr
    y = w1*X(:,n);
    e_l(n) = T(n) - y;
    w1 = w1 + lr_l*e_l(n)*X(:,n)';

    %testing MSE for learning curve
    err_te = T_te'-(w1*X_te);
    mse_te_l(n) = mean(err_te.^2);
end
%=========end of Linear LMS================

%=========Kernel LMS===================

%learning rate (adjustable)
%    lr_k = .1;
lr_k = .2;
%   lr_k = .6;

%init
e_k = zeros(N_tr,1);
y = zeros(N_tr,1);
mse_te_k = zeros(N_tr,1);

% n=1 init
e_k(1) = T(1);
y(1) = 0;
mse_te_k(1) = mean(T_te.^2);
% start
for n=2:N_tr
    %training
    ii = 1:n-1;
    y(n) = lr_k*e_k(ii)'*(exp(-sum((X(:,n)*ones(1,n-1)-X(:,ii)).^2)))';
    e_k(n) = T(n) - y(n);
    
    %testing MSE
    y_te = zeros(N_te,1);
    for jj = 1:N_te
        y_te(jj) = lr_k*e_k(1:n)'*(exp(-sum((X_te(:,jj)*ones(1,n)-X(:,1:n)).^2)))';
    end
    err = T_te - y_te;
    mse_te_k(n) = mean(err.^2);
    
end

%=========end of Kernel LMS================

figure
plot(mse_te_l,'k-','LineWidth',2);
hold on
plot(mse_te_k,'k--','LineWidth',2);

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');

legend('LMS', 'KLMS')
xlabel('iteration')
ylabel('MSE')