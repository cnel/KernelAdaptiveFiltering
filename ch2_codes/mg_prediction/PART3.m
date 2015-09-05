%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Copyright Weifeng Liu 
%CNEL
%July 1, 2008
%
%description:
%compare the solution norm of KLMS and RN for Mackey Glass time series
%one step prediction
%Monte Carlo simulation
%
%Usage:
%ch2, m-g prediction, table 2-3
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
%noise standard deviation
np =.04;
%data size
N_tr = 500;
N_te = 100;%
%%======end of config=======

%======monte carlo init =======
load MK30   %MK30 5000*1

MK30_tmp = MK30;

MC = 50;

solutionNorm_klms = zeros(MC,1);
solutionNorm_rbf = zeros(MC,1);

disp([num2str(MC), ' Monte Carlo simulations. Please wait...']);

for mc = 1:MC
    disp(mc);
    
    %======data formatting===========
    MK30_tmp = MK30; %restore

    MK30_tmp = MK30_tmp + np*randn(size(MK30));
    MK30_tmp = MK30_tmp - mean(MK30_tmp);
    
    %500 training data
    train_set = MK30_tmp(1501:4500);
    
    %100 testing data
    test_set = MK30_tmp(4601:4900);
    
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

    %=========Kernel LMS===================

    %learning rate (adjustable)
%    lr_k = .1;
    lr_k = .2;
 %   lr_k = .6;
 
    %init
    e_k = zeros(N_tr,1);
    y = zeros(N_tr,1);

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
    end
    
    solutionNorm_klms(mc) = lr_k*norm(e_k);

    %=========end of Kernel LMS================

    %=========RBF============================
    %regularization parameter (adjustable)
    lam = 1;

    G = zeros(N_tr,N_tr);
    for i=1:N_tr-1
        j=i+1:N_tr;
        G(i,j)=exp(-sum((X(:,i)*ones(1,N_tr-i)-X(:,j)).^2));
        G(j,i)=G(i,j)';
    end
    G = G + eye(N_tr);
    G_lam =G + lam*eye(N_tr);
    a = inv(G_lam)*T;
    
    solutionNorm_rbf(mc) = norm(a);
    
    %=========End of RBF===================
end%mc

disp('soltion norm')
disp(['Noise power: ',num2str(np)]);
disp('<<KLMS')
disp([num2str(mean(solutionNorm_klms)),'+/-',num2str(std(solutionNorm_klms))])
disp('<<RN')
disp([num2str(mean(solutionNorm_rbf)),'+/-',num2str(std(solutionNorm_rbf))])