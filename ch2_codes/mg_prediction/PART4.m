%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Copyright Aaron Liu 
%CNEL
%July 1, 2008
%
%description:
%compare the performance of LMS, KLMS and RN for Mackey Glass time series
%one step prediction
%Monte Carlo simulation with different levels of noise
%
%Usage:
%ch2, m-g prediction, tables 2-4 and 2-5
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

%data size
N_tr = 500;
N_te = 100;%
%%======end of config=======

%======monte carlo init =======
MC = 50;
mse_tr_l = zeros(MC,1);
mse_te_l = zeros(MC,1);
mse_tr_klms = zeros(MC,1);
mse_te_klms = zeros(MC,1);
mse_tr_rbf = zeros(MC,1);
mse_te_rbf = zeros(MC,1);

%noise standard deviation
np_v =[.04, .1]; %you can add noise deviation inside the vector

for kkk = 1:length(np_v)
    np = np_v(kkk);
    
    disp(['Noise deviation:', num2str(np)]);
    disp([num2str(MC), ' Monte Carlo simulations. Please wait...']);

    for mc = 1:MC
        disp(mc);

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

        %=========Linear LMS===================
        %learning rate (step size)
        lr_l = .2;%learning rate
        w1 = zeros(1,TD);
        e_l = zeros(N_tr,1);
        for n=1:N_tr
            y = w1*X(:,n);
            e_l(n) = T(n) - y;
            w1 = w1 + lr_l*e_l(n)*X(:,n)';
        end
        err_tr = T'-(w1*X);
        mse_tr_l(mc) = mean(err_tr.^2);
        err_te = T_te'-(w1*X_te);
        mse_te_l(mc) = mean(err_te.^2);
        %=========end of Linear LMS================

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

        %training error
        y_tr = zeros(N_tr,1);
        for jj = 1:N_tr
            y_tr(jj) = lr_k*e_k'*(exp(-sum((X(:,jj)*ones(1,N_tr)-X).^2)))';
        end
        err = T - y_tr;
        mse_tr_klms(mc) = mean(err.^2);
        %%testing
        y_te = zeros(N_te,1);
        for jj = 1:N_te
            y_te(jj) = lr_k*e_k'*(exp(-sum((X_te(:,jj)*ones(1,N_tr)-X).^2)))';
        end
        err = T_te - y_te;
        mse_te_klms(mc) = mean(err.^2);
        %=========end of Kernel LMS================

        %=========RN============================
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
        % %training error
        err = lam*a;
        mse_tr_rbf(mc) = mean(err.^2);
        y_te = zeros(N_te,1);
        %testing
        for jj = 1:N_te
            y_te(jj) = a'*(exp(-sum((X_te(:,jj)*ones(1,N_tr)-X).^2)))';
        end
        err = T_te - y_te;
        mse_te_rbf(mc) = mean(err.^2);
        %=========End of RN===================
    end%mc

    disp(['Noise power: ',num2str(np)]);
    disp('<<LMS')
    disp([num2str(mean(mse_tr_l)),'+/-',num2str(std(mse_tr_l))])
    disp([num2str(mean(mse_te_l)),'+/-',num2str(std(mse_te_l))])
    disp('<<KLMS')
    disp([num2str(mean(mse_tr_klms)),'+/-',num2str(std(mse_tr_klms))])
    disp([num2str(mean(mse_te_klms)),'+/-',num2str(std(mse_te_klms))])
    disp('<<RN')
    disp([num2str(mean(mse_tr_rbf)),'+/-',num2str(std(mse_tr_rbf))])
    disp([num2str(mean(mse_te_rbf)),'+/-',num2str(std(mse_te_rbf))])

end%noise deviation