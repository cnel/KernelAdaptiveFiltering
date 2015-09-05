%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Copyright Aaron Liu 
%CNEL
%July 1, 2008
%
%description:
%compare the performance of LMS, KLMS, RN in nonlinear channel equalization
%monte carlo different noise level
%
%Usage:
%ch2, nonlinear channel equalization, table 2-9
%
%Outside functions called:
%None
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all,
close all
clc

%======filter config=======
%time delay (embedding) length
TD = 5;
D = 2;
h = .1;%kernel size

%======end of config=======
MC = 10;
err_l = zeros(MC,1);
err_k = zeros(MC,1);
err_rbf = zeros(MC,1);

np_v = [.1, .4, .8];

for kkk = 1:length(np_v);
    np = np_v(kkk);
    disp([num2str(MC), ' Monte Carlo simulations'])

    for mc=1:MC
        disp(mc)

        %=========data===============
        % Generate binary data
        u = rand(1,6500)>0.5;
        u = 2*u-1;
        z = u+0.5*[0,u(1:end-1)];
        % Channel noise
        ns = np*randn(1,length(u));
        % Ouput of the nonlinear channel
        y = z - 0.9*z.^2 + ns;

        %data size
        N_tr = 1000;
        N_te = 5000;

        %data embedding
        X = zeros(TD,N_tr);
        for k=1:N_tr
            X(:,k) = y(k:k+TD-1)';
        end
        % Test data
        X_te = zeros(TD,N_te);
        for k=1:N_te
            X_te(:,k) = y(k+N_tr:k+TD-1+N_tr)';
        end

        % Desired signal
        T = zeros(N_tr,1);
        for ii=1:N_tr
            T(ii) = u(D+ii);
        end

        T_te = zeros(N_te,1);
        for ii=1:N_te
            T_te(ii) = u(D+ii+N_tr);
        end
        %======end of data===========

        %=========Linear LMS===================
        %learning rate (step size)
        lr = .005;%learning rate
        w1 = zeros(1,TD);
        e_l = zeros(N_tr,1);
        b_l = 0;
        for n=1:N_tr
            y = w1*X(:,n) + b_l;
            e_l(n) = T(n) - y;
            w1 = w1 + lr*e_l(n)*X(:,n)';
            b_l = b_l + lr*e_l(n);
        end
        %testing
        uhat = 2*(w1*X_te + b_l>0)-1;
        err_l(mc) = length(find(T_te'-uhat))/N_te;
        %=========end of Linear LMS================

        %=========Kernel LMS===================
        lr_k = .1;
        %init
        e_k = zeros(N_tr,1);
        b_k = 0;
        y = zeros(N_tr,1);

        % n=1 init
        e_k(1) = T(1);
        y(1) = 0;
        % start
        for n=2:N_tr
            %training
            ii = 1:n-1;
            y(n) = lr_k*e_k(ii)'*(exp(-sum((X(:,n)*ones(1,n-1)-X(:,ii)).^2)*h))'+b_k;
            e_k(n) = T(n) - y(n);
            b_k = b_k + lr_k*e_k(n);
        end
        %testing
        y_te = zeros(N_te,1);
        for jj = 1:N_te
            ii = 1:n;
            y_te(jj) = lr_k*e_k(ii)'*(exp(-sum((X_te(:,jj)*ones(1,n)-X(:,ii)).^2)*h))'+b_k;
        end
        uhat = 2*(y_te>0)-1;
        err_k(mc) = length(find(T_te - uhat))/N_te;
        %=========end of Kernel LMS================

        %========RN===================
        lam = 1;
        G = zeros(N_tr,N_tr);
        for i=1:N_tr-1
            j=i+1:N_tr;
            G(i,j)=exp(-sum((X(:,i)*ones(1,N_tr-i)-X(:,j)).^2)*h);
            G(j,i)=G(i,j)';
        end
        G = G + eye(N_tr);
        G_lam =G + lam*eye(N_tr);
        a = inv(G_lam)*T;

        %testing
        for jj = 1:N_te
            y_te(jj) = a'*(exp(-sum((X_te(:,jj)*ones(1,N_tr)-X).^2)*h))';
        end
        uhat = 2*(y_te>0)-1;
        err_rbf(mc) = length(find(T_te - uhat))/N_te;
        %=========end of RN================
    end%mc

    disp('========================================')
    disp(['noise deviation = ',num2str(np)])
    disp('<<LMS')
    disp([num2str(mean(err_l)),'+/-',num2str(std(err_l))])
    disp('<<KLMS')
    disp([num2str(mean(err_k)),'+/-',num2str(std(err_k))])
    disp('<<RN')
    disp([num2str(mean(err_rbf)),'+/-',num2str(std(err_rbf))])

end