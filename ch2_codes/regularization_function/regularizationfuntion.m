%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Copyright Weifeng Liu 
%CNEL
%July 1, 2008
%
%description:
%illustrate the reg-function of KLMS
%one step prediction
%Learning curves
%
%Usage:
%ch2
%
%Outside functions called:
%none
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


close all
x= 0:0.05:10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%comparison among gradient descent, Tikhonov and PCA
N = 500;
eta = .1;
rf_LMS = 1-(1-eta*x.^2/N).^N;

lam =1;
rf_Tik =x.^2./(x.^2+lam);

a = 0.5;
rf_Tru = x>a;


figure(10),
lineWid = 3;
plot(x,rf_LMS,'k-','LineWidth', lineWid)
hold on
plot(x,rf_Tik,'k-.','LineWidth', lineWid);
% 
plot(x,rf_Tru,'k:','LineWidth', lineWid)
% 
% 
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
legend('KLMS','Tikhonov','PCA')
hold off
xlabel('singular value')
ylabel('reg-function')
grid on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%comparison on gradient descent with different step size
N = 500;

eta = .01;
rf_LMS_1 = 1-(1-eta*x.^2/N).^N;

eta = .1;
rf_LMS_2 = 1-(1-eta*x.^2/N).^N;

eta = 1;
rf_LMS_3 = 1-(1-eta*x.^2/N).^N;

figure(11),
lineWid = 3;
plot(x,rf_LMS_1,'k-','LineWidth', lineWid)
hold on
plot(x,rf_LMS_2,'k--','LineWidth', lineWid);
% 
plot(x,rf_LMS_3,'k-.','LineWidth', lineWid)
% 
% 
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
legend('\eta=.01','\eta=.1','\eta=1')
hold off
xlabel('singular value')
ylabel('reg-function')
grid on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%comparison on gradient descent with different data size
eta = .1;

N = 100;
rf_LMS_1 = 1-(1-eta*x.^2/N).^N;

N = 500;
rf_LMS_2 = 1-(1-eta*x.^2/N).^N;

N = 1000;
rf_LMS_3 = 1-(1-eta*x.^2/N).^N;

figure(12),
lineWid = 3;
plot(x,rf_LMS_1,'k-','LineWidth', lineWid)
hold on
plot(x,rf_LMS_2,'k--','LineWidth', lineWid);
% 
plot(x,rf_LMS_3,'k-.','LineWidth', lineWid)
% 
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');
legend('N = 100','N = 500','N = 1000')
hold off
xlabel('singular value')
ylabel('reg-function')
grid on

