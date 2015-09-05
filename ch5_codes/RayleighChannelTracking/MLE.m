function [Pmle,p,sigma]=MLE(x,w) % MLE estimation
%x is xn by D, w is xn by 1, Pmle is 1 by Dim, sigma by silverKerWidth
[xn,D]=size(x);
sigma=zeros(1,D);
Pmle=zeros(1,D);
p=zeros(xn,D); %smoothed pdf by kernel, xn by D
for i=1:D
    sigma(i)=silverKerWidth(x(:,i));  % Silver Kernel Width
    [h,p(:,i)]=kernel(x(:,i),x(:,i),sigma(i),w); % kernel smooth
end
[PM,IM]=maxp(p); % find the max pdf and index for each column of p
Pmle=x(IM);
