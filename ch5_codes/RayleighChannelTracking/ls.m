function [W,y,e,R,P]=ls(x,d,alpha) % least squre algorithm
[L,T]=size(x); % L-dimension, T-time duration
[N,T]=size(d);
W=zeros(L,N);
R=zeros(L,L);

%     for i=1:L
%         for j=1:i
%             R(i,j)=mean(x(i,:).*x(j,:));
%         end
%         P(i)=mean(d(k,:).*x(i,:));
%     end
%     R=R+R'-diag(diag(R));
    R=x*x'/T; % auto correlation matrix
    P=x*d'/T; % cross correlation matrix
    if nargin < 3
        alpha=0;
    end
    W=inv(R+alpha*eye(size(R)))*P; %W- transfer function / weight



y=W'*x; % prediction
e=d-y; % error