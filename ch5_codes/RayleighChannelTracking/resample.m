function [X,W,In]=resample(x,w)

% resampling

% input: x- m*D sample column vector at time k; 
%        w- m*1 weight column vector at time k;
% input: X- m*D sample column vector at time k after resampling;
%        W- m*1 weight column vector at time k after resampling;
[m,D]=size(x);
X=zeros(m,D);
W=zeros(m,1);
In=zeros(m,1);
c=cumsum(w);  % cdf
i=1;
u1=rand(1)/m; % draw u1 uniformly distribute in [0, 1/N]
u=zeros(m,1);
for j=1:m
    u(j)=u1+(j-1)/m;
    while (u(j)>c(i) && i<m)
        i=i+1;
    end
    X(j,:)=x(i,:);
    W(j)=1/m;
    In(j)=i;
end
