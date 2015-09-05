function [Xexp,Xmle,Xrecord]=SIR(x0,u,z,Q,R,F,paramNonlinear,typeNonlinear)

[xn,D]=size(x0);
T=length(z);

X=x0;
XA=X;
Xrecord=[reshape(x0,xn*D,1),zeros(xn*D,T-1)];
Xexp=zeros(D,T);
Xmle=zeros(D,T);


%compute
for k=2:T
    if(rem(k/T,.2)==0)
        disp([num2str(k/T*100),'%...'])
    end
    XA=genesam(X,F,Q); % generate new approximate samples
    WA=normw(XA,u(:,k),z(k),R,paramNonlinear,typeNonlinear);   % get normalized approximate weight
    [X,W]=resample(XA,WA); % resample to reduce degeneracy
    Xrecord(:,k)=reshape(X,xn*D,1);
    Xexp(:,k)=mean(X);
    Xmle(:,k)=MLE(X,W);
    XA=X;
end


