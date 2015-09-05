function [h,p,gp,Pspk_x]=kernel(x,mu,sigma,w,spk)
% kernel smoothing weighted (w) particles (x) with kernel centered at mu
% and kernel size (sigma). spk is the spike train in case to calculated the
% spike triggered x distribution
% output h -- histogram
%        p -- pdf
% x is m*1 sample column vector at time k; w is m*1 weight column vector at time k
                            % p is posterior density of all samples at time k, m*1 column vector
                            % gp is the Gassian matrix, m by n
if nargin<5
    m=length(x); % x has m intereted points, m by 1
    n=length(mu); %mu,w has n samples as column, n by 1
    if n<=1e4
        X=x*ones(1,n);
        Mu=ones(m,1)*mu';
        if prod(size(sigma))~=1
            sigma=ones(m,1)*sigma';
        end
        gp=normpdf(X,Mu,sigma);%/normpdf(1,1,(sigma));
        h=gp*w;
        p=h/sum(w);%divide by N samples if w are all ones.
    else
        matrixN=floor(n/1e4);
        gp=[];
        X=x*ones(1,1e4);
        for i=1:matrixN
            Mu=ones(m,1)*mu((i-1)*1e4+[1:1e4])';
            if prod(size(sigma))~=1
                sigma=ones(m,1)*sigma((i-1)*1e4+[1:1e4])';
            end
            gp=[gp,normpdf(X,Mu,sigma)];
        end
        h=sum(gp,2);
        p=mean(gp,2);
    end
else
    m=length(x); % x has m intereted points, m by 1
    n=length(mu); %mu,w has n samples as column, n by 1
    if n<=1e4
        X=x*ones(1,n);
        Mu=ones(m,1)*mu';
        if prod(size(sigma))~=1
            sigma=ones(m,1)*sigma';
        end
        gp=normpdf(X,Mu,sigma);%/normpdf(1,1,(sigma));
        h=gp*w;
        p=h/sum(w);%divide by N samples if w are all ones.
        Pspk_x=gp*spk;
    else
        matrixN=floor(n/1e4);
        gp=zeros(m,1e4);
        Pspk_x=zeros(m,size(spk,2));
        X=x*ones(1,1e4);
        for i=1:matrixN
            Mu=ones(m,1)*mu((i-1)*1e4+[1:1e4])';
             if prod(size(sigma))~=1
                sigma=ones(m,1)*sigma((i-1)*1e4+[1:1e4])';
             end
            tmp=normpdf(X,Mu,sigma);
            gp=gp+tmp;
            Pspk_x=Pspk_x+tmp*spk((i-1)*1e4+[1:1e4],:);
        end
        h=sum(gp,2);
        p=mean(gp,2)/matrixN;
        Pspk_x=sum(Pspk_x,2);
    end
end

    
