function w=normw(x,u,z,r,paramNonlinear,typeNonlinear)  
% get normalized weight xn*1 column vector at time k
% x is xn*D sample; u is D*1 colum vector, system input.
% z is 1*1 observation.
[xn,D]=size(x);
if isempty(paramNonlinear)==1 && isempty(typeNonlinear)==1
       w=normpdf(z*ones(xn,1),x.^2/20,sqrt(r)*ones(xn,1)); %p(zk|xk(i))~Gaussian
else
     w=normpdf(z*ones(xn,1),nlG((x*u),paramNonlinear,typeNonlinear),sqrt(r)*ones(xn,1)); %p(zk|xk(i))~Gaussian
end

if sum(w)==0
    w=1/xn*ones(xn,1);
 %   disp('warning: weights are all zeros');
else
    w=w/sum(w);
end
