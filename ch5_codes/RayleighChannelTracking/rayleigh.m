
function r = rayleigh(T_s,Ns,F_d) 

% Generates a Rayleigh fading channel  
%  
% T_s: symbol period 
% Ns : number of symbols 
% F_d: maximum doppler spread. It can be a scalar or a vector.
%      If F_d is a scalar, then it corresponds to a constant 
%      mobile speed over the simulation time. If F_d is a vector
%      whose  length is equal to Ns, then it corresponds to a mobile 
%      speed that is varying over the simulation time. 
%

N = 20;           % Assumed number of scatterers 
 
 if (max(size(F_d))==1) 
    f = (T_s*F_d)*[0:Ns-1]; 
 else 
    f = (T_s*F_d).*[0:Ns-1]; 
 end 
 
 phi = (2*pi)*rand(1,N); 
 C = (randn(1,N)+i*randn(1,N))/sqrt(2*N); 
 r = zeros(1,Ns); 
 for j=1:N 
    r = r+exp(i*2*pi*cos(phi(j))*f)*C(j); 
 end 
