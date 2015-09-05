function [PM,IM]=maxp(p)  % p is N*T posterior density
[N,T]=size(p);
PM=zeros(1,T);
IM=zeros(1,T);
[PM,IM]=max(p);
IM=[0:(T-1)]*N+IM;