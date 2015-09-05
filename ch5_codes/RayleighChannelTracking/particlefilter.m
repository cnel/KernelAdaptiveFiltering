
function [ChannelExp,ChannelMle,Xrecord]=particlefilter(channel0,targetInput,targetOutput,Q,R,xn,F,paramNonlinear,typeNonlinear)

u=targetInput;
z=targetOutput;

x0=channel0;
[ChannelExp,ChannelMle,Xrecord]=SIR(x0,u,z,Q,R,F,paramNonlinear,typeNonlinear); % SIR particle filter
% %draw(X,p,0); % draw all particles, using linear gray colormap
% [XS1,p]=draw(X,1,W,sigma); % get the estimate state by maxlikehood (option 1)
% XS2=draw(X,2,W,sigma); % get the estimate state by expectation (option 2)
% % H=draw(X,3,W,sigma);
% pdfobserv(X,p,XS1,XS2,10,xr,X0);





