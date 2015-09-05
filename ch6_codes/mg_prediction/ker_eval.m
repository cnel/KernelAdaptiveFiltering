function y = ker_eval(X1,X2,ker_type,ker_param)

N1 = size(X1,2);
N2 = size(X2,2);

if strcmp(ker_type,'Gauss')
    if N1 == N2
        y = (exp(-sum((X1-X2).^2,1)*ker_param))';
    elseif N1 == 1
        y = (exp(-sum((X1*ones(1,N2)-X2).^2,1)*ker_param))';
    elseif N2 == 1
        y = (exp(-sum((X1-X2*ones(1,N1)).^2,1)*ker_param))';
    else
        warning('error dimension--')
    end
end
if strcmp(ker_type,'Poly')
    if N1 == N2
        y = ((1 + sum(X1.*X2)).^ker_param)';
    elseif N1 == 1
        y = ((1 + X1'*X2).^ker_param)';
    elseif N2 == 1
        y = ((1 + X2'*X1).^ker_param)';
    else
        warning('error dimension--')
    end
end
if strcmp(ker_type,'CO2')
    
%     if N1 == N2
%         y = (66^2*exp(-(X1-X2).^2/(2*67^2)) + ...
%             2.4^2*exp(-(X1-X2).^2/(2*90^2) - 2*(sin(pi*(X1-X2))).^2/(1.3^2))+...
%             0.66^2*(1+(X1-X2).^2/(2*0.78*1.2^2)).^(-0.78) +...
%             0.18^2*exp(-(X1-X2).^2/(2*0.1333^2)) + 0.19^2*(X1 == X2))';
%     elseif N1 == 1
%         y = (66^2*exp(-(X1*ones(1,N2)-X2).^2/(2*67^2)) + ...
%             2.4^2*exp(-(X1*ones(1,N2)-X2).^2/(2*90^2) - 2*(sin(pi*(X1*ones(1,N2)-X2))).^2/(1.3^2))+...
%             0.66^2*(1+(X1*ones(1,N2)-X2).^2/(2*0.78*1.2^2)).^(-0.78) +...
%             0.18^2*exp(-(X1*ones(1,N2)-X2).^2/(2*0.1333^2)) + 0.19^2*(X1*ones(1,N2) == X2))';
%     elseif N2 == 1
%         y = (66^2*exp(-(X1-X2*ones(1,N1)).^2/(2*67^2)) + ...
%             2.4^2*exp(-(X1-X2*ones(1,N1)).^2/(2*90^2) - 2*(sin(pi*(X1-X2*ones(1,N1)))).^2/(1.3^2))+...
%             0.66^2*(1+(X1-X2*ones(1,N1)).^2/(2*0.78*1.2^2)).^(-0.78) +...
%             0.18^2*exp(-(X1-X2*ones(1,N1)).^2/(2*0.1333^2)) + 0.19^2*(X1 == X2*ones(1,N1)))';
%     else
%         warning('error dimension--')
%     end
   if N1 == N2
        y = (4356*exp(-(X1-X2).^2*1.1138e-004) + ...
            5.76*exp(-(X1-X2).^2*6.1728e-005 - (sin(pi*(X1-X2))).^2*1.1834)+...
            0.4356*(1+(X1-X2).^2*0.4452).^(-0.78) +...
            0.0324*exp(-(X1-X2).^2*28.1391) + 0.0361*(X1 == X2))';
    elseif N1 == 1
        y = (4356*exp(-(X1*ones(1,N2)-X2).^2*1.1138e-004) + ...
            5.76*exp(-(X1*ones(1,N2)-X2).^2*6.1728e-005 - (sin(pi*(X1*ones(1,N2)-X2))).^2*1.1834)+...
            0.4356*(1+(X1*ones(1,N2)-X2).^2*0.4452).^(-0.78) +...
            0.0324*exp(-(X1*ones(1,N2)-X2).^2*28.1391) + 0.0361*(X1*ones(1,N2) == X2))';
    elseif N2 == 1
        y = (4356*exp(-(X1-X2*ones(1,N1)).^2*1.1138e-004) + ...
            5.76*exp(-(X1-X2*ones(1,N1)).^2*6.1728e-005 - (sin(pi*(X1-X2*ones(1,N1)))).^2*1.1834)+...
            0.4356*(1+(X1-X2*ones(1,N1)).^2*0.4452).^(-0.78) +...
            0.0324*exp(-(X1-X2*ones(1,N1)).^2*28.1391) + 0.0361*(X1 == X2*ones(1,N1)))';
    else
        warning('error dimension--')
    end

end
    
return
