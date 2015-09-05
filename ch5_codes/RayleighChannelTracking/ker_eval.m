function y = ker_eval(X1,X2,ker_type,ker_param)

N1 = size(X1,2);
N2 = size(X2,2);

if strcmp(ker_type,'Gauss')
    if N1 == N2
        y = (exp(-sum((X1-X2).^2)*ker_param))';
    elseif N1 == 1
        y = (exp(-sum((X1*ones(1,N2)-X2).^2)*ker_param))';
    elseif N2 == 1
        y = (exp(-sum((X1-X2*ones(1,N1)).^2)*ker_param))';
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
return
