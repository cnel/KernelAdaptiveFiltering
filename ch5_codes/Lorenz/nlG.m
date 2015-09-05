function out = nlG(input,param,flag)

switch flag
    case 0
        out = param*input;
    case 1
        out = (1-exp(-param*input))./(1+exp(-param*input));
    case 2
        out = (1-param)*input + param*input.^2;
    case 3
        out = sin(param*input);
    case 4  %threshold cut off 
        out = input;
        out(find(out>param)) = param;
        out(find(out<-param)) = -param;
    otherwise
        warning('nlG');
end
return