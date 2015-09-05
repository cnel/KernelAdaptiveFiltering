function ker_width=silverKerWidth(x) % silverman's rule for kernel size
sig = std(x);

sx = sort(x);

q3 = median(sx(ceil(length(sx)/2):end));

q1 = median(sx(1:floor(length(sx)/2)));

ker_width = 0.9*min(sig,(q3-q1)/1.34)*length(x)^-0.2;