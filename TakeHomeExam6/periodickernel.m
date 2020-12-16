function [result] = ...
    periodickernel(XN, XM, sigma0, lengthscale, p, return_k)

l = lengthscale;

sin_fun = sin(pi * pdist2(XN,XM) / p);
exp_fun = exp(-(2 / (l^2)) * (sin_fun.^2));

cov = (sigma0^2) * exp_fun;

if return_k == 1
    result = cov;
    return 
else 
    
    % gradient vector
    dk_dsigma0 = 2 * sigma0 * exp_fun;
    dk_dl = sigma0^2 * (4*(sin_fun.^2)/(l^3)) * exp_fun;
    dk_dp = -4 * (sigma0 / (l * p))^2 * pi * pdist2(XN,XM) * exp_fun;
    result = [dk_dsigma0; dk_dl; dk_dp];

end
