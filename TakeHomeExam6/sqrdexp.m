function [result] = sqrdexp(XN, XM, sigma0, lengthscale, return_k)
% Squared Exponential

l = lengthscale;
norm = pdist2(XN,XM).^2;
exp_fun = exp(-(norm)/(2*l^2));

cov = sigma0^2 * exp_fun;

if return_k == 1
    result = cov;
    return 
else
    % partial derivatives
    dk_dsigma = 2 * sigma0 * exp_fun;
    dk_dl = norm * (sigma0^2)/(l^3) * exp_fun;

    result = [dk_dsigma; dk_dl];
end

