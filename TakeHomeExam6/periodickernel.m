function [cov, dk_dsigma, dk_dl, dk_dp] = ...
    periodickernel(XN,XM,sigma,lengthscale,p)

l = lengthscale;

sin_fun = sin(pi * pdist2(XN,XM) / p);
exp_fun = exp(-(2 / (l^2)) * (sin_fun.^2));

cov = (sigma^2) * exp_fun;

dk_dsigma = 2 * sigma * exp_fun;
dk_dl = sigma^2 * (4*(sin_fun.^2)/(l^3)) * exp_fun;
dk_dp = -4 * (sigma / (l * p))^2 * pi * pdist2(XN,XM) * exp_fun;

end
