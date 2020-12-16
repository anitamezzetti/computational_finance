function [result] = linearkernel(XN, XM, sigma0, sigma1, c, return_k)

n = size(XN, 1);
m = size(XM, 1);

XN_XM = (XN-c)*(XM-c)';

cov = (sigma0^2)+(sigma1^2) * XN_XM;

if return_k == 1
    result = cov;
    return 
else 
    % partial derivatives
    dk_dsigma0 = 2 * sigma0 * ones(n,m);
    dk_dsigma1 = 2 * sigma1 * XN_XM;

    n = size(XN, 1);
    m = size(XM, 1);
    dk_dc = zeros(n,m);

    for i=1:n
        for j=1:m
            dk_dc(i,j) = -sum(XN(i,:)-XM(j,:)) + 2 * c;
        end
    end

    result = [dk_dsigma0; dk_dsigma1; dk_dc];

end

