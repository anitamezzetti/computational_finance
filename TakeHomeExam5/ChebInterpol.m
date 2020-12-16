function [I_N] = ChebInterpol(f,x,n,a,b)
% ChebInterpol Implement Chebyshev interpolation
% Input:
%   f: function to interpolate (function handle)
%   x: points to be interpolated
%   n: number of Chebyshev points
%   [a,b]: interval for parameters (optional)
% Output: 
%   I_N: interpolated function values

% Offline

% define Chebyshev points
k = 0:n;
pp = cos(pi*k/n);

% linear tranformation to original intervals
pp = (b-a)/2*(pp+1)+a;

% weights vector initialization
c = zeros(1,n+1);

% function values in Chebyshev points
Cp = zeros(n+1,1);
for i =1:n+1
    Cp(i) = f(pp(i));
end

% interpolation coefficients
S = diag([0.5; ones(n-1,1); 0.5]); % auxiliary matrix
for j = 0:n
    Tp = cos(j*pi*k/n); % Chebyshev polynomials
    c(j+1) = 2^(j>0)/n*(Tp*S*Cp);
end

% Online

% linear tranformation from original interval to [-1,1]
x = 2*(x-a)/(b-a)-1;

% output vector initialization
I_N = zeros(length(x),1);

% interpolated price
for i = 1:length(x)
	T = cos(k*acos(x(i)));
	I_N(i) = dot(c,T);
end

end
