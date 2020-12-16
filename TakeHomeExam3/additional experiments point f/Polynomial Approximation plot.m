
clc
clear all
close all

% initial conditions:
X0 = 0;
V0 = 0.04;

% parameters:
kappa = 0.5;
theta = 0.04;
sigma = 1;
r = 0;
rho = -0.5;
T = 1/12;
v_min = 1e-4;
v_max = 0.08;
k = -0.1;

% weighting gaussian distribution parameters
% GenJacobiCanonic function definded below
G = Generator_Canonic(kappa, theta, sigma, r, rho, v_min, v_max);
M = [1 V0 X0 V0^2 V0*X0 X0^2] * expm(T * G);

% mean and standard deviation:
mu_w = M(3);
sigma_w = sqrt(M(6) - M(3)^2);

% polynomial expansion price
N_pol = [1:2:70];          % approximation degree
price = zeros(1,length(N_pol));

for i = 1:length(N_pol)
    n = N_pol(i);
    price(i) = PriceApprox(n, V0, X0, kappa, sigma, theta, r, rho, ...
        T, v_min, v_max, mu_w, sigma_w, k);
end

plot(N_pol,price,'-*')
title('Polynomial Expansion Prices')
xlabel('Truncation Level N_pol')
ylabel('Price')

function [G] = Generator_Canonic(kappa, theta, sigma, r, rho, v_min, v_max)
% Build the matrix representation of the Jacobi model 
%   generator with canonic polynomial basis up to second degree

% matrix initialization
G = zeros(6,6);

% vector Q
D = (sqrt(v_max)-sqrt(v_min))^(-2);
Q = D*[-v_max*v_min; v_max+v_min; 0; -1; 0; 0];

% generator matrix
G(1:2,2) = [kappa * theta; -kappa];
G(1:2,3) = [r; -0.5];
G(:,4) = [0; 2 * kappa * theta; 0; -2*kappa; 0; 0] + sigma^2 * Q;
G(:,5) = [0; r; kappa * theta; -0.5; -kappa; 0] + rho * sigma * Q;
G(:,6) = [0; 1; 2 * r; 0; -1; 0];

end
