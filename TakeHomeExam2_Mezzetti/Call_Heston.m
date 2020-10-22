function P = Call_Heston(K, T, r, nu, kappa, sigma, rho, S, V)
% Call_Heston:  Compute the value of call option using the formula in
% Heston[1993], see also formula (6) in Albrecher et Al.[2006].
%
% USAGE: P = Call_Heston(K, T, r, nu, kappa, sigma, rho, S, V, modif);
%
% PARAMETERS:
%    Input:
%         K: strike price of the call option
%         T: maturity of the call option
%         r: risk free rate
%         nu, kappa, sigma: parameters of the Heston model
%         rho: correlation parameter between the stock and vol processes
%         S, V: initial stock price and volatility
%    Output:
%        P: price of the call option

b1 = kappa-rho*sigma;
b2 = kappa;
u1 = 0.5;
u2 = -0.5;

x = log(S);
alpha = log(K); % log-strike 

integrand = @(u) S *...
            real(exp(-1i*u*alpha) .* exp(C_CF(u, T, r, nu, kappa, sigma, rho, b1, u1)...
            + V * D_CF(u, T, sigma, rho, b1, u1) + 1i*u*x) ./ (1i*u))...
            - K * exp(-r*T)... 
            * real(exp(-1i*u*alpha) .* exp(C_CF(u, T, r, nu, kappa, sigma, rho, b2, u2)...
            + V * D_CF(u, T, sigma, rho, b2, u2) + 1i*u*x) ./ (1i*u));
            
P = 0.5 * (S - K * exp(-r*T)) + 1/pi * quadgk(integrand, 0, 100);

end

function out = C_CF(u, t, r, nu, kappa, sigma, rho, bj, uj)

d = sqrt( (rho*sigma*u.*1i - bj).^2 - sigma^2 * (2*uj*u.*1i - u.^2) );
g = (bj - rho*sigma*u.*1i + d) ./ (bj - rho*sigma*u.*1i - d);

out = r*u*t.*1i + (nu * kappa) ./ sigma^2 * ...
    ( (bj - rho*sigma*u*1i + d) * t - 2*log( (1 - g.*exp(d*t)) ./ (1 - g)) );

end

function out = D_CF(u, t, sigma, rho, bj, uj)

d = sqrt( (rho*sigma*u.*1i - bj).^2 - sigma^2 * (2*uj*u.*1i - u.^2) );
g = (bj - rho*sigma*u.*1i + d) ./ (bj - rho*sigma*u.*1i - d);

out = (bj - rho*sigma*u*1i + d) ./ (sigma^2) .* ((1 - exp(d*t)) ./ (1 - g.*exp(d*t)));

end
