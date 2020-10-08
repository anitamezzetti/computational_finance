% Take Home Exam 1 (Mezzetti Anita)
% Exercise 4c:
clear all
close all
clc

% given parameters
s = 1;        % initial price
r = 0.1;      % rf interest rate
T = 0.5;      % maturity
K = 0.9;      % strike price
sigma = 0.1;  % volatility
b = 1.3;      % barrier

% MC
% plot the constant MC price computed by MCpriceBarrierUODM 
Ntime = 100;   % steps
Nsim = 1.0e6;   % # simulations
rng default     % default seed

mc_price = MCpriceBarrierUODM(r,sigma,Ntime,Nsim,T,s,K,b);
display(mc_price)

% Binomial prices
N = [2:2:200];   % steps

bin_prices = zeros(length(N),1);        % initialisation
for i = 1:length(N)
    u = 1+r*T/N(i)+sigma*sqrt(T/N(i));  % up 
    d = 1+r*T/N(i)-sigma*sqrt(T/N(i));  % down 
    bin_prices(i) = BinomialpriceBarrierUODM(r,d,u,N(i),T,s,K,b);
end


% plots
figure
plot(N,mc_price*ones(1,length(N)),'g')
hold on
plot(N,bin_prices,'b')
title('Exercise 4c')
xlabel('N')
ylabel('Option Price')
legend('Monte Carlo','Binomial Process')

