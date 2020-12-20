function CallValue = FFT_Heston(kappa,theta,sigma,rho,r ,v0,S,strike,T,alpha)
% PARAMETERS:
%    Input:
%         strike: strike price of the call option
%         T: maturity of the call option
%         r: risk free rate
%         kappa, theta, rho, sigma, v0: parameters of the Heston model
%         S: initial stock price
%         alpha: damping factor
%    Output:
%        Call Value: price of the call option with strike K and maturity T

x0 = log(S);
N= 4096;
c = 600;
eta = c/N;
b =pi/eta;
u = [0:N-1]*eta;
lamda = 2*b/N;
position = (log(strike) + b)/lamda + 1; %position of call
v = u - (alpha+1)*1i;
zeta = -.5*(v.^2 +1i*v);
gamma = kappa - rho*sigma*v*1i;
PHI = sqrt(gamma.^2 - 2*sigma^2*zeta);
A = 1i*v*(x0 + r*T);
B = v0*((2*zeta.*(1-exp(-PHI.*T)))./(2*PHI - (PHI-gamma).*(1-exp(-PHI*T))));
C = -kappa*theta/sigma^2*(2*log((2*PHI -(PHI-gamma).*(1-exp(-PHI*T)))./ (2*PHI)) + (PHI-gamma)*T);
charFunc = exp(A + B + C);
ModifiedCharFunc = charFunc*exp(-r*T)./(alpha^2 ...
+ alpha - u.^2 + 1i*(2*alpha +1)*u);
SimpsonW = 1/3*(3 + (-1i).^[1:N] - [1, zeros(1,N-1)]);
FftFunc = exp(1i*b*u).*ModifiedCharFunc*eta.*SimpsonW;
payoff = real(fft(FftFunc));
CallValueM = exp(-log(strike)*alpha)*payoff/pi;
format short;
CallValue = CallValueM(round(position));
end
