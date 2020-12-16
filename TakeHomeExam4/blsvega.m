function v = blsvega(so,x,r,t,sig,q) 
%BLSVEGA Black-Scholes sensitivity to underlying price volatility. 

%   V = BLSVEGA(SO,X,R,T,SIG,Q) returns the rate of change of the option value
%   with respect to the volatility of the underlying asset.  SO is the current
%   stock price, X is the exercise price, R is the risk-free interest rate, 
%   T is the time to maturity of the option in years, SIG is the standard 
%   deviation of the annualized continuously compounded rate of return of the
%   stock (also known as volatility), and q is the dividend rate.  
%   The default Q is 0.
%       
%   Note: This function uses normpdf, the normal probability
%         density function in the Statistics Toolbox.
% 
%   For example, v = blsvega(50,50,.12,.15,.3,0) returns v = 7.5522. 
% 
%   See also BLSPRICE, BLSDELTA, BLSGAMMA, BLSTHETA, BLSRHO, BLSLAMBDA. 
 
%       Copyright 1995-2010 The MathWorks, Inc.
 
%       Reference: Options, Futures, and Other Derivative Securities,  
%                  Hull, Chapter 13. 
 
if nargin < 5 
  error(message('finance:blsvega:missingInputs')) 
end 
if any(so <= 0 | x <= 0 | r < 0 | t <=0 | sig < 0) 
  error(message('finance:blsvega:invalidInputs')) 
end 
if nargin < 6 
   q = zeros(size(so)); % default dividend rate 
end 
 
blscheck(so, x, r, t, sig, q);

% Perform scalar expansion & guarantee conforming arrays.
try
    [so, x, r, t, sig, q] = finargsz('scalar', so, x, r, t, sig, q);
catch
    error(message('finance:blsvega:InconsistentDimensions'))
end

% blspriceeng works with columns. Get sizes, turn to columns, run engine,
% and finally turn to arrays again:
[m, n] = size(so);

% Double up on fcn calls since blsprice calculates both calls and puts. Do
% this only if nargout>1
NumOpt = numel(so);
OptSpec = {'call'};
OptSpec = OptSpec(ones(NumOpt,1));
OutSpec = {'vega'};
disp('ciao')
% call eng fuction
vega = blspriceeng(OutSpec, OptSpec(:), so(:), x(:), r(:), t(:), sig(:), q(:));
v=reshape(vega{1}, m, n);