function [psi] = BinomialpriceBarrierUODM(r,d,u,N,T,s,K,b)
% Binomial price at time t=0 of the up-and-down call barrier option 
% with discrete monitoring at two monitoring dates (see definition in pdf)

% s: initial stock price

% divide [0 T] in N equal lenght periods [t(k) t(k+1)]:
r_one = r*T/N;  % one period interest rate

% probabilities up and down:
Qu = (1+r_one-d)/(u-d);
Qd = (u-1-r_one)/(u-d);

psi = zeros(N+1,1); %creation empty vector of prices

% monitoring T:
for i = 0:N
    price_halfT = s*u^i*d^(N-i);
    if price_halfT < b && price_halfT > K
        psi(i+1) = price_halfT-K;
    else % payoff >= b or payoff<=K
        psi(i+1) = 0;
    end
end

% price between T and T/2
for k = (N-1):-1:N/2
    psi_k = zeros(k+1,1); 
    for i = 1:k+1
        psi_k(i) = (1/(1+r_one))*(Qu*psi(i+1) + Qd*psi(i));
        %display(psi_k(i))
    end
    psi = psi_k;
end

% monitoring T/2:
for i = 0:N/2
    price_halfT = s*u^i*d^(N-i);
    if price_halfT > b
        psi(i+1) = 0;
    end
end

% price between T/2 and 0
for k = (N/2-1):-1:0
    psi_k = zeros(k+1,1); 
    for i = 1:k+1
        psi_k(i) = 1/(1+r_one)*(Qu*psi(i+1) + Qd*psi(i));
    end
    psi = psi_k;
end

end

