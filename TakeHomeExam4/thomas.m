function [x,L,U] = thomas(e,a,c,b)

% function [x,L,U] = thomas(e,a,c,b)
%
% solves Ax = b with thomas algorhytm, A being a tridiagonal matrix where 
%       -> e = sub diagonal ( e == diag(A,-1) ) 
%       -> a = main diagonal ( a == diag(A) )
%       -> c = super diagonal ( c == diag(A,1) ) 


N=length(a);
alfa=zeros(N,1);
delta=zeros(N-1,1);

% build coefficients for the L, U decomposition
alfa(1)=a(1);

for i=2:N
      delta(i-1)=e(i-1)/alfa(i-1);
      alfa(i)=a(i)-delta(i-1)*c(i-1);
end

% here are L,U
L=diag(ones(N,1), 0) + diag(delta, -1);
U=diag(alfa,0) + diag(c,1);

% forward sobstitution solution for Ly=b
y=zeros(N,1);
y(1)=b(1);

for i=2:N
      y(i)=b(i)-delta(i-1)*y(i-1);
end

% backward sobstituion for Ux=y
x=zeros(N,1);
x(N)=y(N)/alfa(N);

for i=N-1:-1:1
      x(i)=(y(i)-c(i)*x(i+1))/alfa(i);
end
