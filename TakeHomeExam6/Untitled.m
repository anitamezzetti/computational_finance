 clear all
clc

X = rand(3,4);
mu = zeros(size(X));

%for squared exponential kernel
% theta=[1,0.3];
% [cov, dk_dsigma, dk_dl] = sqrdexp(X',X',theta(1),theta(2))
% 
% % for linear kernel
% theta=[1,0.3,0.5];
% [cov, dk_dsigma0, dk_dsigma1, dk_dc] = linearkernel(X',X',theta(1),theta(2),theta(3))

% % for linear kernel1
theta=[1,3,pi];
[cov, dk_dsigma, dk_dl, dk_dp] = periodickernel(X',X',theta(1),theta(2),theta(3))
% %
% numsamp=5;
% Y1=mvnrnd(mu,Sigma1,numsamp);
% 
% for i =1:numsamp
%     plot(X,Y1(i,:))
%     hold on
% end
% plot(X,mu')
% title('samples functions')
% hold off
