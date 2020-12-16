function [x,fval,eflag,output] = ...
    fitGPR(X, y, K_type, theta0, bound_theta, x_star)
% input:
% X: Training input X, where each row x corresponds to one input case.
% y: Training output vector corresponding to each row of the input matrix X.
% K_type:  One of the three kernels defined in part (a)
% theta0: initial parameters of the kernel
% bound_theta = bounds on the parameters of the kernel
% x_star: test set

% [max_lik, theta_opt, m_star, K_post]
% output: 
% max_lik: maxima of marginal likelihood
% theta_opt: optimal hyperparameters
% m_star: posterior mean
% K_post: posterior covariance

% sigma: y \sim N(0, K(X,X)+sigma^2*I)
var_y = var(y);
sigmaa = var_y;
 
% define inputs for fmicon function:
A = [];
b = [];
Aeq = [];
beq = [];
lb = bound_theta(1,:);
ub = bound_theta(2,:);
nonlcon = [];
options = optimoptions('fmincon','SpecifyObjectiveGradient',true);

% define the proper kernel function
if strcmp(K_type,'squaredexponential')

    fun = @(theta) to_minimize_square(theta, X, y, sigmaa);
    
    [x,fval,eflag,output] = fmincon(fun,theta0,A,b,Aeq,beq,lb,ub,nonlcon,options);

elseif strcmp(K_type,'linear')
    
elseif strcmp(K_type,'periodic')
    
end

[x,fval,eflag,output] = fmincon(fun,theta0,A,b,Aeq,beq,lb,ub,nonlcon,options);
% x = 0;
% fval = 0;
% eflag = [];
% output = 0;

end


function [f,g] = to_minimize_square(theta, X, y, sigmaa)

len_X = size(X,1); % number of rows of X

K = sqrdexp(X, X, theta(1), theta(2), 1);
dK = sqrdexp(X, X, theta(1), theta(2), 0);

K_sigma = K + sigmaa^2*ones(size(K));
inv_K_sigma = inv(K_sigma);

if det(K_sigma)==0
    fprintf("No det K_sigma is zero")
end

log_p = -0.5 * y' * inv_K_sigma * y ...
    - 0.5 * log(det(K_sigma)) - 0.5 * len_X * log(2*pi);
f = -log_p; % Calculate objective f

grad_sigma0 = dtheta_log_p(dK(1:len_X,:), y, inv_K_sigma);
grad_l = dtheta_log_p(dK(len_X+1:end,:), y, inv_K_sigma);
% gradient required
g =[grad_sigma0, grad_l];

end

function[dtheta]=  dtheta_log_p(dK, y, inv_K_sigma)
% gradient function

dtheta = - 0.5 * y' * inv_K_sigma * dK * inv_K_sigma...
     * y - 0.5 * trace(inv_K_sigma * dK);
end 