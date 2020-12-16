% Take Home 6 part c - Anita Mezzetti

clc
clear all

% define set size
training_size = 20;
testing_size = 50;

% create sets X, y, X*
[Heston_train, Heston_test] = create_X_sets(training_size, testing_size);
Heston_price_train = create_y_set(training_size, Heston_train);

fprintf('The size of the training X is %d*%d \n', size(Heston_train,1), size(Heston_train,2))
fprintf('The size of the training Y is %d*%d \n', size(Heston_price_train,1), size(Heston_price_train,2))
fprintf('The size of the test X is %d*%d \n', size(Heston_test,1), size(Heston_test,2))

% define input for fitGPR:
K_type = 'squaredexponential';
theta0 = [1 ,0.3];
bound_theta = [0.001, 0; 5, 10];
X = Heston_train;
y = Heston_price_train;
X_star = Heston_test;


fitGPR(X, y, K_type, theta0, bound_theta, X_star);


function [Heston_train, Heston_test] = create_X_sets(training_size, testing_size)

Heston_train = zeros(training_size,8);
Heston_test = zeros(testing_size,8);

% Training Set
strike= 0.4 + (1.6-0.4).*rand(training_size,1);
T= 11/12 + (1-11/12).*rand(training_size,1);
r= 0.015 + (0.025-0.015).*rand(training_size,1);
kappa= 1.4 + (2.6-1.4).*rand(training_size,1);
theta= 0.45 + (0.75-0.45).*rand(training_size,1);
rho= -0.75 + (-0.45+0.75).*rand(training_size,1);
sigma= 0.01 + (0.1-0.01).*rand(training_size,1);
v0= 0.01 + (0.1-0.01).*rand(training_size,1);

Heston_train(:,1)=strike(randperm(training_size));
Heston_train(:,2)=T(randperm(training_size));
Heston_train(:,3)=r(randperm(training_size));
Heston_train(:,4)=kappa(randperm(training_size));
Heston_train(:,5)=theta(randperm(training_size));
Heston_train(:,6)=rho(randperm(training_size));
Heston_train(:,7)=sigma(randperm(training_size));
Heston_train(:,8)=v0(randperm(training_size));

% Test Set
strike= 0.5 + (1.5-0.5).*rand(testing_size,1);
T= 11/12 + (1-11/12).*rand(testing_size,1);
r= 0.015 + (0.025-0.015).*rand(testing_size,1);
kappa= 1.5 + (2.5-1.5).*rand(testing_size,1);
theta= 0.5 + (0.7-0.5).*rand(testing_size,1);
rho= -0.7 + (-0.5+0.7).*rand(testing_size,1);
sigma= 0.02 + (0.1-0.02).*rand(testing_size,1);
v0= 0.02 + (0.1-0.02).*rand(testing_size,1);

Heston_test(:,1)=strike(randperm(testing_size));
Heston_test(:,2)=T(randperm(testing_size));
Heston_test(:,3)=r(randperm(testing_size));
Heston_test(:,4)=kappa(randperm(testing_size));
Heston_test(:,5)=theta(randperm(testing_size));
Heston_test(:,6)=rho(randperm(testing_size));
Heston_test(:,7)=sigma(randperm(testing_size));
Heston_test(:,8)=v0(randperm(testing_size));

end

function Heston_price_train = create_y_set(training_size, Heston_train)
% finding the prices in training set
Heston_price_train=zeros(training_size,1);

for i = 1:training_size
    strike=Heston_train(i,1);
    T=Heston_train(i,2);
    r=Heston_train(i,3);
    kappa=Heston_train(i,4);
    theta=Heston_train(i,5);
    rho=Heston_train(i,6);
    sigma=Heston_train(i,7);
    v0=Heston_train(i,8);
    S=1;
    alpha=2;
    Heston_price_train(i,1)=FFT_Heston(kappa,theta,sigma,rho,r ,v0,S,strike,T,alpha);
end

end

