y_heston_test = heston_test(Heston_test, testing_size);


function[Heston_price_test] = heston_test(Heston_test, testing_size)

Heston_price_test=zeros(testing_size,1);

for i = 1:testing_size
    strike=Heston_test(i,1);
    T=Heston_test(i,2);
    r=Heston_test(i,3);
    kappa=Heston_test(i,4);
    theta=Heston_test(i,5);
    rho=Heston_test(i,6);
    sigma=Heston_test(i,7);
    v0=Heston_test(i,8);
    S=1;
    alpha=2;
    Heston_price_test(i,1)=FFT_Heston(kappa,theta,sigma,rho,r ,v0,S,strike,T,alpha);
end

end