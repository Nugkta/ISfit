params = [5e8 2.8e-8 2e-7 2e-7 7.1e-11 1.93 0];
w = logspace(-4 , 5, 1000);
Z = impedancemodel('one',w,params);
n =  1.93;
J1 = 5.1119e-10;;
VT = 0.026;
C_a = 2e-7;
C_b = 2e-7;
C_g = 2.8e-8;
r_reci = n * VT/ J1 ;
r_rec0 = n *VT /(J1) * (C_a+C_b)/C_a ;
ceff = (C_a^(-1)+C_b^(-1)+C_g^(-1))^(-1)




