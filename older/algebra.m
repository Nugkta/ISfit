%syms w c_a c_b c_g R_i V real;

%z_d = (1/impedance(w,c_g) + 1 / R_i);

%z_a = impedance(w,c_a);

%z_b = impedance(w, c_b);

%z_ion = z_d + z_a + z_b;

%v1 = V * (1 - z_a/z_ion);

%a = simplifyFraction(v1)

%imag(a)
syms c_ion c_g w R_ion real;

%z_d = (1/impedance(w,c_g) + 1 / R_i);

%z_a = impedance(w,c_a);

%z_b = impedance(w, c_b);

z_ion = 1/(1j * w * c_ion) + 1/(1j *w *c_g + 1/R_ion);

%v1 = V * (1 - z_a/z_ion);

a = simplifyFraction(z_ion)

imag(a)



function z = impedance(w, c)

z = 1/(1j * w * c);

end

