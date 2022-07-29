function Z = impedancemodel(transistorno,w,params)

%transistor = 'one', 'two', 'four': number of transistors in circuit model
%w: frequencies to evaluate impedance
%params: contains array of parameters

%Constants
VT = 0.026; %thermal voltage


%ionic circuit branch, define elements
Zcap = @(w,C) 1./(1i*w*C); %impedance of a capacitor
Zg = @(w,R,C) 1./(1/R + 1i*w*C);    %impedance of bulk geometric capacitance in parrallel with ionic resitance

%--------------------------------------------------------------------------
%one transistor model
if transistorno == 'one'

%unpack the variables for one transistor circuit model
R = params(1);  %ionic resistance
Cg = params(2); %geometric bulk capacitance
CA = params(3); %Capacitance of interface A
CB = params(4); %Capacitance of remaining interfaces
JsA = params(5);%Saturation current density of interface A
nA = params(6); %ideality factor of interface A 
V = params(7);  %steady state voltage applied to device

%define impedance of the ionic circuit branch    
Zion = @(w,R,CA,CB,Cg) Zcap(w,CA) + Zcap(w,CB) + Zg(w,R,Cg);

%determine the steady state voltage at interface A
VA = V*CA/(CA + CB);

%find the steady-state current density through the electronic branch based
%on the interface voltage VA
JA = JsA*(exp(VA/(nA*VT)) - exp((VA - V)/(nA*VT)));
%split into the forward and backward components
Jrec = JsA*exp(VA/(nA*VT));
Jgen = JsA*exp((VA - V)/(nA*VT));

%define the impedance prefactor where voltage VA = V*(1-A)
A = Zcap(w,CA)./Zion(w,R,CA,CB,Cg)

%find the electronic impedance by differentiating JA = JsA*exp(V*(1-A)/(nA*VT)) - JsA*exp(V*(1-A-1)/(nA*VT)) with respect to V then inverting;
ZA = 1./((1 - A)*Jrec/(nA*VT) + A*Jgen/(nA*VT));
Z = 1./(1./ZA + 1./Zion(w,R,CA,CB,Cg));
Zi = Zion(w,R,CA,CB,Cg);


%--------------------------------------------------------------------------
%two transistor model
elseif transistorno == 'two'
% model assumes transistor behaviour at interface A/B and C/D. Can also
% assume a non zero photogeneration current in the device.

%unpack the variables for two transistor circuit model
% params = [R Cg CA CB CC JsA JsC nA nC V Jph];
R = params(1);  %ionic resistance
Cg = params(2); %geometric bulk capacitance
CA = params(3); %Capacitance of interface A
CB = params(4); %Capacitance of perovskite space charge layer at interface A
CC = params(5); %Combined capacitance of the other interface
JsA = params(6);%Saturation current density of interface A/B
JsC = params(7);%Saturation current density of barrier C/D
nA = params(8); %ideality factor of interface A/B
nC = params(9); %ideality factor of barrier C/D
V = params(10); %steady state voltage applied to device
Jph = params(11);%photogeneration current density (negative number)


%define impedance of the ionic circuit branch    
Zion = @(w,R,CA,CB,CC,Cg) Zcap(w,CA) + Zcap(w,CB) + Zg(w,R,Cg) + Zcap(w,CC);

%find the steady state voltage at interface A and before the barrier at the other
%interface
VA = V*(1 - CB*CC/(CA*CB + CB*CC + CA*CC));
VC = V*CA*CB/(CA*CB + CB*CC + CA*CC);
    
%find the steady state current through the electronic branch of the circuit
%given the voltage of the interfaces VA and VC. If the ideality factor of
%both interfaces is the same, then analytical solution can be found,
%otherwise a numerical solution is required.

if nA == nC
    % analystical solution for Vn of: JsA*(exp((VA - Vn)/(nA*VT)) - exp((VA - V)/(nA*VT))) + Jph == JsC*(exp(VC/(nA*VT)) - exp((VC - Vn)/(nA*VT)))
    % Jrec + Jgen +Jph == Jinj - Jcol  assumes a negative photogeneration
    % current
    Vn = -VT*nA*log((JsC*exp(VC/(VT*nA)) - Jph + JsA*exp(-V/(VT*nA))*exp(VA/(VT*nA)))/(JsA*exp(VA/(VT*nA)) + JsC*exp(VC/(VT*nA))));
else
    % find numerical solution for Vn if interface ideality factors are not
    % equal
    rootfunction = @(Vn) JsA*(exp((VA - Vn)/(nA*VT)) - exp((VA - V)/(nA*VT))) + Jph - (JsC*(exp(VC/(nC*VT)) - exp((VC - Vn)/(nC*VT))));
    % find the root Vn where rootfunction -->0 assuming an initial guess
    % Vn = 0
    Vn = fzero(rootfunction,0);
end


%determine the steady state currents for each interfacial process
Jrec = JsA*exp((VA - Vn)/(nA*VT));
Jgen = JsA*exp((VA - V)/(nA*VT));
Jinj = JsC*exp(VC/(nC*VT));
Jcol = JsC*exp((VC - Vn)/(nC*VT));

%determine expressions for the impedance prefactors
% VA = V*(1-A)
A = Zcap(w,CA)./Zion(w,R,CA,CB,CC,Cg);
% VC = V*C
C = Zcap(w,CC)./Zion(w,R,CA,CB,CC,Cg);

%Determine the impedance prefactor for the electron quasi-Fermi level Vn where
% Bn = vn/v by solving the expression:
% v*(1 - A - Bn)*Jrec/nA + v*A*Jgen/nA == v*C*Jinj/nC - v*(C - Bn)*Jcol/nC
% for Bn
Bn = (Jrec*nC + A*Jgen*nC - A*Jrec*nC + C*Jcol*nA - C*Jinj*nA)/(Jcol*nA + Jrec*nC);

%evaluate the impedance of the electronic branch
Zelec = 1./((1 - A - Bn)*Jrec/nA + A*Jgen/nA);

%evaluate the total impedance
Z = 1./(1./Zion(w,R,CA,CB,CC,Cg) + 1./Zelec);

end

subplot(1,3,1)
plot(real(Z),-imag(Z))
subplot(1,3,2)
loglog(w,[real(Z);-imag(Z)])
%loglog(w,[real(Z);-imag(Z);imag(Z)]) %also plot negative values
subplot(1,3,3)
loglog(w,1./w.*imag(1./Z))
%loglog(w,[1./w.*imag(1./Z);-1./w.*imag(1./Z)]) %also plot negative values
