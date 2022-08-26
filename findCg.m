function [Cg, Cion] = findCg(wn,wR,wtheta,Rninf,k,lowlimit,highlimit)
% function to find Cg and Cion from the above function inputs (which are
% features in the impedance spectra.
%
% Function inputs
% wn = angular frequncy of the low frequency maximima of -Z'' in the
% Nyquist plot
% wR = angular frequency of the high frequency maxima of -Z'' in the
% Nyquist plot
% wtheta = angular frequency of the minima between circles in the Nyquist
% plot
% Rinf = width of the first circle
% k = ratio of high frequency recombination resistance to low frequency
% recombination resistance
% lowlimit =  lower boundary of Cg estimate
% highlimit = upper boundary of Cg estimate

% Function to find root
%targetfunc = @(Cg) abs(k./(wn./(Rninf.*wR - 1./Cg)).*sqrt(Cg.*(Cg + 1./(Rninf.*wR - 1./Cg)) - 1/wtheta));
targetfunc = @(Cg) (k./(wn./(Rninf.*wR - 1./Cg)).*sqrt(abs(Cg.*(Cg + 1./(Rninf.*wR - 1./Cg)) ))- 1/wtheta);

% plot the target function vs possible Cg values
Cg_ = logspace(-8,-2,10000);
loglog(Cg_,targetfunc(Cg_)+50)

%m = ones(10000).*5;
%loglog(Cg_,m)

% Find the root value of Cg where the above equation will equal zero, use
% starting
%Cg = fminbnd(targetfunc,lowlimit,highlimit,optimset('TolX',1e-12,'Display','off'));
Cg = fminbnd(targetfunc,lowlimit,highlimit,optimset('TolFun',1e-12,'Display','off'));

% Now use the value of Cg to find the value of Cion
Cion = 1./(Rninf*wR - 1/Cg)

%findCg( 0.008708431497690725,169.13295170296504,2.081221569986337,273805.40365415154,0.6,1e-8,1e-5)
%findCg(wn,wR,wtheta,Rninf,k,lowlimit,highlimit)