import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import simpson

As, ns, kstar = 2.1e-9, 0.964, 0.05                                      #kstar in Mpc^-1
lam_comoving = 1.5                                                       #lam_comov(at z = 1100) = (c/H0) * (1/dimensionless tau(at z = 1100)) * (1+z)(at z = 1100); dimensionless tau is derived in paper; Mpc
dEta = lam_comoving                                                      #Mpc

u = np.linspace(np.log(1.0e-5/dEta), np.log(1.0e+2/dEta), 4000)          #4000 pts.
k = np.exp(u)
f = As * (k/kstar)**(ns-1) * spherical_jn(2, k*dEta)**2
C2 = 4*np.pi/25 * simpson(f, x=u)
a2m = np.sqrt(C2)
print("C2 = ", C2)
print("a2m = ", a2m)
