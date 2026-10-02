#We use natural units (8Pi G = c = 1) in transfer + geodesic equations. Thus, length and time have same units.

import numpy as np
import sympy as sp
import scipy.integrate as integrate
import healpy as hp
import multiprocessing as mp
import time
import os
import sys
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.ticker as ticker
from scipy.integrate import quad
import camb

Omega_k = 0.044                                                                                             #Hardcoded from CMB constraints (Prokopec's paper)
Omega_m = 0.315                                                                                             #A value consistent with Planck
Omega_L = 1.0 - Omega_m - Omega_k                                                                           #(= 0.641); Friedmann constraint, Omega_m + Omega_L + Omega_k = 1
H0 = 2.56e-18                                                                                               #Hubble constant in seconds inverse

#Constructing optical depth using CAMB
C_MPC_PER_S = 9.7156e-15
def _build_tau_table(zmax=3000.0, n=4000):
    h = H0*3.0856776e19/100.0
    ombh2 = 0.023
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100.0*h, ombh2=ombh2, omch2=Omega_m*h**2 - ombh2, omk=Omega_k, tau=0.054)
    res = camb.get_background(pars)
    z = np.expm1(np.linspace(0.0, np.log1p(zmax), n))
    ev = res.get_background_redshift_evolution(z, ['x_e', 'opacity'], format='dict')
    rate = ev['opacity']*C_MPC_PER_S*(1.0 + z)
    return np.log1p(z), np.log(rate)

_LNZ, _LNRATE = _build_tau_table()

def tau_of_a(a):
    z = 1.0/a - 1.0
    return np.exp(np.interp(np.log1p(z), _LNZ, _LNRATE))

#Radial comoving distance (chi) placeholder for value at z = 1100 in all the geometries
def chi_of_a(a_target, a_start):
    integrand = lambda a: 1.0/(a**2 * H0*np.sqrt(Omega_m*a**-3 + Omega_k*a**-2 + Omega_L))                  #d(theta)/dt = d(phi)/dt = 0 at initial (z = 1100)
    val, _ = quad(integrand, a_start, a_target, limit=200)
    return val

def derivatives_hpc(a, y):
    y = np.array(y, dtype=float)
    H = H0*np.sqrt(Omega_m*(a**(-3)) + Omega_k*(a**(-2)) + Omega_L)

    #Respecting physical symmetry
    y[9] = y[11]                                                                                            #R0_12 = R0_21
    y[10] = y[14]                                                                                           #R0_13 = R0_31
    y[13] = y[15]                                                                                           #R0_23 = R0_32
    
    y[18] = y[20]                                                                                           #I0_12 = I0_21
    y[19] = y[23]                                                                                           #I0_13 = I0_31
    y[22] = y[24]                                                                                           #I0_23 = I0_32
    
    y[27] = y[29]                                                                                           #R2_12 = R2_21
    y[28] = y[32]                                                                                           #R2_13 = R2_31
    y[31] = y[33]                                                                                           #R2_23 = R2_32
    
    y[36] = y[38]                                                                                           #I2_12 = I2_21
    y[37] = y[41]                                                                                           #I2_13 = I2_31
    y[40] = y[42]                                                                                           #I2_23 = I2_32
    
    #Enforcing tracelessness -----------------------------------------------------------------------------------------------
    y[16] = -(y[8] + y[12])                                                                                 #R0_33 = -(R0_11 + R0_22)
    y[25] = -(y[17] + y[21])                                                                                #I0_33 = -(I0_11 + I0_22)
    y[34] = -(y[26] + y[30])                                                                                #R2_33 = -(R2_11 + R2_22)
    y[43] = -(y[35] + y[39])                                                                                #I2_33 = -(I2_11 + I2_22)
    
    R0_0, I0_0 = y[0], y[1]
    R0_1, R0_2, R0_3 = y[2], y[3], y[4]
    I0_1, I0_2, I0_3 = y[5], y[6], y[7]
    R0_11, R0_12, R0_13 = y[8], y[9], y[10]
    R0_21, R0_22, R0_23 = y[11], y[12], y[13]
    R0_31, R0_32, R0_33 = y[14], y[15], y[16]
    I0_11, I0_12, I0_13 = y[17], y[18], y[19]
    I0_21, I0_22, I0_23 = y[20], y[21], y[22]
    I0_31, I0_32, I0_33 = y[23], y[24], y[25]
    R2_11, R2_12, R2_13 = y[26], y[27], y[28]
    R2_21, R2_22, R2_23 = y[29], y[30], y[31]
    R2_31, R2_32, R2_33 = y[32], y[33], y[34]
    I2_11, I2_12, I2_13 = y[35], y[36], y[37]
    I2_21, I2_22, I2_23 = y[38], y[39], y[40]
    I2_31, I2_32, I2_33 = y[41], y[42], y[43]
    
    psi_real = y[44]
    psi_imag = y[45]
    theta = y[46]
    phi = y[47]
    chi = y[48]
    x_coord = y[49]
    y_coord = y[50]
    z_coord = y[51]
    
    K0 = -5/4                                                                                                #From Prokopec's paper
    k = 3*Omega_k*H0**2 / K0                                                                                #From Prokopec's eq. (3.15), k < 0 in UH2 geometry
    
    #Physical Thomson scattering rate (in seconds inverse)
    tau = tau_of_a(a)
    
    #Geodesic equations in physical time
    d_theta = 0
    d_phi = -np.sqrt(-k)*(np.sin(phi)*np.sin(theta)*np.tanh(x_coord*np.sqrt(-k)) + np.cos(theta))/a
    d_psi_real = -np.sqrt(-k)/(2*a)
    d_psi_imag = 0

    #Re_Zeta from Cramer's rule
    z_r0 = I0_0
    z_r1 = z_r0**2
    z_r2 = R0_0
    z_r3 = z_r2**2
    z_r4 = a*z_r1
    z_r5 = np.sqrt(-k)
    z_r6 = z_r5*np.tanh(x_coord*z_r5)/3
    #Re_Zeta = (-H*a*z_r3 - H*z_r4 + tau*z_r4 + z_r0*z_r6*I0_1 + z_r2*z_r6*R0_1)/(H*a*(z_r1 + z_r3))
    
    #Im_Zeta from Cramer's rule
    z_i0 = I0_0
    z_i1 = R0_0
    z_i2 = np.sqrt(-k)
    z_i3 = z_i2*np.tanh(x_coord*z_i2)
    #Im_Zeta = (3*a*tau*z_i0*z_i1 - z_i0*z_i3*R0_1 + z_i1*z_i3*I0_1)/(3*H*a*(z_i0**2 + z_i1**2))

    #Planck distrn. describes the radiation field
    Re_Zeta = 0.0012
    Im_Zeta = 0

    #Ricci rotation coefficients (spin connections) in physical units
    Gamma0_00 = 0
    Gamma0_01 = 0
    Gamma0_02 = 0
    Gamma0_03 = 0
    Gamma0_10 = 0
    Gamma0_11 = H
    Gamma0_12 = 0
    Gamma0_13 = 0
    Gamma0_20 = 0
    Gamma0_21 = 0
    Gamma0_22 = H
    Gamma0_23 = 0
    Gamma0_30 = 0
    Gamma0_31 = 0
    Gamma0_32 = 0
    Gamma0_33 = H
    Gamma1_00 = 0
    Gamma1_01 = 0
    Gamma1_02 = 0
    Gamma1_03 = 0
    Gamma1_10 = H
    Gamma1_11 = 0
    Gamma1_12 = 0
    Gamma1_13 = 0
    Gamma1_20 = 0
    Gamma1_21 = 0
    Gamma1_22 = -np.sqrt(-k)*np.tanh(x_coord*np.sqrt(-k))/a
    Gamma1_23 = -np.sqrt(-k)/(2*a)
    Gamma1_30 = 0
    Gamma1_31 = 0
    Gamma1_32 = -np.sqrt(-k)/(2*a)
    Gamma1_33 = 0
    Gamma2_00 = 0
    Gamma2_01 = 0
    Gamma2_02 = 0
    Gamma2_03 = 0
    Gamma2_10 = 0
    Gamma2_11 = 0
    Gamma2_12 = 0
    Gamma2_13 = np.sqrt(-k)/(2*a)
    Gamma2_20 = H
    Gamma2_21 = np.sqrt(-k)*np.tanh(x_coord*np.sqrt(-k))/a
    Gamma2_22 = 0
    Gamma2_23 = 0
    Gamma2_30 = 0
    Gamma2_31 = np.sqrt(-k)/(2*a)
    Gamma2_32 = 0
    Gamma2_33 = 0
    Gamma3_00 = 0
    Gamma3_01 = 0
    Gamma3_02 = 0
    Gamma3_03 = 0
    Gamma3_10 = 0
    Gamma3_11 = 0
    Gamma3_12 = -np.sqrt(-k)/(2*a)
    Gamma3_13 = 0
    Gamma3_20 = 0
    Gamma3_21 = np.sqrt(-k)/(2*a)
    Gamma3_22 = 0
    Gamma3_23 = 0
    Gamma3_30 = H
    Gamma3_31 = 0
    Gamma3_32 = 0
    Gamma3_33 = 0
    
    #Boltzmann coefficients in physical units. constructed from Ricci rotation coefficients (spin connections)
    A1_1 = 3*H/5
    B1_1 = 0
    A1_2 = 0
    B1_2 = 0
    A1_3 = 0
    B1_3 = 0
    A2_1 = 0
    B2_1 = 0
    A2_2 = 3*H/5
    B2_2 = 0
    A2_3 = 0
    B2_3 = 0
    A3_1 = 0
    B3_1 = 0
    A3_2 = 0
    B3_2 = 0
    A3_3 = 3*H/5
    B3_3 = 0
    C11_1 = 2*np.sqrt(-k)*np.tanh(x_coord*np.sqrt(-k))/(5*a)
    C11_2 = 0
    C11_3 = 0
    C12_1 = 0
    C12_2 = 2*np.sqrt(-k)*np.tanh(x_coord*np.sqrt(-k))/(5*a)
    C12_3 = 0
    C13_1 = 0
    C13_2 = np.sqrt(-k)/(5*a)
    C13_3 = np.sqrt(-k)*np.tanh(x_coord*np.sqrt(-k))/(5*a)
    C21_1 = 0
    C21_2 = 2*np.sqrt(-k)*np.tanh(x_coord*np.sqrt(-k))/(5*a)
    C21_3 = 0
    C22_1 = -2*np.sqrt(-k)*np.tanh(x_coord*np.sqrt(-k))/(5*a)
    C22_2 = 0
    C22_3 = 0
    C23_1 = -np.sqrt(-k)/(5*a)
    C23_2 = 0
    C23_3 = 0
    C31_1 = 0
    C31_2 = np.sqrt(-k)/(5*a)
    C31_3 = np.sqrt(-k)*np.tanh(x_coord*np.sqrt(-k))/(5*a)
    C32_1 = -np.sqrt(-k)/(5*a)
    C32_2 = 0
    C32_3 = 0
    C33_1 = 0
    C33_2 = 0
    C33_3 = 0
    D1_11 = 0
    D1_12 = 0
    D1_13 = 0
    D1_21 = 0
    D1_22 = 0
    D1_23 = 0
    D1_31 = 0
    D1_32 = 0
    D1_33 = 0
    D2_11 = 0
    D2_12 = 0
    D2_13 = 0
    D2_21 = 0
    D2_22 = 0
    D2_23 = 0
    D2_31 = 0
    D2_32 = 0
    D2_33 = 0
    D3_11 = 0
    D3_12 = 0
    D3_13 = 0
    D3_21 = 0
    D3_22 = 0
    D3_23 = 0
    D3_31 = 0
    D3_32 = 0
    D3_33 = 0
    E_11 = 0
    E_12 = 0
    E_13 = 0
    E_21 = 0
    E_22 = 0
    E_23 = 0
    E_31 = 0
    E_32 = 0
    E_33 = 0
    F11_11 = -10*H/21
    F11_12 = 0
    F11_13 = 0
    F11_21 = 0
    F11_22 = 5*H/21
    F11_23 = 0
    F11_31 = 0
    F11_32 = 0
    F11_33 = 5*H/21
    F12_11 = 0
    F12_12 = -5*H/14
    F12_13 = 0
    F12_21 = -5*H/14
    F12_22 = 0
    F12_23 = 0
    F12_31 = 0
    F12_32 = 0
    F12_33 = 0
    F13_11 = 0
    F13_12 = 0
    F13_13 = -5*H/14
    F13_21 = 0
    F13_22 = 0
    F13_23 = 0
    F13_31 = -5*H/14
    F13_32 = 0
    F13_33 = 0
    F21_11 = 0
    F21_12 = -5*H/14
    F21_13 = 0
    F21_21 = -5*H/14
    F21_22 = 0
    F21_23 = 0
    F21_31 = 0
    F21_32 = 0
    F21_33 = 0
    F22_11 = 5*H/21
    F22_12 = 0
    F22_13 = 0
    F22_21 = 0
    F22_22 = -10*H/21
    F22_23 = 0
    F22_31 = 0
    F22_32 = 0
    F22_33 = 5*H/21
    F23_11 = 0
    F23_12 = 0
    F23_13 = 0
    F23_21 = 0
    F23_22 = 0
    F23_23 = -5*H/14
    F23_31 = 0
    F23_32 = -5*H/14
    F23_33 = 0
    F31_11 = 0
    F31_12 = 0
    F31_13 = -5*H/14
    F31_21 = 0
    F31_22 = 0
    F31_23 = 0
    F31_31 = -5*H/14
    F31_32 = 0
    F31_33 = 0
    F32_11 = 0
    F32_12 = 0
    F32_13 = 0
    F32_21 = 0
    F32_22 = 0
    F32_23 = -5*H/14
    F32_31 = 0
    F32_32 = -5*H/14
    F32_33 = 0
    F33_11 = 5*H/21
    F33_12 = 0
    F33_13 = 0
    F33_21 = 0
    F33_22 = 5*H/21
    F33_23 = 0
    F33_31 = 0
    F33_32 = 0
    F33_33 = -10*H/21
    G11_11 = 0
    G11_12 = 0
    G11_13 = 0
    G11_21 = 0
    G11_22 = 0
    G11_23 = 0
    G11_31 = 0
    G11_32 = 0
    G11_33 = 0
    G12_11 = 0
    G12_12 = 0
    G12_13 = 0
    G12_21 = 0
    G12_22 = 0
    G12_23 = 0
    G12_31 = 0
    G12_32 = 0
    G12_33 = 0
    G13_11 = 0
    G13_12 = 0
    G13_13 = 0
    G13_21 = 0
    G13_22 = 0
    G13_23 = 0
    G13_31 = 0
    G13_32 = 0
    G13_33 = 0
    G21_11 = 0
    G21_12 = 0
    G21_13 = 0
    G21_21 = 0
    G21_22 = 0
    G21_23 = 0
    G21_31 = 0
    G21_32 = 0
    G21_33 = 0
    G22_11 = 0
    G22_12 = 0
    G22_13 = 0
    G22_21 = 0
    G22_22 = 0
    G22_23 = 0
    G22_31 = 0
    G22_32 = 0
    G22_33 = 0
    G23_11 = 0
    G23_12 = 0
    G23_13 = 0
    G23_21 = 0
    G23_22 = 0
    G23_23 = 0
    G23_31 = 0
    G23_32 = 0
    G23_33 = 0
    G31_11 = 0
    G31_12 = 0
    G31_13 = 0
    G31_21 = 0
    G31_22 = 0
    G31_23 = 0
    G31_31 = 0
    G31_32 = 0
    G31_33 = 0
    G32_11 = 0
    G32_12 = 0
    G32_13 = 0
    G32_21 = 0
    G32_22 = 0
    G32_23 = 0
    G32_31 = 0
    G32_32 = 0
    G32_33 = 0
    G33_11 = 0
    G33_12 = 0
    G33_13 = 0
    G33_21 = 0
    G33_22 = 0
    G33_23 = 0
    G33_31 = 0
    G33_32 = 0
    G33_33 = 0
    H1_11 = -np.sqrt(-k)*np.tanh(x_coord*np.sqrt(-k))/(3*a)
    H1_12 = 0
    H1_13 = 0
    H1_21 = 0
    H1_22 = 2*np.sqrt(-k)*np.tanh(x_coord*np.sqrt(-k))/(3*a)
    H1_23 = np.sqrt(-k)/(2*a)
    H1_31 = 0
    H1_32 = np.sqrt(-k)/(2*a)
    H1_33 = -np.sqrt(-k)*np.tanh(x_coord*np.sqrt(-k))/(3*a)
    H2_11 = 0
    H2_12 = -np.sqrt(-k)*np.tanh(x_coord*np.sqrt(-k))/(2*a)
    H2_13 = -np.sqrt(-k)/(2*a)
    H2_21 = -np.sqrt(-k)*np.tanh(x_coord*np.sqrt(-k))/(2*a)
    H2_22 = 0
    H2_23 = 0
    H2_31 = -np.sqrt(-k)/(2*a)
    H2_32 = 0
    H2_33 = 0
    H3_11 = 0
    H3_12 = 0
    H3_13 = 0
    H3_21 = 0
    H3_22 = 0
    H3_23 = 0
    H3_31 = 0
    H3_32 = 0
    H3_33 = 0
    RK11_11 = 4*H/9
    IK11_11 = -2*np.sqrt(-k)/(9*a)
    RK11_12 = 0
    IK11_12 = 0
    RK11_13 = 0
    IK11_13 = 0
    RK11_21 = 0
    IK11_21 = 0
    RK11_22 = -2*H/9
    IK11_22 = np.sqrt(-k)/(9*a)
    RK11_23 = 0
    IK11_23 = 0
    RK11_31 = 0
    IK11_31 = 0
    RK11_32 = 0
    IK11_32 = 0
    RK11_33 = -2*H/9
    IK11_33 = np.sqrt(-k)/(9*a)
    RK12_11 = 0
    IK12_11 = 0
    RK12_12 = H/3
    IK12_12 = -np.sqrt(-k)/(6*a)
    RK12_13 = 0
    IK12_13 = 0
    RK12_21 = H/3
    IK12_21 = -np.sqrt(-k)/(6*a)
    RK12_22 = 0
    IK12_22 = 0
    RK12_23 = 0
    IK12_23 = 0
    RK12_31 = 0
    IK12_31 = 0
    RK12_32 = 0
    IK12_32 = 0
    RK12_33 = 0
    IK12_33 = 0
    RK13_11 = 0
    IK13_11 = 0
    RK13_12 = 0
    IK13_12 = 0
    RK13_13 = H/3
    IK13_13 = -np.sqrt(-k)/(6*a)
    RK13_21 = 0
    IK13_21 = 0
    RK13_22 = 0
    IK13_22 = 0
    RK13_23 = 0
    IK13_23 = 0
    RK13_31 = H/3
    IK13_31 = -np.sqrt(-k)/(6*a)
    RK13_32 = 0
    IK13_32 = 0
    RK13_33 = 0
    IK13_33 = 0
    RK21_11 = 0
    IK21_11 = 0
    RK21_12 = H/3
    IK21_12 = -np.sqrt(-k)/(6*a)
    RK21_13 = 0
    IK21_13 = 0
    RK21_21 = H/3
    IK21_21 = -np.sqrt(-k)/(6*a)
    RK21_22 = 0
    IK21_22 = 0
    RK21_23 = 0
    IK21_23 = 0
    RK21_31 = 0
    IK21_31 = 0
    RK21_32 = 0
    IK21_32 = 0
    RK21_33 = 0
    IK21_33 = 0
    RK22_11 = -2*H/9
    IK22_11 = np.sqrt(-k)/(9*a)
    RK22_12 = 0
    IK22_12 = 0
    RK22_13 = 0
    IK22_13 = 0
    RK22_21 = 0
    IK22_21 = 0
    RK22_22 = 4*H/9
    IK22_22 = -2*np.sqrt(-k)/(9*a)
    RK22_23 = 0
    IK22_23 = 0
    RK22_31 = 0
    IK22_31 = 0
    RK22_32 = 0
    IK22_32 = 0
    RK22_33 = -2*H/9
    IK22_33 = np.sqrt(-k)/(9*a)
    RK23_11 = 0
    IK23_11 = 0
    RK23_12 = 0
    IK23_12 = 0
    RK23_13 = 0
    IK23_13 = 0
    RK23_21 = 0
    IK23_21 = 0
    RK23_22 = 0
    IK23_22 = 0
    RK23_23 = H/3
    IK23_23 = -np.sqrt(-k)/(6*a)
    RK23_31 = 0
    IK23_31 = 0
    RK23_32 = H/3
    IK23_32 = -np.sqrt(-k)/(6*a)
    RK23_33 = 0
    IK23_33 = 0
    RK31_11 = 0
    IK31_11 = 0
    RK31_12 = 0
    IK31_12 = 0
    RK31_13 = H/3
    IK31_13 = -np.sqrt(-k)/(6*a)
    RK31_21 = 0
    IK31_21 = 0
    RK31_22 = 0
    IK31_22 = 0
    RK31_23 = 0
    IK31_23 = 0
    RK31_31 = H/3
    IK31_31 = -np.sqrt(-k)/(6*a)
    RK31_32 = 0
    IK31_32 = 0
    RK31_33 = 0
    IK31_33 = 0
    RK32_11 = 0
    IK32_11 = 0
    RK32_12 = 0
    IK32_12 = 0
    RK32_13 = 0
    IK32_13 = 0
    RK32_21 = 0
    IK32_21 = 0
    RK32_22 = 0
    IK32_22 = 0
    RK32_23 = H/3
    IK32_23 = -np.sqrt(-k)/(6*a)
    RK32_31 = 0
    IK32_31 = 0
    RK32_32 = H/3
    IK32_32 = -np.sqrt(-k)/(6*a)
    RK32_33 = 0
    IK32_33 = 0
    RK33_11 = -2*H/9
    IK33_11 = np.sqrt(-k)/(9*a)
    RK33_12 = 0
    IK33_12 = 0
    RK33_13 = 0
    IK33_13 = 0
    RK33_21 = 0
    IK33_21 = 0
    RK33_22 = -2*H/9
    IK33_22 = np.sqrt(-k)/(9*a)
    RK33_23 = 0
    IK33_23 = 0
    RK33_31 = 0
    IK33_31 = 0
    RK33_32 = 0
    IK33_32 = 0
    RK33_33 = 4*H/9
    IK33_33 = -2*np.sqrt(-k)/(9*a)
    
    #Transfer equations with physical time as the independent variable
    dR0_0 = (1/3)*(Gamma0_11 + Gamma0_22 + Gamma0_33)*R0_0 + (2/15)*(Gamma0_11 * R0_11 + Gamma0_12 * R0_12 + Gamma0_13 * R0_13 + Gamma0_21 * R0_21 + Gamma0_22 * R0_22 + Gamma0_23 * R0_23 + Gamma0_31 * R0_31 + Gamma0_32 * R0_32 + Gamma0_33 * R0_33)*Re_Zeta - (2/15)*(Gamma0_11 * I0_11 + Gamma0_12 * I0_12 + Gamma0_13 * I0_13 + Gamma0_21 * I0_21 + Gamma0_22 * I0_22 + Gamma0_23 * I0_23 + Gamma0_31 * I0_31 + Gamma0_32 * I0_32 + Gamma0_33 * I0_33)*Im_Zeta + (1/3)*((Gamma1_11 * R0_1 + Gamma2_11 * R0_2 + Gamma3_11 * R0_3) + (Gamma1_22 * R0_1 + Gamma2_22 * R0_2 + Gamma3_22 * R0_3) + (Gamma1_33 * R0_1 + Gamma2_33 * R0_2 + Gamma3_33 * R0_3)) + (2/5)*(Gamma0_11 * R0_11 + Gamma0_12 * R0_12 + Gamma0_13 * R0_13 + Gamma0_21 * R0_21 + Gamma0_22 * R0_22 + Gamma0_23 * R0_23 + Gamma0_31 * R0_31 + Gamma0_32 * R0_32 + Gamma0_33 * R0_33)

    dI0_0 = (1/3)*(Gamma0_11 + Gamma0_22 + Gamma0_33)*I0_0 + (2/15)*(Gamma0_11 * I0_11 + Gamma0_12 * I0_12 + Gamma0_13 * I0_13 + Gamma0_21 * I0_21 + Gamma0_22 * I0_22 + Gamma0_23 * I0_23 + Gamma0_31 * I0_31 + Gamma0_32 * I0_32 + Gamma0_33 * I0_33)*Re_Zeta + (2/15)*(Gamma0_11 * R0_11 + Gamma0_12 * R0_12 + Gamma0_13 * R0_13 + Gamma0_21 * R0_21 + Gamma0_22 * R0_22 + Gamma0_23 * R0_23 + Gamma0_31 * R0_31 + Gamma0_32 * R0_32 + Gamma0_33 * R0_33)*Im_Zeta + (1/3)*((Gamma1_11 * I0_1 + Gamma2_11 * I0_2 + Gamma3_11 * I0_3) + (Gamma1_22 * I0_1 + Gamma2_22 * I0_2 + Gamma3_22 * I0_3) + (Gamma1_33 * I0_1 + Gamma2_33 * I0_2 + Gamma3_33 * I0_3)) + (2/5)*(Gamma0_11 * I0_11 + Gamma0_12 * I0_12 + Gamma0_13 * I0_13 + Gamma0_21 * I0_21 + Gamma0_22 * I0_22 + Gamma0_23 * I0_23 + Gamma0_31 * I0_31 + Gamma0_32 * I0_32 + Gamma0_33 * I0_33) - tau*I0_0

#----------------------------------------------------------------------
    dR0_1 = -Re_Zeta*(A1_1*R0_1 + A2_1*R0_2 + A3_1*R0_3) + Im_Zeta*(A1_1*I0_1 + A2_1*I0_2 + A3_1*I0_3) - (B1_1*R0_1 + B2_1*R0_2 + B3_1*R0_3) - (C11_1*R0_11 + C12_1*R0_12 + C13_1*R0_13 + C21_1*R0_21 + C22_1*R0_22 + C23_1*R0_23 + C31_1*R0_31 + C32_1*R0_32 + C33_1*R0_33) - tau*R0_1

    dR0_2 = -Re_Zeta*(A1_2*R0_1 + A2_2*R0_2 + A3_2*R0_3) + Im_Zeta*(A1_2*I0_1 + A2_2*I0_2 + A3_2*I0_3) - (B1_2*R0_1 + B2_2*R0_2 + B3_2*R0_3) - (C11_2*R0_11 + C12_2*R0_12 + C13_2*R0_13 + C21_2*R0_21 + C22_2*R0_22 + C23_2*R0_23 + C31_2*R0_31 + C32_2*R0_32 + C33_2*R0_33) - tau*R0_2

    dR0_3 = -Re_Zeta*(A1_3*R0_1 + A2_3*R0_2 + A3_3*R0_3) + Im_Zeta*(A1_3*I0_1 + A2_3*I0_2 + A3_3*I0_3) - (B1_3*R0_1 + B2_3*R0_2 + B3_3*R0_3) - (C11_3*R0_11 + C12_3*R0_12 + C13_3*R0_13 + C21_3*R0_21 + C22_3*R0_22 + C23_3*R0_23 + C31_3*R0_31 + C32_3*R0_32 + C33_3*R0_33) - tau*R0_3

    dI0_1 = -Re_Zeta*(A1_1*I0_1 + A2_1*I0_2 + A3_1*I0_3) - Im_Zeta*(A1_1*R0_1 + A2_1*R0_2 + A3_1*R0_3) - (B1_1*I0_1 + B2_1*I0_2 + B3_1*I0_3) - (C11_1*I0_11 + C12_1*I0_12 + C13_1*I0_13 + C21_1*I0_21 + C22_1*I0_22 + C23_1*I0_23 + C31_1*I0_31 + C32_1*I0_32 + C33_1*I0_33) - (2/3)*tau*I0_1

    dI0_2 = -Re_Zeta*(A1_2*I0_1 + A2_2*I0_2 + A3_2*I0_3) - Im_Zeta*(A1_2*R0_1 + A2_2*R0_2 + A3_2*R0_3) - (B1_2*I0_1 + B2_2*I0_2 + B3_2*I0_3) - (C11_2*I0_11 + C12_2*I0_12 + C13_2*I0_13 + C21_2*I0_21 + C22_2*I0_22 + C23_2*I0_23 + C31_2*I0_31 + C32_2*I0_32 + C33_2*I0_33) - (2/3)*tau*I0_2

    dI0_3 = -Re_Zeta*(A1_3*I0_1 + A2_3*I0_2 + A3_3*I0_3) - Im_Zeta*(A1_3*R0_1 + A2_3*R0_2 + A3_3*R0_3) - (B1_3*I0_1 + B2_3*I0_2 + B3_3*I0_3) - (C11_3*I0_11 + C12_3*I0_12 + C13_3*I0_13 + C21_3*I0_21 + C22_3*I0_22 + C23_3*I0_23 + C31_3*I0_31 + C32_3*I0_32 + C33_3*I0_33) - (2/3)*tau*I0_3

#----------------------------------------------------------------------
    dR0_11 = -Re_Zeta*R0_0*E_11 + Im_Zeta*I0_0*E_11 - Re_Zeta*(R0_1*D1_11 + R0_2*D2_11 + R0_3*D3_11) + Im_Zeta*(I0_1*D1_11 + I0_2*D2_11 + I0_3*D3_11) - (R0_1*H1_11 + R0_2*H2_11 + R0_3*H3_11) - Re_Zeta*(F11_11*R0_11 + F12_11*R0_12 + F13_11*R0_13 + F21_11*R0_21 + F22_11*R0_22 + F23_11*R0_23 + F31_11*R0_31 + F32_11*R0_32 + F33_11*R0_33) + Im_Zeta*(F11_11*I0_11 + F12_11*I0_12 + F13_11*I0_13 + F21_11*I0_21 + F22_11*I0_22 + F23_11*I0_23 + F31_11*I0_31 + F32_11*I0_32 + F33_11*I0_33) - (G11_11*R0_11 + G12_11*R0_12 + G13_11*R0_13 + G21_11*R0_21 + G22_11*R0_22 + G23_11*R0_23 + G31_11*R0_31 + G32_11*R0_32 + G33_11*R0_33) - (9/10)*tau*R0_11 - (3/10)*tau*R2_11

    dR0_12 = -Re_Zeta*R0_0*E_12 + Im_Zeta*I0_0*E_12 - Re_Zeta*(R0_1*D1_12 + R0_2*D2_12 + R0_3*D3_12) + Im_Zeta*(I0_1*D1_12 + I0_2*D2_12 + I0_3*D3_12) - (R0_1*H1_12 + R0_2*H2_12 + R0_3*H3_12) - Re_Zeta*(F11_12*R0_11 + F12_12*R0_12 + F13_12*R0_13 + F21_12*R0_21 + F22_12*R0_22 + F23_12*R0_23 + F31_12*R0_31 + F32_12*R0_32 + F33_12*R0_33) + Im_Zeta*(F11_12*I0_11 + F12_12*I0_12 + F13_12*I0_13 + F21_12*I0_21 + F22_12*I0_22 + F23_12*I0_23 + F31_12*I0_31 + F32_12*I0_32 + F33_12*I0_33) - (G11_12*R0_11 + G12_12*R0_12 + G13_12*R0_13 + G21_12*R0_21 + G22_12*R0_22 + G23_12*R0_23 + G31_12*R0_31 + G32_12*R0_32 + G33_12*R0_33) - (9/10)*tau*R0_12 - (3/10)*tau*R2_12

    dR0_13 = -Re_Zeta*R0_0*E_13 + Im_Zeta*I0_0*E_13 - Re_Zeta*(R0_1*D1_13 + R0_2*D2_13 + R0_3*D3_13) + Im_Zeta*(I0_1*D1_13 + I0_2*D2_13 + I0_3*D3_13) - (R0_1*H1_13 + R0_2*H2_13 + R0_3*H3_13) - Re_Zeta*(F11_13*R0_11 + F12_13*R0_12 + F13_13*R0_13 + F21_13*R0_21 + F22_13*R0_22 + F23_13*R0_23 + F31_13*R0_31 + F32_13*R0_32 + F33_13*R0_33) + Im_Zeta*(F11_13*I0_11 + F12_13*I0_12 + F13_13*I0_13 + F21_13*I0_21 + F22_13*I0_22 + F23_13*I0_23 + F31_13*I0_31 + F32_13*I0_32 + F33_13*I0_33) - (G11_13*R0_11 + G12_13*R0_12 + G13_13*R0_13 + G21_13*R0_21 + G22_13*R0_22 + G23_13*R0_23 + G31_13*R0_31 + G32_13*R0_32 + G33_13*R0_33) - (9/10)*tau*R0_13 - (3/10)*tau*R2_13

    dR0_21 = dR0_12

    dR0_22 = -Re_Zeta*R0_0*E_22 + Im_Zeta*I0_0*E_22 - Re_Zeta*(R0_1*D1_22 + R0_2*D2_22 + R0_3*D3_22) + Im_Zeta*(I0_1*D1_22 + I0_2*D2_22 + I0_3*D3_22) - (R0_1*H1_22 + R0_2*H2_22 + R0_3*H3_22) - Re_Zeta*(F11_22*R0_11 + F12_22*R0_12 + F13_22*R0_13 + F21_22*R0_21 + F22_22*R0_22 + F23_22*R0_23 + F31_22*R0_31 + F32_22*R0_32 + F33_22*R0_33) + Im_Zeta*(F11_22*I0_11 + F12_22*I0_12 + F13_22*I0_13 + F21_22*I0_21 + F22_22*I0_22 + F23_22*I0_23 + F31_22*I0_31 + F32_22*I0_32 + F33_22*I0_33) - (G11_22*R0_11 + G12_22*R0_12 + G13_22*R0_13 + G21_22*R0_21 + G22_22*R0_22 + G23_22*R0_23 + G31_22*R0_31 + G32_22*R0_32 + G33_22*R0_33) - (9/10)*tau*R0_22 - (3/10)*tau*R2_22

    dR0_23 = -Re_Zeta*R0_0*E_23 + Im_Zeta*I0_0*E_23 - Re_Zeta*(R0_1*D1_23 + R0_2*D2_23 + R0_3*D3_23) + Im_Zeta*(I0_1*D1_23 + I0_2*D2_23 + I0_3*D3_23) - (R0_1*H1_23 + R0_2*H2_23 + R0_3*H3_23) - Re_Zeta*(F11_23*R0_11 + F12_23*R0_12 + F13_23*R0_13 + F21_23*R0_21 + F22_23*R0_22 + F23_23*R0_23 + F31_23*R0_31 + F32_23*R0_32 + F33_23*R0_33) + Im_Zeta*(F11_23*I0_11 + F12_23*I0_12 + F13_23*I0_13 + F21_23*I0_21 + F22_23*I0_22 + F23_23*I0_23 + F31_23*I0_31 + F32_23*I0_32 + F33_23*I0_33) - (G11_23*R0_11 + G12_23*R0_12 + G13_23*R0_13 + G21_23*R0_21 + G22_23*R0_22 + G23_23*R0_23 + G31_23*R0_31 + G32_23*R0_32 + G33_23*R0_33) - (9/10)*tau*R0_23 - (3/10)*tau*R2_23

    dR0_31 = dR0_13

    dR0_32 = dR0_23

    dR0_33 = -dR0_11 - dR0_22

    dI0_11 = -Re_Zeta*I0_0*E_11 - Im_Zeta*R0_0*E_11 - Re_Zeta*(I0_1*D1_11 + I0_2*D2_11 + I0_3*D3_11) - Im_Zeta*(R0_1*D1_11 + R0_2*D2_11 + R0_3*D3_11) - (I0_1*H1_11 + I0_2*H2_11 + I0_3*H3_11) - Re_Zeta*(F11_11*I0_11 + F12_11*I0_12 + F13_11*I0_13 + F21_11*I0_21 + F22_11*I0_22 + F23_11*I0_23 + F31_11*I0_31 + F32_11*I0_32 + F33_11*I0_33) - Im_Zeta*(F11_11*R0_11 + F12_11*R0_12 + F13_11*R0_13 + F21_11*R0_21 + F22_11*R0_22 + F23_11*R0_23 + F31_11*R0_31 + F32_11*R0_32 + F33_11*R0_33) - (G11_11*I0_11 + G12_11*I0_12 + G13_11*I0_13 + G21_11*I0_21 + G22_11*I0_22 + G23_11*I0_23 + G31_11*I0_31 + G32_11*I0_32 + G33_11*I0_33) - tau*I0_11

    dI0_12 = -Re_Zeta*I0_0*E_12 - Im_Zeta*R0_0*E_12 - Re_Zeta*(I0_1*D1_12 + I0_2*D2_12 + I0_3*D3_12) - Im_Zeta*(R0_1*D1_12 + R0_2*D2_12 + R0_3*D3_12) - (I0_1*H1_12 + I0_2*H2_12 + I0_3*H3_12) - Re_Zeta*(F11_12*I0_11 + F12_12*I0_12 + F13_12*I0_13 + F21_12*I0_21 + F22_12*I0_22 + F23_12*I0_23 + F31_12*I0_31 + F32_12*I0_32 + F33_12*I0_33) - Im_Zeta*(F11_12*R0_11 + F12_12*R0_12 + F13_12*R0_13 + F21_12*R0_21 + F22_12*R0_22 + F23_12*R0_23 + F31_12*R0_31 + F32_12*R0_32 + F33_12*R0_33) - (G11_12*I0_11 + G12_12*I0_12 + G13_12*I0_13 + G21_12*I0_21 + G22_12*I0_22 + G23_12*I0_23 + G31_12*I0_31 + G32_12*I0_32 + G33_12*I0_33) - tau*I0_12

    dI0_13 = -Re_Zeta*I0_0*E_13 - Im_Zeta*R0_0*E_13 - Re_Zeta*(I0_1*D1_13 + I0_2*D2_13 + I0_3*D3_13) - Im_Zeta*(R0_1*D1_13 + R0_2*D2_13 + R0_3*D3_13) - (I0_1*H1_13 + I0_2*H2_13 + I0_3*H3_13) - Re_Zeta*(F11_13*I0_11 + F12_13*I0_12 + F13_13*I0_13 + F21_13*I0_21 + F22_13*I0_22 + F23_13*I0_23 + F31_13*I0_31 + F32_13*I0_32 + F33_13*I0_33) - Im_Zeta*(F11_13*R0_11 + F12_13*R0_12 + F13_13*R0_13 + F21_13*R0_21 + F22_13*R0_22 + F23_13*R0_23 + F31_13*R0_31 + F32_13*R0_32 + F33_13*R0_33) - (G11_13*I0_11 + G12_13*I0_12 + G13_13*I0_13 + G21_13*I0_21 + G22_13*I0_22 + G23_13*I0_23 + G31_13*I0_31 + G32_13*I0_32 + G33_13*I0_33) - tau*I0_13

    dI0_21 = dI0_12

    dI0_22 = -Re_Zeta*I0_0*E_22 - Im_Zeta*R0_0*E_22 - Re_Zeta*(I0_1*D1_22 + I0_2*D2_22 + I0_3*D3_22) - Im_Zeta*(R0_1*D1_22 + R0_2*D2_22 + R0_3*D3_22) - (I0_1*H1_22 + I0_2*H2_22 + I0_3*H3_22) - Re_Zeta*(F11_22*I0_11 + F12_22*I0_12 + F13_22*I0_13 + F21_22*I0_21 + F22_22*I0_22 + F23_22*I0_23 + F31_22*I0_31 + F32_22*I0_32 + F33_22*I0_33) - Im_Zeta*(F11_22*R0_11 + F12_22*R0_12 + F13_22*R0_13 + F21_22*R0_21 + F22_22*R0_22 + F23_22*R0_23 + F31_22*R0_31 + F32_22*R0_32 + F33_22*R0_33) - (G11_22*I0_11 + G12_22*I0_12 + G13_22*I0_13 + G21_22*I0_21 + G22_22*I0_22 + G23_22*I0_23 + G31_22*I0_31 + G32_22*I0_32 + G33_22*I0_33) - tau*I0_22

    dI0_23 = -Re_Zeta*I0_0*E_23 - Im_Zeta*R0_0*E_23 - Re_Zeta*(I0_1*D1_23 + I0_2*D2_23 + I0_3*D3_23) - Im_Zeta*(R0_1*D1_23 + R0_2*D2_23 + R0_3*D3_23) - (I0_1*H1_23 + I0_2*H2_23 + I0_3*H3_23) - Re_Zeta*(F11_23*I0_11 + F12_23*I0_12 + F13_23*I0_13 + F21_23*I0_21 + F22_23*I0_22 + F23_23*I0_23 + F31_23*I0_31 + F32_23*I0_32 + F33_23*I0_33) - Im_Zeta*(F11_23*R0_11 + F12_23*R0_12 + F13_23*R0_13 + F21_23*R0_21 + F22_23*R0_22 + F23_23*R0_23 + F31_23*R0_31 + F32_23*R0_32 + F33_23*R0_33) - (G11_23*I0_11 + G12_23*I0_12 + G13_23*I0_13 + G21_23*I0_21 + G22_23*I0_22 + G23_23*I0_23 + G31_23*I0_31 + G32_23*I0_32 + G33_23*I0_33) - tau*I0_23

    dI0_31 = dI0_13

    dI0_32 = dI0_23

    dI0_33 = -dI0_11 - dI0_22

#----------------------------------------------------------------------
    dR2_11 = (1/3)*(Gamma0_11 + Gamma0_22 + Gamma0_33)*(Re_Zeta*R2_11 - Im_Zeta*I2_11) - (RK11_11*R2_11 + RK12_11*R2_12 + RK13_11*R2_13 + RK21_11*R2_21 + RK22_11*R2_22 + RK23_11*R2_23 + RK31_11*R2_31 + RK32_11*R2_32 + RK33_11*R2_33) - (1/5)*tau*R0_11 - (2/5)*tau*R2_11

    dR2_12 = (1/3)*(Gamma0_11 + Gamma0_22 + Gamma0_33)*(Re_Zeta*R2_12 - Im_Zeta*I2_12) - (RK11_12*R2_11 + RK12_12*R2_12 + RK13_12*R2_13 + RK21_12*R2_21 + RK22_12*R2_22 + RK23_12*R2_23 + RK31_12*R2_31 + RK32_12*R2_32 + RK33_12*R2_33) - (1/5)*tau*R0_12 - (2/5)*tau*R2_12

    dR2_13 = (1/3)*(Gamma0_11 + Gamma0_22 + Gamma0_33)*(Re_Zeta*R2_13 - Im_Zeta*I2_13) - (RK11_13*R2_11 + RK12_13*R2_12 + RK13_13*R2_13 + RK21_13*R2_21 + RK22_13*R2_22 + RK23_13*R2_23 + RK31_13*R2_31 + RK32_13*R2_32 + RK33_13*R2_33) - (1/5)*tau*R0_13 - (2/5)*tau*R2_13

    dR2_21 = dR2_12

    dR2_22 = (1/3)*(Gamma0_11 + Gamma0_22 + Gamma0_33)*(Re_Zeta*R2_22 - Im_Zeta*I2_22) - (RK11_22*R2_11 + RK12_22*R2_12 + RK13_22*R2_13 + RK21_22*R2_21 + RK22_22*R2_22 + RK23_22*R2_23 + RK31_22*R2_31 + RK32_22*R2_32 + RK33_22*R2_33) - (1/5)*tau*R0_22 - (2/5)*tau*R2_22

    dR2_23 = (1/3)*(Gamma0_11 + Gamma0_22 + Gamma0_33)*(Re_Zeta*R2_23 - Im_Zeta*I2_23) - (RK11_23*R2_11 + RK12_23*R2_12 + RK13_23*R2_13 + RK21_23*R2_21 + RK22_23*R2_22 + RK23_23*R2_23 + RK31_23*R2_31 + RK32_23*R2_32 + RK33_23*R2_33) - (1/5)*tau*R0_23 - (2/5)*tau*R2_23

    dR2_31 = dR2_13

    dR2_32 = dR2_23

    dR2_33 = -dR2_11 - dR2_22

    dI2_11 = (1/3)*(Gamma0_11 + Gamma0_22 + Gamma0_33)*(Re_Zeta*I2_11 + Im_Zeta*R2_11) - (IK11_11*I2_11 + IK12_11*I2_12 + IK13_11*I2_13 + IK21_11*I2_21 + IK22_11*I2_22 + IK23_11*I2_23 + IK31_11*I2_31 + IK32_11*I2_32 + IK33_11*I2_33) - tau*I2_11

    dI2_12 = (1/3)*(Gamma0_11 + Gamma0_22 + Gamma0_33)*(Re_Zeta*I2_12 + Im_Zeta*R2_12) - (IK11_12*I2_11 + IK12_12*I2_12 + IK13_12*I2_13 + IK21_12*I2_21 + IK22_12*I2_22 + IK23_12*I2_23 + IK31_12*I2_31 + IK32_12*I2_32 + IK33_12*I2_33) - tau*I2_12

    dI2_13 = (1/3)*(Gamma0_11 + Gamma0_22 + Gamma0_33)*(Re_Zeta*I2_13 + Im_Zeta*R2_13) - (IK11_13*I2_11 + IK12_13*I2_12 + IK13_13*I2_13 + IK21_13*I2_21 + IK22_13*I2_22 + IK23_13*I2_23 + IK31_13*I2_31 + IK32_13*I2_32 + IK33_13*I2_33) - tau*I2_13

    dI2_21 = dI2_12

    dI2_22 = (1/3)*(Gamma0_11 + Gamma0_22 + Gamma0_33)*(Re_Zeta*I2_22 + Im_Zeta*R2_22) - (IK11_22*I2_11 + IK12_22*I2_12 + IK13_22*I2_13 + IK21_22*I2_21 + IK22_22*I2_22 + IK23_22*I2_23 + IK31_22*I2_31 + IK32_22*I2_32 + IK33_22*I2_33) - tau*I2_22

    dI2_23 = (1/3)*(Gamma0_11 + Gamma0_22 + Gamma0_33)*(Re_Zeta*I2_23 + Im_Zeta*R2_23) - (IK11_23*I2_11 + IK12_23*I2_12 + IK13_23*I2_13 + IK21_23*I2_21 + IK22_23*I2_22 + IK23_23*I2_23 + IK31_23*I2_31 + IK32_23*I2_32 + IK33_23*I2_33) - tau*I2_23

    dI2_31 = dI2_13

    dI2_32 = dI2_23

    dI2_33 = -dI2_11 - dI2_22
    
    #Photon monopole = Abs. BG temp.
    I0_0 = 0
    dI0_0 = 0
    
    #d_chi = np.sqrt((1/a**2) - (chi**2)*((d_theta)**2 + (np.sin(theta)*(d_phi))**2))
    d_chi = 1.0/a                                                                                           #d_theta = d_phi = 0 for zeroth-order approximation (Sung & Coles paper),
                                                                                                            #treating k ~ Omega_k*H0**2 << 1 as a perturbation in Thurston geometries
    d_x_coord = d_chi*np.sin(theta)*np.cos(phi) + chi*(np.cos(theta)*np.cos(phi)*d_theta - np.sin(theta)*np.sin(phi)*d_phi)
    d_y_coord = d_chi*np.sin(theta)*np.sin(phi) + chi*(np.cos(theta)*np.sin(phi)*d_theta + np.sin(theta)*np.cos(phi)*d_phi)
    d_z_coord = d_chi*np.cos(theta) - chi*np.sin(theta)*d_theta
    
    d_vec = np.array([
        dR0_0, dI0_0,
        dR0_1, dR0_2, dR0_3, dI0_1, dI0_2, dI0_3,
        dR0_11, dR0_12, dR0_13, dR0_21, dR0_22, dR0_23, dR0_31, dR0_32, dR0_33,
        dI0_11, dI0_12, dI0_13, dI0_21, dI0_22, dI0_23, dI0_31, dI0_32, dI0_33,
        dR2_11, dR2_12, dR2_13, dR2_21, dR2_22, dR2_23, dR2_31, dR2_32, dR2_33,
        dI2_11, dI2_12, dI2_13, dI2_21, dI2_22, dI2_23, dI2_31, dI2_32, dI2_33,
        d_psi_real, d_psi_imag, d_theta, d_phi, d_chi, d_x_coord, d_y_coord, d_z_coord
    ])
    return d_vec / (a * H)                                                                                #Independent variable changed from 't' to 'a'

#Q>0 means polarization aligns with et_ .. (North-South) -----------------------------------------------------------------------------------------------
#Q<0 means polarization aligns with ep_ .. (East-West)
def get_basis_vectors(theta, phi):
    et_x = np.cos(theta) * np.cos(phi)
    et_y = np.cos(theta) * np.sin(phi)
    et_z = -np.sin(theta)
    ep_x = -np.sin(phi)
    ep_y = np.cos(phi)
    ep_z = 0.0
    
    return np.array([et_x, et_y, et_z]), np.array([ep_x, ep_y, ep_z])

def build_y0(theta0, phi0, m_case, chi_init):
    y0 = np.zeros(52)
    
    #Fractional temperature variation measured relative to a background baseline of 1 (100%) to keep the baseline math consistent, as shown in below three cases 
    #np.since we do not have a kinematic shear in this form [Sung & Coles]
    y0[0] = 1.0e-6                                                                                          #Initial monopole

    #Initial seeds (from CAMB, normalized by T0 = 2.725e+6 uK) -----------------------------------------
    if m_case == 0:
        #1. l = 2, m = 0
        y0[8] = -2.765e-6                                                                                   #R0_11
        y0[12] = y0[8]                                                                                      #R0_22
        y0[16] = -(y0[8] + y0[12])                                                                          #R0_33

    elif m_case == 1:
        #2. l = 2, m = 1 (m = -1 is also included here by healpy)
        y0[10] = -6.774e-6                                                                                  #R0_13
        y0[14] = y0[10]                                                                                     #R0_31 = R0_13

    elif m_case == 2:
        #3. l = 2, m = 2 (m = -2 is also included here by healpy)
        y0[8] = 6.774e-6                                                                                    #R0_11
        y0[12] = -y0[8]                                                                                     #R0_22
        y0[16] = -(y0[8] + y0[12])                                                                          #R0_33

    y0[46], y0[47], y0[48] = theta0, phi0, chi_init
    y0[49] = chi_init*np.sin(theta0)*np.cos(phi0)
    y0[50] = chi_init*np.sin(theta0)*np.sin(phi0)
    y0[51] = chi_init*np.cos(theta0)
    return y0
    
def compute_T_local_and_prefactor(a_eval_pts, T_background, nu_instrument, h, c,
                                   Omega_m=0.315, Omega_L=0.641):
    T_local_arr = np.zeros(len(a_eval_pts))
    prefactor_arr = np.zeros(len(a_eval_pts))

    for idx, a_i in enumerate(a_eval_pts):
        z_val = 1.0 / a_i - 1.0

        T_local = T_background * (1.0 + z_val)
        prefactor = 2.0 * h * nu_instrument**3 / c**2

        T_local_arr[idx] = T_local
        prefactor_arr[idx] = prefactor

    return T_local_arr, prefactor_arr

def solve_single_pixel(args):
    idx, theta_0, phi_0, a_eval_pts, T_local_arr, prefactor_arr, chi_init = args   # renamed
    n = len(a_eval_pts)
    dT_tot, I_tot, Q_tot, U_tot, V_tot = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

    for m_case in (0, 1, 2):
        y0 = build_y0(theta_0, phi_0, m_case, chi_init)
        #sol = integrate.solve_ivp(fun=lambda a, y: derivatives_hpc(a, y),
                                   #t_span=(a_eval_pts[0], a_eval_pts[-1]), y0=y0,
                                   #t_eval=a_eval_pts, method='LSODA', rtol=1e-8, atol=1e-10)
                                   
        atol = np.full(len(y0), 1e-13)
        atol[44:48] = 1e-10                                                                                     #psi_real, psi_imag, theta, phi
        atol[48:] = 1e6                                                                                         #chi, x_coord, y_coord, z_coord
        sol = integrate.solve_ivp(fun=derivatives_hpc,
                                  t_span=(a_eval_pts[0], a_eval_pts[-1]), y0=y0,
                                  t_eval=a_eval_pts, method='Radau', rtol=1e-6, atol=atol)

        if sol.y.shape[1] != n or not np.all(np.isfinite(sol.y)):
            nan_l = [np.nan]*n
            return (idx, nan_l, nan_l, nan_l, nan_l, nan_l)
        
        for i in range(n):
            s = sol.y[:, i]
            th_f, ph_f, chi_f = s[46], s[47], s[48]
            x_coord_f, y_coord_f, z_coord_f = s[49], s[50], s[51]

            R0_0, R0_vec = s[0], s[2:5]
            R0_tensor = s[8:17].reshape(3, 3)
            k_vec = np.array([np.sin(th_f)*np.cos(ph_f), np.sin(th_f)*np.sin(ph_f), np.cos(th_f)])
            k_tensor = np.outer(k_vec, k_vec) - np.eye(3)/3.0

            I_val = R0_0 + np.dot(R0_vec, k_vec) + np.sum(R0_tensor * k_tensor)
            dT_over_T = np.sum(R0_tensor * k_tensor)

            R2_tensor, I2_tensor = s[26:35].reshape(3, 3), s[35:44].reshape(3, 3)
            e_th, e_ph = get_basis_vectors(th_f, ph_f)
            R_tt, R_pp = e_th @ R2_tensor @ e_th, e_ph @ R2_tensor @ e_ph
            R_tp, R_pt = e_th @ R2_tensor @ e_ph, e_ph @ R2_tensor @ e_th
            I_tt, I_pp = e_th @ I2_tensor @ e_th, e_ph @ I2_tensor @ e_ph
            I_tp, I_pt = e_th @ I2_tensor @ e_ph, e_ph @ I2_tensor @ e_th

            Q_raw = 0.5*(R_tt - R_pp) - 0.5*(I_tp + I_pt)
            U_raw = -0.5*(I_tt - I_pp) - 0.5*(R_tp + R_pt)
            Psi = s[44] + s[45]
            c2, s2 = np.cos(2*Psi), np.sin(2*Psi)
            Q_val = Q_raw*c2 + U_raw*s2                                                                     #Q/I
            U_val = -Q_raw*s2 + U_raw*c2                                                                    #U/I

            I0_0, I0_vec, I0_tensor = s[1], s[5:8], s[17:26].reshape(3, 3)     

            T_local = T_local_arr[i]
            prefactor_value = prefactor_arr[i]
            
            dT_tot[i] += dT_over_T * T_local                                                                #Pure temperature fluctuations
            I_tot[i] += I_val * prefactor_value                                                             #Pure Stokes: total intensity 'I'
            Q_tot[i] += Q_val * prefactor_value                                                             #Pure Stokes 'Q'
            U_tot[i] += U_val * prefactor_value                                                             #Pure Stokes 'U'
            
            #Consistency check for circular polarization
            V_val = I0_0 + np.dot(I0_vec, k_vec) + np.sum(I0_tensor * k_tensor)
            V_val = V_val * prefactor_value
            V_tot[i] = max(V_tot[i], abs(V_val))                                                            #Pure Stokes 'V'

    return (idx, dT_tot.tolist(), I_tot.tolist(), Q_tot.tolist(), U_tot.tolist(), V_tot.tolist())

if __name__ == "__main__":

    NSIDE = 32                                                                                               #Uniform in z
    z_targets = np.array([1200.0, 1100.0, 550.0, 0.0])
    steps = len(z_targets)
    a_eval = 1.0 / (1.0 + z_targets)
    
    T_background = 2.725                                                    #Background zeroth-order temperature
    c = 3e8                                                                 #Speed of light in vacuum
    h = 6.626e-34                                                           #Planck's constant
    nu_instrument = 1.5e11                                                  #Detector frequency
    
    T_local_arr, prefactor_arr = compute_T_local_and_prefactor(
        a_eval, T_background, nu_instrument, h, c)

    NPIX = hp.nside2npix(NSIDE)
    theta_arr, phi_arr = hp.pix2ang(NSIDE, np.arange(NPIX))

    #cores = len(os.sched_getaffinity(0))
    cores = 8
    print(f"Cores = {cores}", flush=True)

    a_init = 1.0/(1.0 + z_targets[0])
    chi_init = chi_of_a(1.0, a_start=a_init)
    print(f"chi_init = {chi_init:.6f}", flush=True)
    
    tasks = [(i, theta_arr[i], phi_arr[i], a_eval, T_local_arr, prefactor_arr, chi_init) for i in range(NPIX)]

    T_maps = np.full((steps, NPIX), np.nan)
    I_maps = np.full((steps, NPIX), np.nan)
    Q_maps = np.full((steps, NPIX), np.nan)
    U_maps = np.full((steps, NPIX), np.nan)

    count = 0
    t0 = time.time()
    pbar = tqdm(total=NPIX, desc="Solving pixels", unit="pix", dynamic_ncols=True, smoothing=0.1)
    
    #Stokes-V consistency check
    max_V_per_step = np.zeros(steps)

    with mp.Pool(processes=cores) as pool:
        for result in pool.imap_unordered(solve_single_pixel, tasks, chunksize=32):
            idx, res_dt_list, res_i_list, res_q_list, res_u_list, res_v_list = result
            
            max_V_per_step = np.fmax(max_V_per_step, res_v_list)
            
            for i in range(steps):
                T_maps[i, idx] = res_dt_list[i]
                I_maps[i, idx] = res_i_list[i]
                Q_maps[i, idx] = res_q_list[i]
                U_maps[i, idx] = res_u_list[i]
            count += 1
            pbar.update(1)
    pbar.close()
    
    #Consistency checks
    print("Consistency checks:")
    for i, a_val in enumerate(a_eval):
        print(f"At z = {z_targets[i]:.4f}: T_local = {T_local_arr[i]:.4f} K, Max |V| = {max_V_per_step[i]:.2e}")

    T_maps = np.nan_to_num(T_maps, nan=0.0, posinf=0.0, neginf=0.0)
    I_maps = np.nan_to_num(I_maps, nan=0.0, posinf=0.0, neginf=0.0)
    Q_maps = np.nan_to_num(Q_maps, nan=0.0, posinf=0.0, neginf=0.0)
    U_maps = np.nan_to_num(U_maps, nan=0.0, posinf=0.0, neginf=0.0)

    P_maps = np.sqrt(Q_maps**2 + U_maps**2)/I_maps
    
    #T,E,B maps
    tag = "UH2"
    os.makedirs("maps", exist_ok=True)
    Q_K = Q_maps / prefactor_arr[:, None] * T_local_arr[:, None]
    U_K = U_maps / prefactor_arr[:, None] * T_local_arr[:, None]

    np.savez_compressed(f"maps/{tag}_maps.npz",
                        T=T_maps, Q_K=Q_K, U_K=U_K,
                        I=I_maps, Q=Q_maps, U=U_maps, P=P_maps,
                        z=z_targets, nside=NSIDE)

    plt.rcParams['font.family'] = 'serif'
    output_dir = "maps"
    os.makedirs(output_dir, exist_ok=True)

    rows, cols = steps, 5
    fig = plt.figure(figsize=(28, 4.5 * rows))

    def plot_styled_map(data, cmap, v_min, v_max, title_str, position, is_top_row):
        hp.mollview(data, cmap=cmap, min=v_min, max=v_max, cbar=False, sub=(rows, cols, position), title="")
        ax = plt.gca()
        if is_top_row:
            plt.title(f"{title_str}", fontsize=22, pad=15)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=v_min, vmax=v_max))
        sm._A = []
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0 + 0.05, pos.y0 - 0.04, pos.width - 0.1, 0.015])
        cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cb.set_ticks([v_min, v_max])
        cb.ax.tick_params(labelsize=16, length=0)
        cb.formatter = ticker.FormatStrFormatter('%.2e')
        cb.update_ticks()

    for i, a_val in enumerate(a_eval):
        z_val = 1.0 / a_val - 1.0

        T_scale = np.nanmax(np.abs(T_maps[i]))
        P_scale = np.nanmax(P_maps[i])
        Q_scale = np.nanmax(np.abs(Q_maps[i]))
        U_scale = np.nanmax(np.abs(U_maps[i]))

        row_idx = (steps - 1) - i
        base_pos = row_idx * cols
        is_top = (row_idx == 0)

        ax_text = fig.add_subplot(rows, cols, base_pos + 1)
        ax_text.axis('off')

        if i == 0:
            z_str = "z = 1200"
        elif i == steps - 1:
            z_str = "z = 0"
        elif z_val > 100:
            z_str = f"z = {z_val:.0f}"
        elif z_val > 10:
            z_str = f"z = {z_val:.1f}"
        else:
            z_str = f"z = {z_val:.2f}"

        ax_text.text(0.1, 0.5, f"\n{z_str}", fontsize=26, ha='left', va='center', fontweight='bold')

        plot_styled_map(T_maps[i], 'turbo', -T_scale, T_scale, r"$\Delta \text{T} \, \left(\text{K} \right)$", base_pos + 2, is_top)
        plot_styled_map(P_maps[i], 'turbo', 0, P_scale, r"$\text{P}$", base_pos + 3, is_top)
        plot_styled_map(Q_maps[i], 'turbo', -Q_scale, Q_scale, r"$\text{Q} \, \left(\text{W} \, \text{sr}^{-1} \, \text{m}^{-2} \, \text{Hz}^{-1} \right)$", base_pos + 4, is_top)
        plot_styled_map(U_maps[i], 'turbo', -U_scale, U_scale, r"$\text{U} \,  \left(\text{W} \, \text{sr}^{-1} \, \text{m}^{-2} \, \text{Hz}^{-1} \right)$", base_pos + 5, is_top)

    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    save_path = os.path.join(output_dir, "UH2.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Maps saved to: {save_path}")
