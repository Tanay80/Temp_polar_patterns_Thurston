import numpy as np
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

def derivatives_hpc(t, y):
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
    
    #Respecting physical symmetry -----------------------------------------------------------------------------------------------
    y[11] = (y[9] + y[11]) / 2.0
    y[14] = (y[10] + y[14]) / 2.0
    y[15] = (y[13] + y[15]) / 2.0
    y[20] = (y[18] + y[20]) / 2.0
    y[23] = (y[19] + y[23]) / 2.0
    y[24] = (y[22] + y[24]) / 2.0
    y[29] = (y[27] + y[29]) / 2.0
    y[32] = (y[28] + y[32]) / 2.0
    y[33] = (y[31] + y[33]) / 2.0
    y[38] = (y[36] + y[38]) / 2.0
    y[41] = (y[37] + y[41]) / 2.0
    y[42] = (y[40] + y[42]) / 2.0    
    
    y[9] = y[11]
    y[10] = y[14]
    y[13] = y[15]
    y[18] = y[20]
    y[19] = y[23]
    y[22] = y[24]
    y[27] = y[29]
    y[28] = y[32]
    y[31] = y[33]
    y[36] = y[38]
    y[37] = y[41]
    y[40] = y[42]
    
    #Enforcing tracelessness -----------------------------------------------------------------------------------------------
    y[16] = -y[8] - y[12]
    y[25] = -y[17] - y[21]
    y[34] = -y[26] - y[30]
    y[43] = -y[35] - y[39]
    
    psi_real = y[44]
    psi_imag = y[45]
    theta = y[46]
    phi = y[47]
    
    Omega_m = 0.31
    Omega_L = 0.68
    Omega_k = 1.0 - (Omega_m + Omega_L)
    sqrt_L = np.sqrt(Omega_L)
    
    arg = 1.5 * sqrt_L * t                                                                                  		  #Dimensionless time parameter 't' (= Physical time multilpied with H_0)    
    a_t = (Omega_m / Omega_L)**(1/3) * (np.sinh(arg))**(2/3)						    		  #Matter + DE domination
    Ha = sqrt_L * (np.cosh(arg) / (np.sinh(arg) + 1e-15))						    		  #Matter + DE dimination

    alpha = Ha						                                                    		  #alpha = actual Ha = adot/a

    # Piecewise Thomson scattering rate -----------------------------------------------------------------------------------------------
    if t < 3.33e-5:											    		  #z = 1100 and beyond
        tau = 2.02 * (Omega_L/Omega_m) * (np.sinh(arg))**(-2) * 1.0e-3
    elif t < 6.56e-5:											    		  #z = 1100 to 700
        tau = ((3.33e-3) * (((Omega_m/Omega_L)**(-1/3)) * ((np.sinh(arg))**(-2/3)) - 1) - 2.33) * (2.02 * (Omega_L/Omega_m) * (np.sinh(arg))**(-2) * 1.0e-3)
    else:												    		  #z = 700 to 0
        tau = 2.02 * (Omega_L/Omega_m) * (np.sinh(arg))**(-2) * 1.0e-6

    eps = 1e-30
    H = Ha
    L = 1/tau
    c = 1
    k = Omega_k * (2.56e-18)**2
    q = np.sqrt(k) / (c * a_t * 2.56e-18 + eps)                                                             		  #Dimensionless qty, since numerator is in sec inverse
      
    calD = Ha + eps
    
    d_theta = 0
    d_phi   = 0
    d_psi_real = 0
    d_psi_imag = 0
    
    # -------------------------------------------------------------------------------------------------------------------
    Tr_R0 = R0_11 + R0_22 + R0_33
    Tr_I0 = I0_11 + I0_22 + I0_33
    Sum_R0_vec = R0_1 + R0_2 + R0_3
    Sum_I0_vec = I0_1 + I0_2 + I0_3

    S_Re = H * R0_0 + (1.0/3.0) * H * Sum_R0_vec + 0.4 * H * Tr_R0
    S_Im = H * I0_0 + (1.0/3.0) * H * Sum_I0_vec + 0.4 * H * Tr_I0 - (1.0/L) * I0_0

    D_real = -H * (R0_0 + (2.0/15.0) * Tr_R0)
    D_imag = -H * (I0_0 + (2.0/15.0) * Tr_I0)

    inv_den = 1.0 / (D_real**2 + D_imag**2 + eps)

    D_minus_N_Re = D_real - R0_0
    D_minus_N_Im = D_imag - I0_0

    inv_D_minus_N_sq = 1.0 / (D_minus_N_Re**2 + D_minus_N_Im**2 + eps)
    C_Re = (D_real * D_minus_N_Re + D_imag * D_minus_N_Im) * inv_D_minus_N_sq
    C_Im = (D_imag * D_minus_N_Re - D_real * D_minus_N_Im) * inv_D_minus_N_sq
    # -------------------------------------------------------------------------------------------------------------------

    dR0_0 = S_Re * C_Re - S_Im * C_Im
    dI0_0 = S_Re * C_Im + S_Im * C_Re
    
    zetaR = (S_Re * D_real + S_Im * D_imag) * inv_den / calD
    zetaI = (S_Im * D_real - S_Re * D_imag) * inv_den / calD
    
    dR0_1 = -(0.6 * H * (zetaR * R0_1 - zetaI * I0_1) + (-0.4 * q * R0_23 + 0.4 * q * R0_32)) - tau * R0_1
    dR0_2 = -(0.6 * H * (zetaR * R0_2 - zetaI * I0_2) + (0.4 * q * R0_13 - 0.4 * q * R0_31)) - tau * R0_2
    dR0_3 = -(0.6 * H * (zetaR * R0_3 - zetaI * I0_3) + (-0.4 * q * R0_12 + 0.4 * q * R0_21)) - tau * R0_3
    dI0_1 = -(0.6 * H * (zetaR * I0_1 + zetaI * R0_1) + (-0.4 * q * I0_23 + 0.4 * q * I0_32)) - tau * (2.0/3.0) * I0_1
    dI0_2 = -(0.6 * H * (zetaR * I0_2 + zetaI * R0_2) + (0.4 * q * I0_13 - 0.4 * q * I0_31)) - tau * (2.0/3.0) * I0_2
    dI0_3 = -(0.6 * H * (zetaR * I0_3 + zetaI * R0_3) + (-0.4 * q * I0_12 + 0.4 * q * I0_21)) - tau * (2.0/3.0) * I0_3

    dR0_11 = -tau * (0.9 * R0_11 + 0.3 * R2_11)
    dR0_12 = -tau * (0.9 * R0_12 + 0.3 * R2_12)
    dR0_13 = -tau * (0.9 * R0_13 + 0.3 * R2_13)
    dR0_21 = -tau * (0.9 * R0_21 + 0.3 * R2_21)
    dR0_22 = -tau * (0.9 * R0_22 + 0.3 * R2_22)
    dR0_23 = -tau * (0.9 * R0_23 + 0.3 * R2_23)
    dR0_31 = -tau * (0.9 * R0_31 + 0.3 * R2_31)
    dR0_32 = -tau * (0.9 * R0_32 + 0.3 * R2_32)
    dR0_33 = -tau * (0.9 * R0_33 + 0.3 * R2_33)

    dI0_11 = -tau * I0_11
    dI0_12 = -tau * I0_12
    dI0_13 = -tau * I0_13
    dI0_21 = -tau * I0_21
    dI0_22 = -tau * I0_22
    dI0_23 = -tau * I0_23
    dI0_31 = -tau * I0_31
    dI0_32 = -tau * I0_32
    dI0_33 = -tau * I0_33

    dR2_11 = -H * (zetaR * R2_11 - zetaI * I2_11) - (0.444444444444444 * H * R2_11 - 2.0 * q * I2_11 - 0.222222222222222 * H * R2_22 - 0.222222222222222 * H * R2_33) - tau * (0.2 * R0_11 + 0.4 * R2_11)
    dR2_12 = -H * (zetaR * R2_12 - zetaI * I2_12) - (0.333333333333333 * H * R2_12 - 2.0 * q * I2_12 + 0.333333333333333 * H * R2_21) - tau * (0.2 * R0_12 + 0.4 * R2_12)
    dR2_13 = -H * (zetaR * R2_13 - zetaI * I2_13) - (0.333333333333333 * H * R2_13 - 2.0 * q * I2_13 + 0.333333333333333 * H * R2_31) - tau * (0.2 * R0_13 + 0.4 * R2_13)
    dR2_21 = -H * (zetaR * R2_21 - zetaI * I2_21) - (0.333333333333333 * H * R2_12 + 0.333333333333333 * H * R2_21 - 2.0 * q * I2_21) - tau * (0.2 * R0_21 + 0.4 * R2_21)
    dR2_22 = -H * (zetaR * R2_22 - zetaI * I2_22) - (-0.222222222222222 * H * R2_11 + 0.444444444444444 * H * R2_22 - 2.0 * q * I2_22 - 0.222222222222222 * H * R2_33) - tau * (0.2 * R0_22 + 0.4 * R2_22)
    dR2_23 = -H * (zetaR * R2_23 - zetaI * I2_23) - (0.333333333333333 * H * R2_23 - 2.0 * q * I2_23 + 0.333333333333333 * H * R2_32) - tau * (0.2 * R0_23 + 0.4 * R2_23)
    dR2_31 = -H * (zetaR * R2_31 - zetaI * I2_31) - (0.333333333333333 * H * R2_13 + 0.333333333333333 * H * R2_31 - 2.0 * q * I2_31) - tau * (0.2 * R0_31 + 0.4 * R2_31)
    dR2_32 = -H * (zetaR * R2_32 - zetaI * I2_32) - (0.333333333333333 * H * R2_23 + 0.333333333333333 * H * R2_32 - 2.0 * q * I2_32) - tau * (0.2 * R0_32 + 0.4 * R2_32)
    dR2_33 = -H * (zetaR * R2_33 - zetaI * I2_33) - (-0.222222222222222 * H * R2_11 - 0.222222222222222 * H * R2_22 + 0.444444444444444 * H * R2_33 - 2.0 * q * I2_33) - tau * (0.2 * R0_33 + 0.4 * R2_33)

    dI2_11 = -H * (zetaR * I2_11 + zetaI * R2_11) - (0.444444444444444 * H * I2_11 + 2.0 * q * R2_11 - 0.222222222222222 * H * I2_22 - 0.222222222222222 * H * I2_33) - tau * I2_11
    dI2_12 = -H * (zetaR * I2_12 + zetaI * R2_12) - (0.333333333333333 * H * I2_12 + 2.0 * q * R2_12 + 0.333333333333333 * H * I2_21) - tau * I2_12
    dI2_13 = -H * (zetaR * I2_13 + zetaI * R2_13) - (0.333333333333333 * H * I2_13 + 2.0 * q * R2_13 + 0.333333333333333 * H * I2_31) - tau * I2_13
    dI2_21 = -H * (zetaR * I2_21 + zetaI * R2_21) - (0.333333333333333 * H * I2_12 + 0.333333333333333 * H * I2_21 + 2.0 * q * R2_21) - tau * I2_21
    dI2_22 = -H * (zetaR * I2_22 + zetaI * R2_22) - (-0.222222222222222 * H * I2_11 + 0.444444444444444 * H * I2_22 + 2.0 * q * R2_22 - 0.222222222222222 * H * I2_33) - tau * I2_22
    dI2_23 = -H * (zetaR * I2_23 + zetaI * R2_23) - (0.333333333333333 * H * I2_23 + 2.0 * q * R2_23 + 0.333333333333333 * H * I2_32) - tau * I2_23
    dI2_31 = -H * (zetaR * I2_31 + zetaI * R2_31) - (0.333333333333333 * H * I2_13 + 0.333333333333333 * H * I2_31 + 2.0 * q * R2_31) - tau * I2_31
    dI2_32 = -H * (zetaR * I2_32 + zetaI * R2_32) - (0.333333333333333 * H * I2_23 + 0.333333333333333 * H * I2_32 + 2.0 * q * R2_32) - tau * I2_32
    dI2_33 = -H * (zetaR * I2_33 + zetaI * R2_33) - (-0.222222222222222 * H * I2_11 - 0.222222222222222 * H * I2_22 + 0.444444444444444 * H * I2_33 + 2.0 * q * R2_33) - tau * I2_33

    #Enforce tracelessness -----------------------------------------------------------------------------------------------
    dR0_33 = -dR0_11 - dR0_22
    dI0_33 = -dI0_11 - dI0_22
    dR2_33 = -dR2_11 - dR2_22
    dI2_33 = -dI2_11 - dI2_22
    
    return np.array([
        dR0_0, dI0_0,
        dR0_1, dR0_2, dR0_3, dI0_1, dI0_2, dI0_3,
        dR0_11, dR0_12, dR0_13, dR0_21, dR0_22, dR0_23, dR0_31, dR0_32, dR0_33,
        dI0_11, dI0_12, dI0_13, dI0_21, dI0_22, dI0_23, dI0_31, dI0_32, dI0_33,
        dR2_11, dR2_12, dR2_13, dR2_21, dR2_22, dR2_23, dR2_31, dR2_32, dR2_33,
        dI2_11, dI2_12, dI2_13, dI2_21, dI2_22, dI2_23, dI2_31, dI2_32, dI2_33,
        d_psi_real, d_psi_imag, d_theta, d_phi
    ])

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

def solve_single_pixel(args):
    idx, theta_0, phi_0, t_eval_pts = args
    
    y0 = np.zeros(48)

    #Plus-polarized tensor mode ((l,m) = (2,2)) -----------------------------------------------------------------------------------------------
    y0[8]  =  1.0e-6  													  # R0_11
    y0[12] = -1.0e-6  													  # R0_22

    y0[46] = theta_0
    y0[47] = phi_0
    
    sol = integrate.solve_ivp(fun=lambda t, y: derivatives_hpc(t, y), t_span=(t_eval_pts[0], t_eval_pts[-1]),  y0=y0,  t_eval=t_eval_pts, method='LSODA', rtol=1e-8, atol=1e-10)
        
    results_T = []
    results_Q = []
    results_U = []
    results_V = []

    for i in range(len(t_eval_pts)):
        state_at_t = sol.y[:, i]
        
        th_f = state_at_t[46]
        ph_f = state_at_t[47]

        T_monopole = state_at_t[0]

        n_vec = np.array([np.sin(th_f)*np.cos(ph_f), np.sin(th_f)*np.sin(ph_f), np.cos(th_f)])
        dipole_vec = state_at_t[2:5]
        T_dipole = np.dot(dipole_vec, n_vec)

        shear_tensor = state_at_t[8:17].reshape(3,3)
        T_quadrupole = n_vec @ shear_tensor @ n_vec

        T_val = T_monopole + T_dipole + T_quadrupole

        P_tensor = state_at_t[26:35].reshape(3,3)
            
        e_th, e_ph = get_basis_vectors(th_f, ph_f)
            
        P_tt = e_th @ P_tensor @ e_th
        P_pp = e_ph @ P_tensor @ e_ph
        P_tp = e_th @ P_tensor @ e_ph
        P_pt = e_ph @ P_tensor @ e_th
            
        Q_raw = P_tt - P_pp
        U_raw = P_tp + P_pt
        #V_raw = P_tp - P_pt
            
        Psi_total = state_at_t[44] + state_at_t[45]
        cos_2psi = np.cos(2 * Psi_total)
        sin_2psi = np.sin(2 * Psi_total)
            
        Q_val = Q_raw * cos_2psi + U_raw * sin_2psi
        U_val = -Q_raw * sin_2psi + U_raw * cos_2psi

        # Full Stokes-V reconstruction from Im(N^0) (CONSISTENCY CHECK IN AMPLITUDE) -----------------------------------------------------------------------------------------------
        I0_0 = state_at_t[1]												  #I0_0
        I0_vec = state_at_t[5:8]											  #I0_1, I0_2, I0_3
        I0_tensor = state_at_t[17:26].reshape(3, 3)									  #I9_11 to I0_33

        k_vec = np.array([
            np.sin(th_f) * np.cos(ph_f),
            np.sin(th_f) * np.sin(ph_f),
            np.cos(th_f)
        ])

        k_tensor = np.outer(k_vec, k_vec) - np.eye(3) / 3.0

        V_val = (
            I0_0
            + np.dot(I0_vec, k_vec)
            + np.sum(I0_tensor * k_tensor)
        )
            
        results_T.append(T_val)
        results_Q.append(Q_val)
        results_U.append(U_val)
        results_V.append(V_val)
        
    return (idx, results_T, results_Q, results_U, results_V)

if __name__ == "__main__":
    
    NSIDE = 32
    T_START = 3.33e-5													  #z = 1100
    T_END = 0.96													  #z = 0
    steps = 4
    eval_times = np.linspace(T_START, T_END, steps)
    
    NPIX = hp.nside2npix(NSIDE)
    theta_arr, phi_arr = hp.pix2ang(NSIDE, np.arange(NPIX))
    
    cores = 8
    print(f"Cores = {cores}", flush=True)
    
    tasks = [(i, theta_arr[i], phi_arr[i], eval_times) for i in range(NPIX)]
    
    T_maps = np.full((steps, NPIX), np.nan)
    Q_maps = np.full((steps, NPIX), np.nan)
    U_maps = np.full((steps, NPIX), np.nan)
    #V_maps = np.full((steps, NPIX), np.nan)
    
    count = 0
    t0 = time.time()
    pbar = tqdm(total=NPIX, desc="Solving pixels", unit="pix", dynamic_ncols=True, smoothing=0.1)
    
    with mp.Pool(processes=cores) as pool:
        for result in pool.imap_unordered(solve_single_pixel, tasks, chunksize=32):
            idx, res_t_list, res_q_list, res_u_list, res_v_list = result 
            for i in range(steps):
                T_maps[i, idx] = res_t_list[i]
                Q_maps[i, idx] = res_q_list[i]
                U_maps[i, idx] = res_u_list[i]      
                #V_maps[i, idx] = res_v_list[i]
            count += 1
            pbar.update(1)
    pbar.close()
    
    T_maps = np.nan_to_num(T_maps, nan=0.0, posinf=0.0, neginf=0.0)
    Q_maps = np.nan_to_num(Q_maps, nan=0.0, posinf=0.0, neginf=0.0)
    U_maps = np.nan_to_num(U_maps, nan=0.0, posinf=0.0, neginf=0.0)
    #V_maps = np.nan_to_num(V_maps, nan=0.0, posinf=0.0, neginf=0.0)
    
    T_cmb_K = 2.725
    T_maps *= T_cmb_K
    Q_maps *= T_cmb_K
    U_maps *= T_cmb_K
    #V_maps *= T_cmb_K
    P_maps = np.sqrt(Q_maps**2 + U_maps**2)

    T_scale = np.nanmax(np.abs(T_maps[0]))
    P_scale = np.nanmax(P_maps[0])
    Q_scale = np.nanmax(np.abs(Q_maps[0]))
    U_scale = np.nanmax(np.abs(U_maps[0]))
    #V_scale = np.nanmax(np.abs(V_maps[0]))

    HUBBLE_TIME_GYR = 13.97 												  #1/H0 conversion factor
    
    plt.rcParams['font.family'] = 'serif'
    output_dir = "maps"
    os.makedirs(output_dir, exist_ok=True)

    #Figure parameters -----------------------------------------------------------------------------------------------
    rows = steps
    cols = 5
    fig = plt.figure(figsize=(24, 4.5 * rows)) 

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
        cb.ax.tick_params(labelsize=10, length=0)
        cb.formatter = ticker.FormatStrFormatter('%.2e')
        cb.update_ticks()

    for i, t_val in enumerate(eval_times):
        real_age = t_val * HUBBLE_TIME_GYR
        
        T_scale = np.nanmax(np.abs(T_maps[i]))
        P_scale = np.nanmax(P_maps[i])
        Q_scale = np.nanmax(np.abs(Q_maps[i]))
        U_scale = np.nanmax(np.abs(U_maps[i]))
        #V_scale = np.nanmax(np.abs(V_maps[i]))
        
        #Reverse row index: Bottom row = steps - 1, Top row = 0 -----------------------------------------------------------------------------------------------
        row_idx = (steps - 1) - i
        
        #Calculate base position (matplotlib subplots are 1-indexed) -----------------------------------------------------------------------------------------------
        base_pos = row_idx * cols
        
        #Flag to determine if we need to draw column titles -----------------------------------------------------------------------------------------------
        is_top = (row_idx == 0)

        #Column 1: Time Text Label -----------------------------------------------------------------------------------------------
        ax_text = fig.add_subplot(rows, cols, base_pos + 1)
        ax_text.axis('off')
        ax_text.text(0.1, 0.5, f"\n{real_age:.2f} Gyrs", 
                     fontsize=18, ha='left', va='center', fontweight='bold')
        
        #Columns 2-5: The Maps -----------------------------------------------------------------------------------------------
        plot_styled_map(T_maps[i], 'turbo', -T_scale, T_scale, "T [K]", base_pos + 2, is_top)
        plot_styled_map(P_maps[i], 'turbo', 0, P_scale, "P [K]", base_pos + 3, is_top)
        plot_styled_map(Q_maps[i], 'turbo', -Q_scale, Q_scale, "Q [K]", base_pos + 4, is_top)
        plot_styled_map(U_maps[i], 'turbo', -U_scale, U_scale, "U [K]", base_pos + 5, is_top)
        #plot_styled_map(V_maps[i], 'turbo', -V_scale, V_scale, "V [K]", base_pos + 6, is_top)

    #Adjust spacing and save the final master grid -----------------------------------------------------------------------------------------------
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    save_path = os.path.join(output_dir, "S3.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Master grid saved to: {save_path}")
