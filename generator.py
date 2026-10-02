#This code generates Ricci rotation coefficients, Boltzmann coefficients, geodesic equations and implicitly solved real & imaginary components of Zeta using the monopole evolution equation using Cramer's rule

import sympy as sp
from injector import Gam, make_coefficients

r3 = range(1, 4)
t = sp.symbols('t')
theta, phi = sp.symbols('theta phi', real=True)
tau = sp.symbols('tau', positive=True)
s = -1

k_vec = [sp.sin(theta)*sp.cos(phi), sp.sin(theta)*sp.sin(phi), sp.cos(theta)]
a_vec = [sp.cos(theta)*sp.cos(phi), sp.cos(theta)*sp.sin(phi), -sp.sin(theta)]
b_vec = [-sp.sin(phi), sp.cos(phi), 0]

def G(u, l1, l2):
    return Gam[(u, l1, l2)]

def gamma_bar(a):
    return sp.simplify(
        G(a, 0, 0)
        + sum(G(a, 0, j)*k_vec[j-1] for j in r3)
        + sum(G(a, j, 0)*k_vec[j-1] for j in r3)
        + sum(G(a, j, m)*k_vec[j-1]*k_vec[m-1] for j in r3 for m in r3))

def photon_transport():
    gb = {i: gamma_bar(i) for i in r3}

    dtheta_dt = sp.simplify(-sum(a_vec[i-1]*gb[i] for i in r3))
    dphi_dt = sp.simplify(-sum(b_vec[i-1]*gb[i] for i in r3)/sp.sin(theta))

    m_re = lambda i, k: sp.Rational(1, 2)*(a_vec[i-1]*b_vec[k-1] - a_vec[k-1]*b_vec[i-1])
    m_im = lambda i, k: sp.Rational(1, 2)*(a_vec[i-1]*a_vec[k-1] + b_vec[i-1]*b_vec[k-1])

    first_re = sum(m_re(i, k)*G(k, l, i)*k_vec[l-1] for i in r3 for k in r3 for l in r3)
    first_im = sum(m_im(i, k)*G(k, l, i)*k_vec[l-1] for i in r3 for k in r3 for l in r3)
    cot_term = sum(b_vec[l-1]*gb[l] for l in r3)*sp.cot(theta)

    dpsi_dt_re = sp.simplify(first_re - cot_term)
    dpsi_dt_im = sp.simplify(first_im)

    return gb, dtheta_dt, dphi_dt, dpsi_dt_re, dpsi_dt_im

def solve_zeta(gamma0):
    f = lambda name: sp.Function(name, real=True)(t)

    def sym_traceless(prefix):
        n11, n12, n13 = f(prefix + '_11'), f(prefix + '_12'), f(prefix + '_13')
        n22, n23 = f(prefix + '_22'), f(prefix + '_23')
        n33 = -n11 - n22
        return sp.Matrix([[n11, n12, n13], [n12, n22, n23], [n13, n23, n33]])

    R00, I00 = f('ReN0_0'), f('ImN0_0')
    Nk_re = [f(f'ReN0_{k}') for k in r3]
    Nk_im = [f(f'ImN0_{k}') for k in r3]
    Nkl_re = sym_traceless('ReN0')
    Nkl_im = sym_traceless('ImN0')

    C = sum(G(0, m, m) for m in r3)
    A_re = sum(G(0, k, l)*Nkl_re[k-1, l-1] for k in r3 for l in r3)
    A_im = sum(G(0, k, l)*Nkl_im[k-1, l-1] for k in r3 for l in r3)
    B_re = sum(sum(G(k, l, l) for l in r3)*Nk_re[k-1] for k in r3)
    B_im = sum(sum(G(k, l, l) for l in r3)*Nk_im[k-1] for k in r3)

    D_re = sp.Rational(1, 3)*C*R00 + sp.Rational(1, 3)*B_re + sp.Rational(2, 5)*A_re
    D_im = sp.Rational(1, 3)*C*I00 + sp.Rational(1, 3)*B_im + sp.Rational(2, 5)*A_im - tau*I00

    p = sp.Rational(2, 15)*A_re
    q = sp.Rational(2, 15)*A_im

    U_re = s*gamma0*R00 - p
    U_im = s*gamma0*I00 - q
    U2 = U_re**2 + U_im**2

    Re_zeta = sp.simplify((D_re*U_re + D_im*U_im)/U2)
    Im_zeta = sp.simplify((D_im*U_re - D_re*U_im)/U2)
    return Re_zeta, Im_zeta


def print_compact(name, expr, prefix='s', file=None):
    replacements, reduced = sp.cse(expr, symbols=sp.numbered_symbols(prefix))
    print(f"    # {name}", file=file)
    for sym, sub_expr in replacements:
        print(f"    {sym} = {str(sub_expr)}", file=file)
    print(f"    {name} = {str(reduced[0])}", file=file)
    print(file=file)


gamma_i, dtheta_dt, dphi_dt, dpsi_dt_re, dpsi_dt_im = photon_transport()
gamma0 = gamma_bar(0)
Re_zeta, Im_zeta = solve_zeta(gamma0)
        
def report():
    #print("Gamma^a_bc (all 64 components):")
    for (a, b, c), v in Gam.items():
        print(f"    Gamma{a}_{b}{c} = {v}")

    #print("\nBoltzmann coefficients (all components):")
    for key, v in make_coefficients(Gam).items():
        print(f"    {key} = {v}")

    print("\nPhoton transport:")
    print(f"gamma^0 = {gamma0}\n")
    for i in r3:
        print(f"gamma^{i} = {gamma_i[i]}\n")
    print(f"dtheta/dt = {dtheta_dt}\n")
    print(f"dphi/dt = {dphi_dt}\n")
    print(f"Re(dpsi/dt) = {dpsi_dt_re}\n")
    print(f"Im(dpsi/dt) = {dpsi_dt_im}\n")

    print("Zeta:")
    print_compact("Re_Zeta", Re_zeta, prefix="z_r")
    print_compact("Im_Zeta", Im_zeta, prefix="z_i")


if __name__ == "__main__":
    report()
