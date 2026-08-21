# This code lets you supply the veirbein matrix and calculates the Ricci rotation coefficients (Gamma's) from it and
# then the Boltzmann coefficients (A, B, C, ..., K) from the Gamma's

import sympy as sp

def unified_calculation(coords, vierbein_matrix):
    """
    1. Calculates Metric & Ricci Rotation Coefficients from Vierbein.
    2. Uses those coefficients to calculate Boltzmann Coefficients.
    """
    dim = len(coords)
    
    # =========================================================================
    # PART 1: GEOMETRY & RICCI ROTATION COEFFICIENTS
    # =========================================================================
    
    # 1. Define Frame Metric (Minkowski)
    eta = sp.diag(-1, 1, 1, 1) if dim == 4 else sp.eye(dim)

    # 2. Compute Inverse Vierbein and Spacetime Metric
    vierbein_inv = vierbein_matrix.inv()
    g_metric = sp.zeros(dim, dim)
    
    for mu in range(dim):
        for nu in range(dim):
            val = 0
            for a in range(dim):
                for b in range(dim):
                    val += eta[a, b] * vierbein_matrix[a, mu] * vierbein_matrix[b, nu]
            g_metric[mu, nu] = sp.simplify(val)

    # --- DISPLAY INPUTS ---
    print("\n" + "="*60)
    print("1. GEOMETRY INPUTS")
    print("="*60)
    print("Input Vierbein (e^a_mu):")
    sp.pprint(vierbein_matrix)
    print("\nDerived Metric (g_munu):")
    sp.pprint(g_metric)
    print("-" * 60 + "\n")

    g_inv = g_metric.inv()

    # 3. Compute Christoffel Symbols
    christoffel = [[[0]*dim for _ in range(dim)] for _ in range(dim)]
    for k in range(dim):
        for i in range(dim):
            for j in range(dim):
                val = 0
                for l in range(dim):
                    term = sp.diff(g_metric[i, l], coords[j]) + \
                           sp.diff(g_metric[j, l], coords[i]) - \
                           sp.diff(g_metric[i, j], coords[l])
                    val += 0.5 * g_inv[k, l] * term
                christoffel[k][i][j] = sp.simplify(val)

    # 4. Compute Covariant Derivative of Vierbein: e^a_{i;j}
    cov_deriv_e = [[[0]*dim for _ in range(dim)] for _ in range(dim)]
    for a in range(dim):
        for i in range(dim):
            for j in range(dim):
                partial = sp.diff(vierbein_matrix[a, i], coords[j])
                connection = sum(christoffel[k][i][j] * vierbein_matrix[a, k] for k in range(dim))
                cov_deriv_e[a][i][j] = sp.simplify(partial - connection)

    # 5. Compute Ricci Rotation Coefficients: Gamma^a_bc
    # Store in dictionary: gamma[(a, b, c)] = value
    gamma = {}
    
    print("="*60)
    print("2. CALCULATED RICCI ROTATION COEFFICIENTS (Gamma)")
    print("="*60)
    
    found_gamma = False
    # Check all permutations
    for a in range(dim):
        for b in range(dim):
            for c in range(dim):
                val = 0
                for i in range(dim):
                    for j in range(dim):
                        # Formula: - e^i_c * e^a_{i;j} * e^j_b
                        val += -1 * vierbein_inv[i, c] * cov_deriv_e[a][i][j] * vierbein_inv[j, b]
                
                res = sp.simplify(val)
                if res != 0:
                    found_gamma = True
                    gamma[(a, b, c)] = res
                    # Print immediately in LaTeX format
                    print(f"\\Gamma^{{{a}}}_{{{b}{c}}} = {sp.latex(res)}")
    
    if not found_gamma:
        print("All Gamma coefficients are zero.")
    print("-" * 60 + "\n")

    # =========================================================================
    # PART 2: BOLTZMANN COEFFICIENTS
    # =========================================================================

    # Helper functions for Part 2
    def G(u, l1, l2):
        return gamma.get((u, l1, l2), 0)

    def delta(i, j):
        return 1 if i == j else 0

    def epsilon(i, j, k):
        if (i, j, k) in [(1, 2, 3), (2, 3, 1), (3, 1, 2)]: return 1
        if (i, j, k) in [(3, 2, 1), (2, 1, 3), (1, 3, 2)]: return -1
        return 0

    results = {}
    r3 = range(1, 4) # Spatial indices 1, 2, 3

    # --- A_hat^k_i ---
    for k in r3:
        for i in r3:
            sum_G0ll = sum(G(0, l, l) for l in r3)
            term = (1/5) * (G(0, i, k) - G(0, k, i) + sum_G0ll * delta(i, k))
            res = sp.simplify(term)
            if res != 0: results[f"\\hat{{A}}^{{{k}}}_{{{i}}}"] = res

    # --- B_hat^k_i ---
    for k in r3:
        for i in r3:
            sum_Gll0 = sum(G(l, l, 0) for l in r3)
            term = -G(k, 0, i) + (1/5)*(G(0, k, i) - 4*G(0, i, k) + sum_Gll0 * delta(i, k))
            res = sp.simplify(term)
            if res != 0: results[f"\\hat{{B}}^{{{k}}}_{{{i}}}"] = res

    # --- C_hat^kl_i ---
    for k in r3:
        for l in r3:
            for i in r3:
                sum_Gkmm = sum(G(k, m, m) for m in r3)
                term = -(2/5) * (G(k, l, i) + sum_Gkmm * delta(i, l) + 3*G(k, 0, 0)*delta(l, i))
                res = sp.simplify(term)
                if res != 0: results[f"\\hat{{C}}^{{{k}{l}}}_{{{i}}}"] = res

    # --- D_hat^k_ij ---
    for k in r3:
        for i in r3:
            for j in r3:
                term = (1/3)*G(0, 0, k)*delta(i, j) \
                       - (1/2)*G(0, 0, i)*delta(k, j) \
                       - (1/2)*G(0, 0, j)*delta(k, i)
                res = sp.simplify(term)
                if res != 0: results[f"\\hat{{D}}^{{{k}}}_{{{i}{j}}}"] = res

    # --- E_hat_ij ---
    for i in r3:
        for j in r3:
            sum_G0ll = sum(G(0, l, l) for l in r3)
            term = (1/3)*sum_G0ll*delta(i, j) - (1/2)*G(0, i, j) - (1/2)*G(0, j, i)
            res = sp.simplify(term)
            if res != 0: results[f"\\hat{{E}}_{{{i}{j}}}"] = res

    # --- H_hat^k_ij ---
    for k in r3:
        for i in r3:
            for j in r3:
                sum_Gkmm = sum(G(k, m, m) for m in r3)
                term = -0.5*G(k, i, j) - 0.5*G(k, j, i) \
                       + (1/3)*sum_Gkmm*delta(i, j) \
                       + 0.5*G(j, 0, 0)*delta(i, k) \
                       + 0.5*G(i, 0, 0)*delta(j, k) \
                       - (1/3)*G(k, 0, 0)*delta(i, j)
                res = sp.simplify(term)
                if res != 0: results[f"\\hat{{H}}^{{{k}}}_{{{i}{j}}}"] = res
                
    # --- K_hat^kl_ij ---
    for k in r3:
        for l in r3:
            for i in r3:
                for j in r3:
                    levi_term = 0
                    for r in r3:
                        for s in r3:
                            for t_idx in r3:
                                levi_term += G(s, t_idx, r) * epsilon(r, s, t_idx)
                    
                    term = -(2/9)*G(0, k, l)*delta(i, j) \
                           + (1/3)*(G(k, 0, i) + G(k, i, 0))*delta(l, j) \
                           + (1/3)*(G(j, 0, k) + G(j, k, 0))*delta(l, i) \
                           + (sp.I / 3)*delta(k, i)*delta(l, j)*levi_term
                    
                    res = sp.simplify(term)
                    if res != 0: results[f"\\hat{{K}}^{{{k}{l}}}_{{{i}{j}}}"] = res

    # --- OUTPUT ---
    print("="*60)
    print("3. CALCULATED BOLTZMANN COEFFICIENTS")
    print("="*60)
    
    if not results:
        print("All Boltzmann coefficients are zero.")
    else:
        for key, val in results.items():
            print(f"{key} = {sp.latex(val)}")
    print("="*60)


# =============================================================================
# USER INPUT SECTION
# =============================================================================

# 1. DEFINE COORDINATES
t, x, y, z = sp.symbols('t x y z')
coords = [t, x, y, z]

# 2. DEFINE PARAMETERS
# 'a' is scale factor, 'k' curvature, 'c' speed of light
a = sp.Function('a')(t)
k, c = sp.symbols('k c')

# 3. DEFINE VIERBEIN MATRIX
# Rows = frame index 'a' (0..3), Cols = spacetime index 'mu' (0..3)
# Note: Ensure you use sp.exp(), sp.sin(), etc.

# Nil geometry
vierbein = sp.Matrix([
    [1, 0, 0, 0],
    [0, a, 0, 0],
    [0, 0, a, 0],
    [0, 0, -a*x*sp.sqrt(-k)/c, a]
])

unified_calculation(coords, vierbein)
