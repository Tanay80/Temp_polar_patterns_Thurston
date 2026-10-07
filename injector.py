#This code lets you supply ypur spacetime metrices

import sympy as sp

R = sp.Rational
dim = 4
ETA = [-1, 1, 1, 1]

def make_gamma(entries):
    g = {(a, b, c): sp.Integer(0) for a in range(dim) for b in range(dim) for c in range(dim)}
    for (a, b, c), v in entries.items():
        g[(a, b, c)] += v
        g[(a, c, b)] -= v
    return g

def make_Gamma(g):
    Gam = {}
    for d in range(dim):
        for b in range(dim):
            for c in range(dim):
                Gam[(d, b, c)] = sp.simplify(R(1, 2) * (
                    g[(d, b, c)]
                    + ETA[c]*ETA[d]*g[(c, d, b)]
                    - ETA[b]*ETA[d]*g[(b, c, d)]))
    for d in range(dim):
        for b in range(dim):
            for c in range(dim):
                tors = sp.simplify(Gam[(d, b, c)] - Gam[(d, c, b)] - g[(d, b, c)])
                comp = sp.simplify(ETA[d]*Gam[(d, b, c)] + ETA[c]*Gam[(c, b, d)])
                if tors != 0 or comp != 0:
                    print(f"CHECK FAILED at ({d},{b},{c}): torsion={tors}, compat={comp}")
    return Gam

def make_coefficients(Gam):
    dl = lambda i, j: 1 if i == j else 0
    r3 = range(1, 4)
    out = {}

    for k in r3:
        for i in r3:
            out[f"A{k}_{i}"] = R(1, 5)*(
                Gam[(0, i, k)] - Gam[(0, k, i)]
                + sum(Gam[(0, m, m)] for m in r3)*dl(i, k))
            out[f"B{k}_{i}"] = -Gam[(k, 0, i)] + R(1, 5)*(
                Gam[(0, k, i)] - 4*Gam[(0, i, k)]
                + sum(Gam[(m, m, 0)] for m in r3)*dl(i, k))

    for k in r3:
        for l in r3:
            for i in r3:
                out[f"C{k}{l}_{i}"] = -R(2, 5)*(
                    Gam[(k, l, i)]
                    + sum(Gam[(k, m, m)] for m in r3)*dl(i, l)
                    + 3*Gam[(k, 0, 0)]*dl(l, i))

    for k in r3:
        for i in r3:
            for j in r3:
                out[f"D{k}_{i}{j}"] = (
                    R(1, 3)*Gam[(0, 0, k)]*dl(i, j)
                    - R(1, 2)*Gam[(0, 0, i)]*dl(k, j)
                    - R(1, 2)*Gam[(0, 0, j)]*dl(k, i))

    for i in r3:
        for j in r3:
            out[f"E_{i}{j}"] = (
                R(1, 3)*sum(Gam[(0, m, m)] for m in r3)*dl(i, j)
                - R(1, 2)*Gam[(0, i, j)]
                - R(1, 2)*Gam[(0, j, i)])

    for k in r3:
        for l in r3:
            for i in r3:
                for j in r3:
                    out[f"F{k}{l}_{i}{j}"] = (
                        R(2, 21)*Gam[(0, k, l)]*dl(i, j)
                        - R(1, 7)*(
                            dl(k, i)*dl(l, j)*sum(Gam[(0, m, m)] for m in r3)
                            + Gam[(0, k, i)]*dl(j, l)
                            + Gam[(0, k, j)]*dl(i, l)))

    for k in r3:
        for l in r3:
            for i in r3:
                for j in r3:
                    out[f"G{k}{l}_{i}{j}"] = (
                        R(2, 21)*Gam[(0, k, l)]*dl(i, j)
                        + R(2, 7)*(Gam[(0, k, i)]*dl(j, l) + Gam[(0, k, j)]*dl(i, l))
                        - Gam[(k, 0, i)]*dl(l, j)
                        - Gam[(k, 0, j)]*dl(l, i)
                        - R(5, 7)*(Gam[(0, i, k)]*dl(j, l) + Gam[(0, j, k)]*dl(i, l))
                        + R(2, 7)*sum(Gam[(0, m, m)] for m in r3)*dl(i, k)*dl(j, l))

    for k in r3:
        for i in r3:
            for j in r3:
                out[f"H{k}_{i}{j}"] = (
                    -R(1, 2)*Gam[(k, i, j)]
                    - R(1, 2)*Gam[(k, j, i)]
                    + R(1, 3)*sum(Gam[(k, m, m)] for m in r3)*dl(i, j)
                    + R(1, 2)*Gam[(j, 0, 0)]*dl(i, k)
                    + R(1, 2)*Gam[(i, 0, 0)]*dl(j, k)
                    - R(1, 3)*Gam[(k, 0, 0)]*dl(i, j))

    for k in r3:
        for l in r3:
            for i in r3:
                for j in r3:
                    out[f"RK{k}{l}_{i}{j}"] = (
                        -R(2, 9)*Gam[(0, k, l)]*dl(i, j)
                        + R(1, 3)*(Gam[(k, 0, i)] + Gam[(k, i, 0)])*dl(l, j)
                        + R(1, 3)*(Gam[(j, 0, k)] + Gam[(j, k, 0)])*dl(l, i))
                    out[f"IK{k}{l}_{i}{j}"] = (
                        R(1, 3)*dl(k, i)*dl(l, j)
                        * sum(Gam[(s, t, r)]*sp.LeviCivita(r, s, t)
                              for r in r3 for s in r3 for t in r3))

    def P(i, j, m, n):
        return R(1, 2)*(dl(i, m)*dl(j, n) + dl(i, n)*dl(j, m)) - R(1, 3)*dl(i, j)*dl(m, n)

    raw = out
    proj = dict(raw)

    for k in r3:
        for l in r3:
            for i in r3:
                proj[f"C{k}{l}_{i}"] = sum(
                    P(k, l, p, q)*raw[f"C{p}{q}_{i}"]
                    for p in r3 for q in r3 if P(k, l, p, q) != 0)

    for name in ("D", "H"):
        for k in r3:
            for i in r3:
                for j in r3:
                    proj[f"{name}{k}_{i}{j}"] = sum(
                        P(i, j, m, n)*raw[f"{name}{k}_{m}{n}"]
                        for m in r3 for n in r3 if P(i, j, m, n) != 0)

    for i in r3:
        for j in r3:
            proj[f"E_{i}{j}"] = sum(
                P(i, j, m, n)*raw[f"E_{m}{n}"]
                for m in r3 for n in r3 if P(i, j, m, n) != 0)

    for name in ("F", "G", "RK", "IK"):
        for k in r3:
            for l in r3:
                for i in r3:
                    for j in r3:
                        s = 0
                        for p in r3:
                            for q in r3:
                                w1 = P(k, l, p, q)
                                if w1 == 0:
                                    continue
                                for m in r3:
                                    for n in r3:
                                        w2 = P(i, j, m, n)
                                        if w2 == 0:
                                            continue
                                        s += w1*w2*raw[f"{name}{p}{q}_{m}{n}"]
                        proj[f"{name}{k}{l}_{i}{j}"] = s

    return {key: sp.simplify(v) for key, v in proj.items()}

def run(entries):
    g = make_gamma(entries)
    Gam = make_Gamma(g)
    coef = make_coefficients(Gam)
    
    r3 = range(1, 4)
    for name in ("F", "G", "RK", "IK"):
        for k in r3:
            for l in r3:
                for i in r3:
                    for j in r3:
                        v = coef[f"{name}{k}{l}_{i}{j}"]
                        assert sp.simplify(v - coef[f"{name}{l}{k}_{i}{j}"]) == 0  # symmetric in kl
                        assert sp.simplify(v - coef[f"{name}{k}{l}_{j}{i}"]) == 0  # symmetric in ij
        for i in r3:
            for j in r3:
                assert sp.simplify(sum(coef[f"{name}{k}{k}_{i}{j}"] for k in r3)) == 0  # traceless in kl
        for k in r3:
            for l in r3:
                assert sp.simplify(sum(coef[f"{name}{k}{l}_{i}{i}"] for i in r3)) == 0  # traceless in ij
    print("STF check passed for F, G, RK, IK")

    print("Gamma^a_bc (all 64 components):")
    for (a, b, c), v in Gam.items():
        print(f"  Gamma{a}_{b}{c} = {v}")
    print("\nBoltzmann coefficients (all components):")
    for key, v in coef.items():
        print(f"  {key} = {v}")
    return Gam, coef

H, chi, theta, phi, x, y, z = sp.symbols("H chi theta phi x y z", real=True)
a = sp.symbols("a", positive=True)

#1. R3
#entries = {
#    (1, 0, 1): -H,
#    (2, 0, 2): -H,
#    (3, 0, 3): -H,
#    (2, 1, 2): -1/(a*chi),
#    (3, 1, 3): -1/(a*chi),
#    (3, 2, 3): -1/(a*chi*sp.tan(theta)),
#}

#2. S3
#k = sp.symbols("k", positive=True)
#entries = {
#    (1, 0, 1): -H,
#    (2, 0, 2): -H,
#    (3, 0, 3): -H,
#    (2, 1, 2): -(sp.sqrt(k))/(a*sp.tan(chi*sp.sqrt(k))),
#    (3, 1, 3): -(sp.sqrt(k))/(a*sp.tan(chi*sp.sqrt(k))),
#    (3, 2, 3): -(sp.sqrt(k))/(a*sp.sin(chi*sp.sqrt(k))*sp.tan(theta)),
#}

#3. H3
#k = sp.symbols("k", negative=True)
#entries = {
#    (1, 0, 1): -H,
#    (2, 0, 2): -H,
#    (3, 0, 3): -H,
#    (2, 1, 2): -(sp.sqrt(-k))/(a*sp.tanh(chi*sp.sqrt(-k))),
#    (3, 1, 3): -(sp.sqrt(-k))/(a*sp.tanh(chi*sp.sqrt(-k))),
#    (3, 2, 3): -(sp.sqrt(-k))/(a*sp.sinh(chi*sp.sqrt(-k))*sp.tan(theta)),
#}

#4. RS2
#k = sp.symbols("k", positive=True)
#entries = {
#    (1, 0, 1): -H,
#    (2, 0, 2): -H,
#    (3, 0, 3): -H,
#    (2, 1, 2): -(sp.sqrt(k))/(a*sp.tan(chi*sp.sqrt(k))),
#}

#5. RH2
#k = sp.symbols("k", negative=True)
#entries = {
#    (1, 0, 1): -H,
#    (2, 0, 2): -H,
#    (3, 0, 3): -H,
#    (2, 1, 2): -(sp.sqrt(-k))/(a*sp.tanh(chi*sp.sqrt(-k))),
#}

#6. UH2
#k = sp.symbols("k", negative=True)
#entries = {
#    (1, 0, 1): -H,
#    (2, 0, 2): -H,
#    (3, 0, 3): -H,
#    (2, 1, 2): -(sp.sqrt(-k)*sp.tanh(x*sp.sqrt(-k)))/a,
#    (3, 1, 2): -sp.sqrt(-k)/a,
#}

#7. Nil
#k = sp.symbols("k", negative=True)
#entries = {
#    (1, 0, 1): -H,
#    (2, 0, 2): -H,
#    (3, 0, 3): -H,
#    (3, 1, 2): -sp.sqrt(-k)/a,
#}

#8. Solv
k = sp.symbols("k", negative=True)
entries = {
    (1, 0, 1): -H,
    (2, 0, 2): -H,
    (3, 0, 3): -H,
    (1, 3, 1): -sp.sqrt(-k)/a,
    (2, 3, 2): sp.sqrt(-k)/a,
}

Gam, coef = run(entries)
