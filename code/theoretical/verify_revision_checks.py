import numpy as np
from scipy.integrate import quad

rng = np.random.default_rng(20261013)
np.set_printoptions(precision=6, suppress=True)

print("A  causal counterexample: martingale preserved, conditional law anticipated")
M1 = np.array([.2, .4, .6, .8])
X0 = (np.abs(M1 - .5) > .25).astype(float)
for v in [0, 1]:
    sel = X0 == v
    print(f"  E[M1|X0={v}]={M1[sel].mean():.3f}  E[(M1-.5)^2|X0={v}]={((M1[sel]-.5)**2).mean():.3f}")
print(f"  E[(M1-.5)^2]={((M1-.5)**2).mean():.3f}")
pj = np.zeros((2, 2))
for m, x in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    pj[m, x] = 2 * (m == x) / 4
print(f"  tilt 2*1(X0=M1): M marginal={pj.sum(1)}  P(M1=X0)={pj[0,0]+pj[1,1]:.2f}")

print("B  reference invariance and budget interval")
q = 4
g = rng.normal(size=q)
w = rng.normal(size=q)
G = g @ g
chihat = g @ w / G
R0 = w @ w - (g @ w) ** 2 / G
B = 0.5 * R0 + 0.07
lo = chihat - np.sqrt((2 * B - R0) / G)
hi = chihat + np.sqrt((2 * B - R0) / G)
grid = np.linspace(-2, 3, 200001)
cost = 0.5 * np.sum((w[None, :] - grid[:, None] * g[None, :]) ** 2, axis=1)
feas = grid[cost <= B]
print(f"  formula [{lo:.6f},{hi:.6f}]  grid [{feas.min():.6f},{feas.max():.6f}]")
for chi in [0.1, 0.7]:
    u = w - chi * g
    print(f"  chi={chi}: reconstructed b identical={np.allclose(chi * g + u, w)}")

print("C  order-one quadrature correction, limit -(1-log2)/4")
f = lambda t: 0.5 * (t - np.log(1 + t))
I = quad(f, 0, 1)[0]
target = -(1 - np.log(2)) / 4
for n in [10**3, 10**4, 10**5, 10**6]:
    k = np.arange(n)
    S = f(k / n).sum()
    print(f"  n={n:7d}  S_n - n I = {S - n * I:+.6f}   target {target:+.6f}")

print("D  factor rho/4 in the balanced regime")
def rand_pd(q):
    A = rng.normal(size=(q, q))
    return A @ A.T + q * np.eye(q)
abar = rand_pd(3)
Bm = rng.normal(size=(3, 3)); Bm = (Bm + Bm.T) / 2
rho = 2.3
rt = np.linalg.cholesky(abar)
def ell(a, ab):
    L = np.linalg.cholesky(ab)
    M = np.linalg.solve(L, np.linalg.solve(L, a).T).T
    s = np.linalg.eigvalsh((M + M.T) / 2)
    return 0.5 * np.sum(s - 1 - np.log(s))
for n in [10**2, 10**4, 10**6]:
    a = rt @ (np.eye(3) + np.sqrt(rho / n) * Bm) @ rt.T
    print(f"  n={n:7d}  n*ell={n * ell(a, abar):.6f}   rho tr(B^2)/4={rho * np.trace(Bm @ Bm) / 4:.6f}")

print("E  innovation energy under shared noise (Schur)")
for _ in range(3):
    A = rand_pd(2); D = rand_pd(2)
    C = 0.3 * rng.normal(size=(2, 2))
    J = np.block([[A, C], [C.T, D]])
    J = J + (abs(min(np.linalg.eigvalsh(J).min(), 0)) + 0.2) * np.eye(4)
    A, C, D = J[:2, :2], J[:2, 2:], J[2:, 2:]
    S = D - C.T @ np.linalg.solve(A, C)
    u = S @ rng.normal(size=2)
    bX = np.linalg.solve(S, u)
    bM = -np.linalg.solve(A, C @ bX)
    beta = np.concatenate([bM, bX])
    drift = J @ beta
    en = 0.5 * beta @ J @ beta
    enf = 0.5 * u @ np.linalg.solve(S, u)
    print(f"  drift=(0,u) ok={np.allclose(drift[:2], 0) and np.allclose(drift[2:], u)}  energy={en:.6f}  u'S^-1 u/2={enf:.6f}")
r = 0.6; u1 = 1.3
print(f"  scalar r={r}: u^2/(2(1-r^2))={u1**2/(2*(1-r**2)):.6f}")

print("F  price pair: formulas, pi*, intervals, anchoring, efficient information")
vL, vH = 30.0, 90.0
L = vH - vL
def Cfun(p, a, b): return vL + L * a * p / (a * p + b * (1 - p))
def Ufun(p): return vL + L * p
def Sfun(p, chi, a, b): return Cfun(p, a, b) + chi * (Ufun(p) - Cfun(p, a, b))
a, b = .2, .8
pstar = np.sqrt(b) / (np.sqrt(a) + np.sqrt(b))
Dstar = L * (np.sqrt(b) - np.sqrt(a)) / (np.sqrt(b) + np.sqrt(a))
grid = np.linspace(1e-4, 1 - 1e-4, 200001)
Dg = Ufun(grid) - Cfun(grid, a, b)
print(f"  pi*={pstar:.6f} (grid {grid[np.argmax(Dg)]:.6f})  D(pi*)={Dstar:.6f} (grid {Dg.max():.6f})")
print(f"  pi=.4: U={Ufun(.4):.4f}  C={Cfun(.4,a,b):.4f}  S(chi=.25)={Sfun(.4,.25,a,b):.4f}")
lo_all, hi_all = [], []
for p in np.linspace(1/3, 5/12, 20001):
    Cp, Dp = Cfun(p, a, b), Ufun(p) - Cfun(p, a, b)
    lo_all.append(max(0.0, (40 - Cp) / Dp)); hi_all.append(min(1.0, (45 - Cp) / Dp))
print(f"  windows U[50,55] S[40,45]: chi in [{min(lo_all):.4f}, {max(hi_all):.4f}]")
z, c = .10, 57.5
for (aa, bb) in [(.2, .8), (.4, .6), (.3, .7)]:
    Uo = (1 - z) * Ufun(.4) + z * c
    So = (1 - z) * Cfun(.4, aa, bb) + z * c
    ph = (Uo - vL) / L
    chih = (So - Cfun(ph, aa, bb)) / (Ufun(ph) - Cfun(ph, aa, bb))
    print(f"  anchoring z=.1 c=57.5 (a,b)=({aa},{bb}): apparent chi={chih:.4f}")
eps = 1e-6
p0, chi0 = .45, .3
Jm = np.array([[L, 0.0],
               [(Sfun(p0+eps,chi0,a,b)-Sfun(p0-eps,chi0,a,b))/(2*eps), Ufun(p0)-Cfun(p0,a,b)]])
Sig = np.array([[1.3, .4], [.4, 2.1]])
Ieff_matrix = 1.0 / np.linalg.inv(Jm.T @ np.linalg.solve(Sig, Jm))[1, 1]
Sp = Jm[1, 0]
Dq = Jm[1, 1]
Ieff_formula = Dq**2 / (Sig[1,1] + (Sp/L)**2 * Sig[0,0] - 2*(Sp/L)*Sig[0,1])
print(f"  efficient information: matrix={Ieff_matrix:.6f}  formula={Ieff_formula:.6f}")

print("G  depth variance is not injective in tau")
for tau in [.25, 4.0, 1.0]:
    print(f"  tau={tau}: Var={tau/(1+tau)**2:.4f}")

print("H  parametric sender-receiver determinant (spot check)")
pi0, tH, tL = .4, .6, .7
muh = pi0*tH/(pi0*tH+(1-pi0)*(1-tL))
mul = pi0*(1-tH)/(pi0*(1-tH)+(1-pi0)*tL)
lam, chi, tau = 1.8, .2, 2.5
alph = 1-np.exp(-tau)
sig = lambda v: 1/(1+np.exp(-v))
def qm(mu, lam, chi):
    zm = 2*((1-chi)*mu+chi*pi0)-1
    return alph*sig(lam*zm)+(1-alph)/2
e = 1e-6
Jq = np.array([[(qm(muh,lam+e,chi)-qm(muh,lam-e,chi))/(2*e), (qm(muh,lam,chi+e)-qm(muh,lam,chi-e))/(2*e)],
               [(qm(mul,lam+e,chi)-qm(mul,lam-e,chi))/(2*e), (qm(mul,lam,chi+e)-qm(mul,lam,chi-e))/(2*e)]])
zh = 2*((1-chi)*muh+chi*pi0)-1
zl = 2*((1-chi)*mul+chi*pi0)-1
det_f = 2*lam*alph**2*sig(lam*zh)*(1-sig(lam*zh))*sig(lam*zl)*(1-sig(lam*zl))*(muh-mul)*(2*pi0-1)
print(f"  numerical det={np.linalg.det(Jq):.8f}  formula={det_f:.8f}")

print("I  joint statements, pull between prices, geometric counterpart")
def Jfun(p, a, b): return a * p / (a * p + b * (1 - p))
def Sgeo(p, chi, a, b):
    qj = Jfun(p, a, b)
    num = qj ** (1 - chi) * p ** chi
    return vL + L * num / (num + (1 - qj) ** (1 - chi) * (1 - p) ** chi)
contrasts = [(.35, .65), (.2, .8), (.05, .95)]
worst = 0.0
for (aa, bb) in contrasts:
    for p in [.25, .4, .6, .8]:
        for chi in [0.0, .15, .25, .6]:
            for zu in [0.0, .1, .3]:
                Sa = Sfun(p, chi, aa, bb)
                Sj = (1 - zu) * Sa + zu * Ufun(p)
                app = (Sj - Cfun(p, aa, bb)) / (Ufun(p) - Cfun(p, aa, bb))
                worst = max(worst, abs(app - (1 - (1 - chi) * (1 - zu))))
                if chi < 1:
                    worst = max(worst, abs((Sj - Sa) / (Ufun(p) - Sa) - zu))
print(f"  pull: largest deviation from 1-(1-chi)(1-z_U) and from the recovered weight = {worst:.2e}")
for chi in [0.0, .15, .25]:
    Sa = Sfun(.4, chi, .2, .8)
    Sj = .9 * Sa + .1 * Ufun(.4)
    print(f"  pull z_U=.10, chi={chi}: apparent contraction={(Sj - Cfun(.4,.2,.8)) / (Ufun(.4) - Cfun(.4,.2,.8)):.4f}")
p0, chi0, aa, bb = .4, .3, .2, .8
tab = np.array([[(1 - p0) * (1 - bb), (1 - p0) * bb], [p0 * (1 - aa), p0 * aa]])
prod = np.outer(tab.sum(1), tab.sum(0))
geo = tab ** (1 - chi0) * prod ** chi0
geo /= geo.sum()
mix = (1 - chi0) * tab + chi0 * prod
print(f"  geometric table: conditional={geo[1,1]/geo[:,1].sum():.6f}  formula={(Sgeo(p0,chi0,aa,bb)-vL)/L:.6f}  high-value marginal={geo[1].sum():.4f} (prior {p0})")
print(f"  mixture table:   conditional={mix[1,1]/mix[:,1].sum():.6f}  formula={(Sfun(p0,chi0,aa,bb)-vL)/L:.6f}  high-value marginal={mix[1].sum():.4f}")
lg = lambda x: np.log(x / (1 - x))
lin = lg((Sgeo(p0, chi0, aa, bb) - vL) / L) - (lg(p0) + (1 - chi0) * np.log(aa / bb))
print(f"  log-odds linearity residual={abs(lin):.2e}")
chis = np.linspace(0, 1, 100001)
for (aa, bb) in contrasts:
    ps = np.sqrt(bb) / (np.sqrt(aa) + np.sqrt(bb))
    gap = L * (np.sqrt(bb) - np.sqrt(aa)) / (np.sqrt(bb) + np.sqrt(aa))
    d4 = Sfun(.4, .5, aa, bb) - Sgeo(.4, .5, aa, bb)
    ds = np.abs(Sfun(ps, chis, aa, bb) - Sgeo(ps, chis, aa, bb)).max()
    print(f"  (a,b)=({aa},{bb}): pi*={ps:.4f}  gap={gap:.3f}  pi_J(pi*)+pi*={Jfun(ps,aa,bb)+ps:.6f}  S-Sgeo at pi=.4, chi=.5: {d4:.3f}  at pi*, chi=.5: {Sfun(ps,.5,aa,bb)-Sgeo(ps,.5,aa,bb):.1e}  max over chi at pi*: {ds:.3f}")
