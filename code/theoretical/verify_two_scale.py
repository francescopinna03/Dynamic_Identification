import numpy as np
from scipy.linalg import sqrtm, expm

rng = np.random.default_rng(20261013)
np.set_printoptions(precision=6, suppress=True)

def rand_pd(q, scale=1.0):
    A = rng.normal(size=(q, q))
    return scale * (A @ A.T + q * np.eye(q))

def rand_sym(q, scale=1.0):
    A = rng.normal(size=(q, q))
    return scale * (A + A.T) / 2

def h_sp(a, abar):
    L = np.linalg.cholesky(abar)
    M = np.linalg.solve(L, np.linalg.solve(L, a).T).T
    s = np.linalg.eigvalsh((M + M.T) / 2)
    if np.min(s) <= 0:
        return np.inf
    return 0.5 * np.sum(s - 1.0 - np.log(s))

def gauss_kl(m0, S0, m1, S1):
    q = len(m0)
    S1i = np.linalg.inv(S1)
    tr = np.trace(S1i @ S0)
    ld = np.log(np.linalg.det(S1i @ S0))
    mq = (m1 - m0) @ S1i @ (m1 - m0)
    return 0.5 * (tr - q - ld + mq)

def bures2(a, b):
    rb = sqrtm(b)
    m = sqrtm(rb @ a @ rb)
    return float(np.trace(a) + np.trace(b) - 2.0 * np.real(np.trace(m)))

def fr_form(alpha, abar):
    X = np.linalg.solve(abar, alpha)
    return 0.25 * np.trace(X @ X)

def bures_form(alpha, abar):
    w, V = np.linalg.eigh(abar)
    At = V.T @ alpha @ V
    G = At / (w[:, None] + w[None, :])
    return 0.5 * np.trace(At @ G)

q = 3
T = 1.0

print("CHECK 1  Gaussian KL formula against h_sp plus drift term")
for _ in range(5):
    abar = rand_pd(q, 0.5)
    a = rand_pd(q, 0.4)
    b = rng.normal(size=q)
    bbar = rng.normal(size=q)
    Delta = 0.01
    kl = gauss_kl(b * Delta, a * Delta, bbar * Delta, abar * Delta)
    rhs = h_sp(a, abar) + 0.5 * Delta * (b - bbar) @ np.linalg.inv(abar) @ (b - bbar)
    print(f"  kl={kl:.10f}  decomp={rhs:.10f}  diff={abs(kl-rhs):.2e}")

print("CHECK 2  smooth flows, H_n/n -> Csp and H_n - n R_n -> Cb")
A0 = rand_pd(q, 0.3)
A1 = rand_sym(q, 0.05)
Ab0 = rand_pd(q, 0.3)
Ab1 = rand_sym(q, 0.03)
B0 = rng.normal(size=q)
B1 = rng.normal(size=q)

def a_t(t): return A0 + np.sin(2 * np.pi * t) * A1
def abar_t(t): return Ab0 + np.cos(np.pi * t) * Ab1
def u_t(t): return B0 + t * B1

def H_n(n):
    Delta = T / n
    Hs = 0.0
    Rs = 0.0
    Cb = 0.0
    for k in range(n):
        t = k * Delta
        hk = h_sp(a_t(t), abar_t(t))
        Hs += hk + 0.5 * Delta * u_t(t) @ np.linalg.inv(abar_t(t)) @ u_t(t)
        Rs += hk
        Cb += 0.5 * Delta * u_t(t) @ np.linalg.inv(abar_t(t)) @ u_t(t)
    return Hs, Rs / n, Cb

from scipy.integrate import quad
Csp = quad(lambda t: h_sp(a_t(t), abar_t(t)), 0, T, limit=200)[0] / T
Cb_lim = quad(lambda t: 0.5 * u_t(t) @ np.linalg.inv(abar_t(t)) @ u_t(t), 0, T, limit=200)[0]
for n in [10, 100, 1000, 10000]:
    Hs, Rn, Cbn = H_n(n)
    print(f"  n={n:6d}  H/n={Hs/n:.8f}  Csp={Csp:.8f}  H-nRn={Hs-n*Rn:.8f}  Cb_n={Cbn:.8f}  Cb={Cb_lim:.8f}")

print("CHECK 3  Taylor expansion of h_sp, constant 1/4 in the Fisher-Rao metric")
abar = rand_pd(q, 0.5)
alpha = rand_sym(q, 0.2)
for eps in [1e-1, 1e-2, 1e-3, 1e-4]:
    val = h_sp(abar + eps * alpha, abar) / eps**2
    print(f"  eps={eps:.0e}  h/eps^2={val:.8f}  FR={fr_form(alpha, abar):.8f}")

print("CHECK 4  local limit at scale sqrt(Delta) and exponent dichotomy")
def H_n_lan(n, gamma):
    Delta = T / n
    Hs = 0.0
    for k in range(n):
        t = k * Delta
        ab = abar_t(t)
        a = ab + Delta**gamma * (0.5 * A1 + 0.1 * np.eye(q) * np.sin(3 * t))
        Hs += h_sp(a, ab) + 0.5 * Delta * u_t(t) @ np.linalg.inv(ab) @ u_t(t)
    return Hs

def alpha_fun(t): return 0.5 * A1 + 0.1 * np.eye(q) * np.sin(3 * t)
lan_cov = quad(lambda t: fr_form(alpha_fun(t), abar_t(t)), 0, T, limit=200)[0]
print(f"  expected limit = Cb + int FR = {Cb_lim + lan_cov:.8f}")
for n in [100, 1000, 10000, 100000]:
    print(f"  n={n:6d}  gamma=0.5  H_n={H_n_lan(n, 0.5):.8f}")
for n in [100, 1000, 10000]:
    print(f"  n={n:6d}  gamma=0.4  H_n={H_n_lan(n, 0.4):.6f}   gamma=0.6  H_n={H_n_lan(n, 0.6):.8f}  (Cb={Cb_lim:.8f})")

print("CHECK 5  Bures tensor against Fisher-Rao, sandwich [vmin, vmax]")
for _ in range(6):
    abar = rand_pd(q, rng.uniform(0.2, 2.0))
    alpha = rand_sym(q, 0.3)
    w = np.linalg.eigvalsh(abar)
    gb = bures_form(alpha, abar)
    gfr = fr_form(alpha, abar)
    ratio = gb / gfr
    print(f"  ratio={ratio:.6f}  in [{w[0]:.6f}, {w[-1]:.6f}]  ok={w[0] - 1e-12 <= ratio <= w[-1] + 1e-12}")
    d2 = bures2(abar + 1e-4 * alpha, abar) / 1e-8
    print(f"    quadratic expansion dB^2/eps^2 = {d2:.6f}  g_B = {gb:.6f}")

print("CHECK 6  global comparability on the cone slice eps<=s<=Lam")
def phi(s):
    s = np.asarray(s, float)
    out = np.where(np.abs(s - 1) < 1e-8, 0.5, (s - 1 - np.log(np.maximum(s, 1e-300))) / (s - 1)**2)
    return out

eps0, Lam = 0.2, 4.0
viol = 0
for trial in range(400):
    x = rng.dirichlet(np.ones(q + 1) * 3)
    x = 0.9 * x + 0.1 / (q + 1)
    kappa = 0.7
    Cx = np.diag(x) - np.outer(x, x)
    n_full = q + 1
    Hh = np.linalg.svd(np.eye(n_full) - np.ones((n_full, n_full)) / n_full)[0][:, :q]
    abar = kappa * Hh.T @ Cx @ Hh
    w = np.linalg.eigvalsh(abar)
    rb = np.real(sqrtm(abar))
    U = np.linalg.qr(rng.normal(size=(q, q)))[0]
    s = rng.uniform(eps0, Lam, size=q)
    a = rb @ U @ np.diag(s) @ U.T @ rb
    h = h_sp(a, abar)
    d2 = bures2(a, abar)
    fro = np.linalg.norm(a - abar, "fro")
    sand_up = fro / (np.sqrt(np.linalg.eigvalsh(a)[0]) + np.sqrt(w[0]))
    sand_lo = np.linalg.norm(a - abar, 2) / (np.sqrt(np.linalg.eigvalsh(a)[-1]) + np.sqrt(w[-1]))
    if not (sand_lo**2 - 1e-9 <= d2 <= sand_up**2 + 1e-9):
        viol += 1
    ra = np.real(sqrtm(a))
    D = ra - rb
    if np.linalg.norm(D, "fro") > fro / (np.sqrt(np.linalg.eigvalsh(a)[0]) + np.sqrt(w[0])) + 1e-9:
        viol += 1
    if d2 > np.linalg.norm(D, "fro")**2 + 1e-9:
        viol += 1
    c_up = 0.5 * phi(eps0) * (1 / w[0]**2) * q * (np.sqrt(Lam * w[-1]) + np.sqrt(w[-1]))**2
    c_lo = 0.5 * phi(Lam) * (1 / w[-1]**2) * (np.sqrt(eps0 * w[0]) + np.sqrt(w[0]))**2
    if not (c_lo * d2 - 1e-9 <= h <= c_up * d2 + 1e-9):
        viol += 1
print(f"  violations over 400 draws: {viol}")

print("CHECK 7  smallest tangent eigenvalue of C(x) >= n delta^2")
worst = np.inf
for _ in range(2000):
    n_full = rng.integers(3, 7)
    delta = rng.uniform(0.01, 1.0 / n_full * 0.8)
    x = rng.dirichlet(np.ones(n_full))
    x = np.maximum(x, delta)
    x = x / x.sum()
    if x.min() < delta - 1e-12:
        continue
    Cx = np.diag(x) - np.outer(x, x)
    Hh = np.linalg.svd(np.eye(n_full) - np.ones((n_full, n_full)) / n_full)[0][:, :n_full - 1]
    lam = np.linalg.eigvalsh(Hh.T @ Cx @ Hh)[0]
    worst = min(worst, lam / (n_full * x.min()**2))
print(f"  minimum over 2000 draws of lam_min / (n x_min^2) = {worst:.6f}  (>=1 expected)")

print("CHECK 8  perspective closure: r h(A/r||abar) -> tr(abar^-1 A)/2")
abar = rand_pd(q, 0.5)
A = rand_pd(q, 0.3)
lim = 0.5 * np.trace(np.linalg.solve(abar, A))
for r in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6]:
    val = r * h_sp(A / r, abar)
    print(f"  r={r:.0e}  r h(A/r)={val:.8f}  limit={lim:.8f}")

print("CHECK 9  degeneracy: dB^2 bounded on the cone, h_sp = inf for singular a")
abar = rand_pd(q, 0.5)
P = np.zeros((q, q))
P[0, 0] = 1.0
rb = np.real(sqrtm(abar))
a_sing = rb @ P @ rb
print(f"  dB^2(a_sing, abar) = {bures2(a_sing, abar):.6f}  tr(abar)(1+sqrt(Lam))^2 bound = {np.trace(abar)*(1+np.sqrt(Lam))**2:.6f}")
print(f"  h_sp(a_sing || abar) = {h_sp(a_sing, abar)}")
print(f"  dB^2(0, abar) = {bures2(np.zeros((q,q))+1e-18*np.eye(q), abar):.6f}  tr(abar) = {np.trace(abar):.6f}")

print("CHECK 10  shadow price: v(c) convex and rho in -dv(c) on a convex instance")
abar = np.array([[0.5]])
k_grid = np.linspace(0.05, 2.5, 60)

def pair_costs(u, s):
    return 0.5 * u**2 / abar[0, 0], 0.5 * (np.sqrt(s * abar[0, 0]) - np.sqrt(abar[0, 0]))**2

def solve_scalarized(rho, target=1.2):
    best = None
    for s in np.linspace(0.05, 4.0, 4001):
        u = target - s
        cb, cB = pair_costs(u, s)
        val = cb + rho * cB
        if best is None or val < best[0]:
            best = (val, u, s, cb, cB)
    return best

def solve_budget(c, target=1.2):
    best = None
    for s in np.linspace(0.05, 4.0, 4001):
        u = target - s
        cb, cB = pair_costs(u, s)
        if cB <= c and (best is None or cb < best[0]):
            best = (cb, u, s, cB)
    return best

cs = np.linspace(0.0005, 0.06, 30)
vs = np.array([solve_budget(c)[0] for c in cs])
conv = np.all(np.diff(vs, 2) >= -1e-7)
print(f"  v(c) nonincreasing={np.all(np.diff(vs) <= 1e-12)}  convex (diff2>=0)={conv}")
rho = 3.0
val, u, s, cb, cB = solve_scalarized(rho)
i = np.argmin(np.abs(cs - cB))
dv = (vs[min(i + 1, len(cs) - 1)] - vs[max(i - 1, 0)]) / (cs[min(i + 1, len(cs) - 1)] - cs[max(i - 1, 0)])
print(f"  rho={rho}  cB*={cB:.5f}  estimated -v'(c)={-dv:.4f}")
