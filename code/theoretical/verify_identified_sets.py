import numpy as np
import cvxpy as cp
from scipy.stats import chi2

rng = np.random.default_rng(20261013)
np.set_printoptions(precision=5, suppress=True)

x = np.array([.25, .5, .75])
kappa = .48
abar = kappa * x * (1 - x)
Delta = .5
L = 2
T = L * Delta
dx = .25
b0 = .5 * (.4 - x)
g = .05 + .1 * (x - .5)
pi0 = np.array([.25, .5, .25])
TOL_ROOT = 1e-7
TOL_CELL = 1e-4
TOL_RANK = 1e-10

def idx(i, j, l):
    return 9 * i + 3 * j + l

Mr0 = np.zeros((3, 27)); Mq0 = np.zeros((3, 27)); Mr1 = np.zeros((3, 27)); Mq1 = np.zeros((3, 27))
Mrh = np.zeros((9, 27)); Mqh = np.zeros((9, 27))
for i in range(3):
    for j in range(3):
        for l in range(3):
            k = idx(i, j, l)
            Mr0[i, k] = 1; Mq0[i, k] = (x[j] - x[i]) / Delta
            Mr1[j, k] = 1; Mq1[j, k] = (x[l] - x[j]) / Delta
            Mrh[3 * i + j, k] = 1; Mqh[3 * i + j, k] = (x[l] - x[j]) / Delta

pl1 = np.array([.8, .5, .2])
pl2 = np.array([.85, .5, .15])
K = np.zeros((4, 27))
for i in range(3):
    for j in range(3):
        for l in range(3):
            for c1 in range(2):
                for c2 in range(2):
                    K[2 * c1 + c2, idx(i, j, l)] = (pl1[j] if c1 == 0 else 1 - pl1[j]) * (pl2[l] if c2 == 0 else 1 - pl2[l])
K02 = np.zeros((9, 27))
for i in range(3):
    for j in range(3):
        for l in range(3):
            K02[3 * i + l, idx(i, j, l)] = 1

def transitions(b):
    Tm = np.zeros((3, 3))
    for i in range(3):
        m = b[i] * Delta / dx
        s = abar[i] * Delta / dx ** 2
        pu, pd = (s + m) / 2, (s - m) / 2
        Tm[i, i] = 1 - pu - pd
        if i + 1 < 3: Tm[i, i + 1] += pu
        else: Tm[i, i] += pu
        if i - 1 >= 0: Tm[i, i - 1] += pd
        else: Tm[i, i] += pd
    return Tm

def effective(b):
    return (transitions(b) * (x[None, :] - x[:, None])).sum(1) / Delta

bref = effective(b0)
gref = effective(b0 + g) - bref

def chain(T0, T1):
    P = np.zeros(27)
    for i in range(3):
        for j in range(3):
            for l in range(3):
                P[idx(i, j, l)] = pi0[i] * T0[i, j] * T1[j, l]
    return P

def path_masses(b_steps):
    return chain(transitions(b_steps[0]), transitions(b_steps[1]))

def cost_np(P, chi, prefix=True):
    r0, q0 = Mr0 @ P, Mq0 @ P
    r1, q1 = (Mrh @ P, Mqh @ P) if prefix else (Mr1 @ P, Mq1 @ P)
    c = 0.0
    for i in range(3):
        c += Delta * (q0[i] - r0[i] * (bref[i] + chi * gref[i])) ** 2 / (2 * r0[i] * abar[i])
    for h in range(len(r1)):
        j = h % 3
        if r1[h] > 0:
            c += Delta * (q1[h] - r1[h] * (bref[j] + chi * gref[j])) ** 2 / (2 * r1[h] * abar[j])
    return c

def cost_cp(P, chi, prefix=True):
    r0, q0 = Mr0 @ P, Mq0 @ P
    r1, q1 = (Mrh @ P, Mqh @ P) if prefix else (Mr1 @ P, Mq1 @ P)
    terms = []
    for i in range(3):
        terms.append(Delta * cp.quad_over_lin(q0[i] - r0[i] * bref[i] - chi * gref[i] * r0[i], r0[i]) / (2 * abar[i]))
    for h in range(9 if prefix else 3):
        j = h % 3
        terms.append(Delta * cp.quad_over_lin(q1[h] - r1[h] * bref[j] - chi * gref[j] * r1[h], r1[h]) / (2 * abar[j]))
    return cp.sum(terms)

def feasible(P):
    return [P >= 0, cp.sum(P) == 1, Mr0 @ P == pi0]

class Profile:
    def __init__(self, kernel, prefix=True):
        self.chi = cp.Parameter()
        self.p = cp.Parameter(kernel.shape[0])
        self.P = cp.Variable(27)
        self.con = feasible(self.P) + [kernel @ self.P == self.p]
        self.prob = cp.Problem(cp.Minimize(cost_cp(self.P, self.chi, prefix)), self.con)
    def __call__(self, p, chi):
        self.p.value = p
        self.chi.value = chi
        self.prob.solve(solver=cp.CLARABEL)
        return self.prob.value

w_profile = Profile(K)
w_state = Profile(K, prefix=False)

def relaxed_terms(P, t0, t1):
    r0, q0, r1, q1 = Mr0 @ P, Mq0 @ P, Mrh @ P, Mqh @ P
    terms = []
    for i in range(3):
        terms.append(Delta * cp.quad_over_lin(q0[i] - r0[i] * bref[i] - t0[i] * gref[i], r0[i]) / (2 * abar[i]))
    for h in range(9):
        j = h % 3
        terms.append(Delta * cp.quad_over_lin(q1[h] - r1[h] * bref[j] - t1[h] * gref[j], r1[h]) / (2 * abar[j]))
    return cp.sum(terms), [t0 >= 0, t1 >= 0, t0 <= r0, t1 <= r1], Delta * (cp.sum(t0) + cp.sum(t1)) / T

def w_relaxed(p, chi):
    P = cp.Variable(27); t0 = cp.Variable(3); t1 = cp.Variable(9)
    obj, cons, avg = relaxed_terms(P, t0, t1)
    prob = cp.Problem(cp.Minimize(obj), feasible(P) + cons + [K @ P == p, avg == chi])
    prob.solve(solver=cp.CLARABEL)
    return prob.value

def relaxed_endpoints(p, B):
    out = []
    for sense in [cp.Minimize, cp.Maximize]:
        P = cp.Variable(27); t0 = cp.Variable(3); t1 = cp.Variable(9)
        obj, cons, avg = relaxed_terms(P, t0, t1)
        prob = cp.Problem(sense(avg), feasible(P) + cons + [K @ P == p, obj <= B])
        prob.solve(solver=cp.CLARABEL)
        out.append(prob.value)
    return out

def root(p, B, inside, outside):
    while outside - inside > TOL_ROOT or inside - outside > TOL_ROOT:
        mid = (inside + outside) / 2
        if w_profile(p, mid) <= B: inside = mid
        else: outside = mid
    return (inside + outside) / 2

def merge(cells):
    out = []
    for a, b in sorted(cells):
        if out and a <= out[-1][1] + 1e-12: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out

def certify(p, B, lip, nodes, roots):
    sB = np.sqrt(B)
    pending = [(nodes[k], nodes[k + 1], roots[k], roots[k + 1]) for k in range(len(nodes) - 1)]
    inner, outer, solves = [], [], 0
    while pending:
        split = []
        for a, b, sa, sb in pending:
            if (sa + sb + lip * (b - a)) / 2 <= sB:
                inner.append((a, b)); outer.append((a, b))
            elif (sa + sb - lip * (b - a)) / 2 > sB:
                continue
            elif b - a <= TOL_CELL:
                outer.append((a, b))
            else:
                mid = (a + b) / 2
                sm = np.sqrt(max(w_profile(p, mid), 0.0)); solves += 1
                split += [(a, mid, sa, sm), (mid, b, sm, sb)]
        pending = split
    return merge(inner), merge(outer), solves

chi_star = .4
u_star = np.array([.01, -.015, .005])
b_star = b0 + chi_star * g + u_star
P_star = path_masses([b_star, b_star])
p_true = K @ P_star
C_star = cost_np(P_star, chi_star)
B = C_star
grid = np.linspace(0, 1, 41)
w = np.array([w_profile(p_true, c) for c in grid])
ws = np.array([w_state(p_true, c) for c in grid])
print("TOLERANCES  conic solver CLARABEL at default accuracy  root bisection=%.0e  certified cells=%.0e  singular values=%.0e" % (TOL_ROOT, TOL_CELL, TOL_RANK))
print("INSTANCE  reference drift=%s  load=%s" % (bref, gref))
print("          generating law: cursedness=%.2f  cost=%.6f  budget=%.6f  p=%s" % (chi_star, C_star, B, p_true))
print("          profile minimum on the grid=%.6f at chi=%.3f  profile at the generating cursedness=%.6f" % (w.min(), grid[w.argmin()], w_profile(p_true, chi_star)))
print("PREFIX    prefix cost against state cost of the generating law: %.2e" % abs(cost_np(P_star, chi_star) - cost_np(P_star, chi_star, prefix=False)))
print("          largest gap between prefix and state profiles on the grid: %.2e" % np.max(np.abs(w - ws)))
T_tilt = [transitions(b0 + chi_star * g + s * np.array([0, .08, 0])) for s in (1, 0, -1)]
P_hist = np.zeros(27)
for i in range(3):
    for j in range(3):
        for l in range(3):
            P_hist[idx(i, j, l)] = pi0[i] * transitions(b0 + chi_star * g)[i, j] * T_tilt[i][j, l]
print("          history-dependent law: prefix cost=%.6f  state cost=%.6f" % (cost_np(P_hist, chi_star), cost_np(P_hist, chi_star, prefix=False)))
w02_prefix = Profile(K02); w02_state = Profile(K02, prefix=False)
print("          kernel on initial and final state at chi=%.1f: prefix profile=%.6f  state profile=%.6f" % (chi_star, w02_prefix(K02 @ P_hist, chi_star), w02_state(K02 @ P_hist, chi_star)))
wb = np.array([w_relaxed(p_true, c) for c in grid])
inset = grid[w <= B]
print("PROFILE   w(chi) at nodes 0,.2,.4,.6,.8,1 =", w[::8])
print("          negative second differences:", int(np.sum(np.diff(w, 2) < -1e-9)), "out of", len(w) - 2)
print("          admitted nodes: first=%.4f last=%.4f  count=%d  consecutive=%s" % (inset.min(), inset.max(), len(inset), np.all(np.diff(inset) < 0.026)))
chi_minus = root(p_true, B, inset.min(), inset.min() - .025) if inset.min() > 0 else 0.0
chi_plus = root(p_true, B, inset.max(), inset.max() + .025) if inset.max() < 1 else 1.0
print("          threshold crossings: lower=%.6f upper=%.6f  width=%.6f" % (chi_minus, chi_plus, chi_plus - chi_minus))
gbar = np.sqrt(np.max(gref ** 2 / abar))
Lip = gbar * np.sqrt(T / 2)
dsq = np.abs(np.diff(np.sqrt(w))) / np.diff(grid)
print("          Lipschitz sqrt(w): max slope=%.5f  theoretical constant=%.5f  ok=%s" % (dsq.max(), Lip, dsq.max() <= Lip + 1e-6))
inner, outer, solves = certify(p_true, B, Lip, grid, np.sqrt(np.maximum(w, 0.0)))
print("          certified inner set=%s  certified outer set=%s  additional solves=%d" % (np.round(inner, 6).tolist(), np.round(outer, 6).tolist(), solves))
print("          w_rel <= w everywhere:", np.all(wb <= w + 1e-8), "  largest gap w-w_rel=%.6f" % np.max(w - wb))
ends = relaxed_endpoints(p_true, B)
print("          relaxed interval [%.4f, %.4f]" % (ends[0], ends[1]))

ua = np.array([.03, .0, -.02]); ub = np.array([-.02, .01, .03])
Pa = path_masses([b0 + .1 * g + ua, b0 + .1 * g + ua]); Pb = path_masses([b0 + .9 * g + ub, b0 + .9 * g + ub])
p_mix = K @ (.5 * Pa + .5 * Pb)
wm = np.array([w_profile(p_mix, c) for c in grid])
print("MIXTURE   w(chi) at nodes 0,.2,.4,.6,.8,1 =", wm[::8])
print("          negative second differences:", int(np.sum(np.diff(wm, 2) < -1e-9)))

def linear_part(v):
    return transitions(v) - transitions(np.zeros(3))

T_star = transitions(b_star)
D = np.zeros((4, 7))
D[:, 0] = K @ (chain(linear_part(g), T_star) + chain(T_star, linear_part(g)))
for k in range(3):
    e = np.zeros(3); e[k] = 1.0
    D[:, 1 + k] = K @ chain(linear_part(e), T_star)
    D[:, 4 + k] = K @ chain(T_star, linear_part(e))
root_p = np.sqrt(p_true)
tangent = np.eye(4) - np.outer(root_p, root_p)
A = tangent @ (D[:, [0]] / root_p[:, None])
Bfull = tangent @ (D[:, 1:] / root_p[:, None])

def projector(Bn):
    U, s, _ = np.linalg.svd(Bn, full_matrices=False)
    rank = int(np.sum(s > TOL_RANK * s[0]))
    return U[:, :rank], rank, s

def eff_info(Bn):
    if Bn is None:
        return float((A.T @ A)[0, 0])
    Q, _, _ = projector(Bn)
    res = A - Q @ (Q.T @ A)
    return float((res.T @ res)[0, 0])

Btime = Bfull @ np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]]).T
Bstate = Bfull @ np.block([[np.eye(3)], [np.eye(3)]])
print("EFFICIENCY  full information=%.5f  time-only=%.5f  state-only=%.3e  free=%.3e" % (eff_info(None), eff_info(Btime), eff_info(Bstate), eff_info(Bfull)))
print("            singular values of the free block=%s  rank=%d (tangent space of the 4-cell simplex = 3)" % (np.array2string(projector(Bfull)[2], precision=5, suppress_small=False), projector(Bfull)[1]))
print("            rank of the state-only block=%d  rank of the time-only block=%d" % (projector(Bstate)[1], projector(Btime)[1]))

g0 = rng.normal(size=5); g1 = rng.normal(size=5); w0, w1 = .3, .7
h = (w0 * g0 + w1 * g1) / (w0 + w1)
print("INTERVENTION  inf_h=%.6f  formula w0w1/(w0+w1)||g1-g0||^2=%.6f" % (w0 * np.sum((g0 - h) ** 2) + w1 * np.sum((g1 - h) ** 2), w0 * w1 / (w0 + w1) * np.sum((g1 - g0) ** 2)))

print("ELBOW (d=1): slope of log sqrt(MSE) against log eps")
for m in [.1, .25, .5]:
    vals = []
    epss = np.logspace(-1, -4, 7)
    for e in epss:
        J = np.logspace(0, 18, 8000)
        mse = J ** (-4 * m) + 2 * e ** 4 * (2 * J + 1) + 4 * e ** 2
        vals.append(np.sqrt(mse.min()))
    slope = np.polyfit(np.log(epss[-4:]), np.log(vals[-4:]), 1)[0]
    print("   m=%.2f  slope=%.3f  expected=%.3f" % (m, slope, min(1.0, 8 * m / (4 * m + 1))))

cgrid = np.linspace(0, 1, 17)
c95 = chi2.ppf(.95, 3)
true_in = np.array([w_profile(p_true, c) <= B for c in cgrid])
phat_par = cp.Parameter(4)
pearson = []
hoeffding = []
for c in cgrid:
    P = cp.Variable(27); y = K @ P
    pearson.append(cp.Problem(cp.Minimize(cp.sum([cp.quad_over_lin(phat_par[j] - y[j], y[j]) for j in range(4)])), feasible(P) + [cost_cp(P, c) <= B]))
    P2 = cp.Variable(27)
    hoeffding.append(cp.Problem(cp.Minimize(cp.norm_inf(phat_par - K @ P2)), feasible(P2) + [cost_cp(P2, c) <= B]))

def solved(prob):
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        return np.inf, 1
    if prob.value is None:
        return np.inf, 1
    return prob.value, int(prob.status != "optimal")

for N, R in [(300, 1000), (5000, 1000)]:
    hN = np.sqrt(np.log(2 * 4 / .05) / (2 * N))
    cov_pe = cov_ho = 0; wid_pe = wid_ho = 0; inR = 0; flagged = 0
    for rep in range(R):
        counts = rng.multinomial(N, p_true); phat = counts / N
        phat_par.value = phat
        inR += (N * np.sum((phat - p_true) ** 2 / p_true) <= c95)
        inset_pe = np.zeros(len(cgrid), bool); inset_ho = np.zeros(len(cgrid), bool)
        for k in range(len(cgrid)):
            value, flag = solved(pearson[k]); flagged += flag
            inset_pe[k] = N * value <= c95
            value, flag = solved(hoeffding[k]); flagged += flag
            inset_ho[k] = value <= hN
        cov_pe += np.all(inset_pe[true_in]); cov_ho += np.all(inset_ho[true_in])
        wid_pe += inset_pe.sum(); wid_ho += inset_ho.sum()
    print("INFERENCE N=%d R=%d  P(p in R_N)=%.3f  set coverage: Pearson=%.3f Hoeffding=%.3f  mean nodes: Pearson=%.2f Hoeffding=%.2f  true=%d  solves not reported optimal=%d" % (N, R, inR / R, cov_pe / R, cov_ho / R, wid_pe / R, wid_ho / R, true_in.sum(), flagged))

e = 1e-4
dchi = (w_profile(p_true, chi_plus + e) - w_profile(p_true, chi_plus - e)) / (2 * e)
grad = np.zeros(4)
for j in range(4):
    dp = np.zeros(4); dp[j] = e; dp -= e / 4
    grad[j] = (w_profile(p_true + dp, chi_plus) - w_profile(p_true - dp, chi_plus)) / (2 * e)
varg = np.sum(p_true * grad ** 2) - np.sum(p_true * grad) ** 2
binfo = dchi ** 2 / varg
print("BOUNDARY  chi_plus=%.5f  d_chi w=%.5e  Var_p(grad)=%.5e  boundary information=%.5e" % (chi_plus, dchi, varg, binfo))
w_profile(p_true, chi_plus)
mult = w_profile.con[-1].dual_value
print("          multiplier (centered) =", np.round(mult - mult.mean(), 6), " numerical gradient (centered) =", np.round(grad - grad.mean(), 6))

def pop_distance(chi):
    P = cp.Variable(27); y = K @ P
    prob = cp.Problem(cp.Minimize(cp.sum([cp.quad_over_lin(p_true[j] - y[j], y[j]) for j in range(4)])), feasible(P) + [cost_cp(P, chi) <= B])
    prob.solve(solver=cp.CLARABEL)
    return max(prob.value, 0.0)

for eta in [.01, .02, .04, .08]:
    d = pop_distance(chi_plus + eta)
    print("          eta=%.2f  delta=%.4e  info*eta^2=%.4e  ratio=%.3f" % (eta, d, binfo * eta ** 2, d / (binfo * eta ** 2)))
for c in [.7, .8, .9, 1.0]:
    d = pop_distance(c)
    print("          chi=%.2f  delta=%.3e  critical value over delta: projection=%.0f  one-sided direction=%.0f" % (c, d, chi2.ppf(.95, 3) / d, chi2.ppf(.90, 1) / d))
print("          one-sided half-width for chi_plus at N=300 and N=5000:", [round(float(np.sqrt(chi2.ppf(.90, 1) / (binfo * n))), 3) for n in (300, 5000)])
