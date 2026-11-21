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

def idx(i, j, l):
    return 9 * i + 3 * j + l

Mr0 = np.zeros((3, 27)); Mq0 = np.zeros((3, 27)); Mr1 = np.zeros((3, 27)); Mq1 = np.zeros((3, 27))
for i in range(3):
    for j in range(3):
        for l in range(3):
            k = idx(i, j, l)
            Mr0[i, k] = 1; Mq0[i, k] = (x[j] - x[i]) / Delta
            Mr1[j, k] = 1; Mq1[j, k] = (x[l] - x[j]) / Delta

pl1 = np.array([.8, .5, .2])
pl2 = np.array([.85, .5, .15])
K = np.zeros((4, 27))
for i in range(3):
    for j in range(3):
        for l in range(3):
            for c1 in range(2):
                for c2 in range(2):
                    K[2 * c1 + c2, idx(i, j, l)] = (pl1[j] if c1 == 0 else 1 - pl1[j]) * (pl2[l] if c2 == 0 else 1 - pl2[l])

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

def path_masses(b_steps):
    P = np.zeros(27)
    T0, T1 = transitions(b_steps[0]), transitions(b_steps[1])
    for i in range(3):
        for j in range(3):
            for l in range(3):
                P[idx(i, j, l)] = pi0[i] * T0[i, j] * T1[j, l]
    return P

def cost_np(P, chi):
    r0, q0, r1, q1 = Mr0 @ P, Mq0 @ P, Mr1 @ P, Mq1 @ P
    c = 0.0
    for i in range(3):
        v0 = q0[i] - r0[i] * (b0[i] + chi * g[i]); v1 = q1[i] - r1[i] * (b0[i] + chi * g[i])
        c += Delta * v0 ** 2 / (2 * r0[i] * abar[i]) + Delta * v1 ** 2 / (2 * r1[i] * abar[i])
    return c

def cost_cp(P, chi):
    r0, q0, r1, q1 = Mr0 @ P, Mq0 @ P, Mr1 @ P, Mq1 @ P
    terms = []
    for i in range(3):
        v0 = q0[i] - r0[i] * (b0[i] + chi * g[i]); v1 = q1[i] - r1[i] * (b0[i] + chi * g[i])
        terms.append(Delta * cp.quad_over_lin(v0, r0[i]) / (2 * abar[i]))
        terms.append(Delta * cp.quad_over_lin(v1, r1[i]) / (2 * abar[i]))
    return cp.sum(terms)

def feasible(P):
    return [P >= 0, cp.sum(P) == 1, Mr0 @ P == pi0]

def w_profile(p, chi):
    P = cp.Variable(27)
    prob = cp.Problem(cp.Minimize(cost_cp(P, chi)), feasible(P) + [K @ P == p])
    prob.solve(solver=cp.CLARABEL)
    return prob.value

def w_relaxed(p, chi):
    P = cp.Variable(27); t0 = cp.Variable(3); t1 = cp.Variable(3)
    r0, q0, r1, q1 = Mr0 @ P, Mq0 @ P, Mr1 @ P, Mq1 @ P
    terms = []
    for i in range(3):
        v0 = q0[i] - r0[i] * b0[i] - t0[i] * g[i]; v1 = q1[i] - r1[i] * b0[i] - t1[i] * g[i]
        terms.append(Delta * cp.quad_over_lin(v0, r0[i]) / (2 * abar[i]))
        terms.append(Delta * cp.quad_over_lin(v1, r1[i]) / (2 * abar[i]))
    cons = feasible(P) + [K @ P == p, t0 >= 0, t1 >= 0, t0 <= r0, t1 <= r1, Delta * (cp.sum(t0) + cp.sum(t1)) / T == chi]
    prob = cp.Problem(cp.Minimize(cp.sum(terms)), cons)
    prob.solve(solver=cp.CLARABEL)
    return prob.value

def relaxed_endpoints(p, B):
    out = []
    for sense in [cp.Minimize, cp.Maximize]:
        P = cp.Variable(27); t0 = cp.Variable(3); t1 = cp.Variable(3)
        r0, q0, r1, q1 = Mr0 @ P, Mq0 @ P, Mr1 @ P, Mq1 @ P
        terms = []
        for i in range(3):
            v0 = q0[i] - r0[i] * b0[i] - t0[i] * g[i]; v1 = q1[i] - r1[i] * b0[i] - t1[i] * g[i]
            terms.append(Delta * cp.quad_over_lin(v0, r0[i]) / (2 * abar[i]))
            terms.append(Delta * cp.quad_over_lin(v1, r1[i]) / (2 * abar[i]))
        cons = feasible(P) + [K @ P == p, t0 >= 0, t1 >= 0, t0 <= r0, t1 <= r1, cp.sum(terms) <= B]
        prob = cp.Problem(sense(Delta * (cp.sum(t0) + cp.sum(t1)) / T), cons)
        prob.solve(solver=cp.CLARABEL)
        out.append(prob.value)
    return out

chi_star = .4
u_star = np.array([.02, -.03, .01])
b_star = b0 + chi_star * g + u_star
P_star = path_masses([b_star, b_star])
p_true = K @ P_star
C_star = cost_np(P_star, chi_star)
grid = np.linspace(0, 1, 41)
w = np.array([w_profile(p_true, c) for c in grid])
B = 2.0 * w.min()
print("INSTANCE  true cost=%.6f  profile minimum=%.6f  budget=%.6f  p=%s" % (C_star, w.min(), B, p_true))
wb = np.array([w_relaxed(p_true, c) for c in grid])
inset = grid[w <= B]
print("PROFILE   w(chi) at nodes 0,.2,.4,.6,.8,1 =", w[::8])
print("          negative second differences:", int(np.sum(np.diff(w, 2) < -1e-9)), "out of", len(w) - 2)
print("          exact identified set: [%.4f, %.4f]  nodes=%d  connected=%s" % (inset.min(), inset.max(), len(inset), np.all(np.diff(inset) < 0.026)))
print("          w_rel <= w everywhere:", np.all(wb <= w + 1e-8), "  largest gap w-w_rel=%.6f" % np.max(w - wb))
ends = relaxed_endpoints(p_true, B)
print("          relaxed interval [%.4f, %.4f]" % (ends[0], ends[1]))
gbar = np.sqrt(np.max(g ** 2 / abar))
Lip = gbar * np.sqrt(T / 2)
dsq = np.abs(np.diff(np.sqrt(w))) / np.diff(grid)
print("          Lipschitz sqrt(w): max slope=%.5f  theoretical constant=%.5f  ok=%s" % (dsq.max(), Lip, dsq.max() <= Lip + 1e-6))

ua = np.array([.03, .0, -.02]); ub = np.array([-.02, .01, .03])
Pa = path_masses([b0 + .1 * g + ua, b0 + .1 * g + ua]); Pb = path_masses([b0 + .9 * g + ub, b0 + .9 * g + ub])
p_mix = K @ (.5 * Pa + .5 * Pb)
wm = np.array([w_profile(p_mix, c) for c in grid])
print("MIXTURE   w(chi) at nodes 0,.2,.4,.6,.8,1 =", wm[::8])
print("          negative second differences:", int(np.sum(np.diff(wm, 2) < -1e-9)))

def pmap(theta):
    chi = theta[0]; u0 = theta[1:4]; u1 = theta[4:7]
    return K @ path_masses([b0 + chi * g + u0, b0 + chi * g + u1])

theta0 = np.concatenate([[chi_star], u_star, u_star])
p0 = pmap(theta0)
eps = 1e-6
D = np.zeros((4, 7))
for k in range(7):
    e = np.zeros(7); e[k] = eps
    D[:, k] = (pmap(theta0 + e) - pmap(theta0 - e)) / (2 * eps)
Wh = np.diag(1 / np.sqrt(p0))
A = Wh @ D[:, [0]]
Bfull = Wh @ D[:, 1:]
def eff_info(Bn):
    if Bn is None or Bn.shape[1] == 0:
        return float((A.T @ A)[0, 0])
    Q, _ = np.linalg.qr(Bn)
    res = A - Q @ (Q.T @ A)
    return float((res.T @ res)[0, 0])
Btime = Bfull @ np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]]).T
Bstate = Bfull @ np.block([[np.eye(3)], [np.eye(3)]])
print("EFFICIENCY  full information=%.5f  time-only=%.5f  state-only=%.5f  free=%.3e" % (eff_info(None), eff_info(Btime), eff_info(Bstate), eff_info(Bfull)))
print("            rank of free B=%d (tangent space of the 4-cell simplex = 3)" % np.linalg.matrix_rank(Bfull))

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
for N, R in [(300, 40), (5000, 20)]:
  hN = np.sqrt(np.log(2 * 4 / .05) / (2 * N))
  cov_pe = cov_ho = 0; wid_pe = wid_ho = 0; inR = 0
  for rep in range(R):
      counts = rng.multinomial(N, p_true); phat = counts / N
      inR += (N * np.sum((phat - p_true) ** 2 / p_true) <= c95)
      inset_pe = np.zeros(len(cgrid), bool); inset_ho = np.zeros(len(cgrid), bool)
      for k, c in enumerate(cgrid):
          P = cp.Variable(27); y = K @ P
          obj = cp.sum([cp.quad_over_lin(phat[j] - y[j], y[j]) for j in range(4)])
          prob = cp.Problem(cp.Minimize(obj), feasible(P) + [cost_cp(P, c) <= B])
          prob.solve(solver=cp.CLARABEL)
          inset_pe[k] = prob.status == "optimal" and N * prob.value <= c95
          P2 = cp.Variable(27)
          prob2 = cp.Problem(cp.Minimize(0), feasible(P2) + [cost_cp(P2, c) <= B, cp.abs(phat - K @ P2) <= hN])
          prob2.solve(solver=cp.CLARABEL)
          inset_ho[k] = prob2.status == "optimal"
      cov_pe += np.all(inset_pe[true_in]); cov_ho += np.all(inset_ho[true_in])
      wid_pe += inset_pe.sum(); wid_ho += inset_ho.sum()
  print("INFERENCE N=%d R=%d  P(p in R_N)=%.3f  set coverage: Pearson=%.3f Hoeffding=%.3f  mean nodes: Pearson=%.2f Hoeffding=%.2f  true=%d" % (N, R, inR / R, cov_pe / R, cov_ho / R, wid_pe / R, wid_ho / R, true_in.sum()))

lo, hi = .7, .8
for _ in range(40):
    mid = (lo + hi) / 2
    if w_profile(p_true, mid) <= B: lo = mid
    else: hi = mid
chi_plus = (lo + hi) / 2
e = 1e-4
dchi = (w_profile(p_true, chi_plus + e) - w_profile(p_true, chi_plus - e)) / (2 * e)
grad = np.zeros(4)
for j in range(4):
    dp = np.zeros(4); dp[j] = e; dp -= e / 4
    grad[j] = (w_profile(p_true + dp, chi_plus) - w_profile(p_true - dp, chi_plus)) / (2 * e)
varg = np.sum(p_true * grad ** 2) - np.sum(p_true * grad) ** 2
binfo = dchi ** 2 / varg
print("BOUNDARY  chi_plus=%.5f  d_chi w=%.5e  Var_p(grad)=%.5e  boundary information=%.5e" % (chi_plus, dchi, varg, binfo))
P = cp.Variable(27); con = feasible(P) + [K @ P == p_true]
prob = cp.Problem(cp.Minimize(cost_cp(P, chi_plus)), con); prob.solve(solver=cp.CLARABEL)
mult = con[-1].dual_value
print("          multiplier (centered) =", np.round(mult - mult.mean(), 6), " numerical gradient (centered) =", np.round(grad - grad.mean(), 6))

def pop_distance(chi):
    P = cp.Variable(27); y = K @ P
    prob = cp.Problem(cp.Minimize(cp.sum([cp.quad_over_lin(p_true[j] - y[j], y[j]) for j in range(4)])), feasible(P) + [cost_cp(P, chi) <= B])
    prob.solve(solver=cp.CLARABEL)
    return max(prob.value, 0.0)

for eta in [.01, .02, .04, .08]:
    d = pop_distance(chi_plus + eta)
    print("          eta=%.2f  delta=%.4e  info*eta^2=%.4e  ratio=%.3f" % (eta, d, binfo * eta ** 2, d / (binfo * eta ** 2)))
for c in [.8, .9, 1.0]:
    d = pop_distance(c)
    print("          chi=%.2f  delta=%.3e  N to exclude: projection=%.0f  one-sided direction=%.0f" % (c, d, chi2.ppf(.95, 3) / d, chi2.ppf(.90, 1) / d))
print("          one-sided half-width for chi_plus at N=300 and N=5000:", [round(float(np.sqrt(chi2.ppf(.90, 1) / (binfo * n))), 3) for n in (300, 5000)])
