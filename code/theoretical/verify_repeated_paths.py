import numpy as np
rng = np.random.default_rng(20261030)
FLOOR = 0.25

def cell_ids(x, t, K, d):
    cid = np.full(x.shape[0], min(int(t * K), K - 1))
    cells = (x * K).astype(int).clip(0, K - 1)
    for j in range(d - 1):
        cid = cid * K + cells[:, j]
    return cid

def occupation_moments(K, d, npaths, dt):
    steps = int(round(1 / dt))
    x = rng.random((npaths, d - 1))
    L = np.zeros((npaths, K ** d))
    rows = np.arange(npaths)
    for s in range(steps):
        np.add.at(L, (rows, cell_ids(x, s * dt, K, d)), K ** d * dt)
        x = (x + np.sqrt(dt) * rng.standard_normal((npaths, d - 1))) % 1.0
    return L.mean(), (L ** 2).mean()

def sup_occupation(K, d, npaths, dt):
    steps = int(round(1 / (K * dt)))
    x = np.full((npaths, d - 1), 0.5 / K)
    L = np.zeros(npaths)
    for s in range(steps):
        inside = np.all((x * K).astype(int) == 0, axis=1)
        L += K ** d * dt * inside
        x = (x + np.sqrt(dt) * rng.standard_normal((npaths, d - 1))) % 1.0
    return L.mean()

print("OCCUPATION MOMENTS  (expected: d=2 E L^2 ~ K^0.5, d=3 E L^2 ~ K log K)")
for d in [2, 3]:
    Ks = [4, 8, 16, 32] if d == 2 else [4, 8, 16]
    vals = []
    for K in Ks:
        dt = min(1e-4, 1 / (40 * K * K))
        EL, EL2 = occupation_moments(K, d, 1500, dt)
        sup = sup_occupation(K, d, 3000, dt)
        vals.append((K, EL, EL2, sup))
        print("  d=%d K=%2d  E L=%.3f  E L^2=%.3f  sup_x E_x L=%.3f" % (d, K, EL, EL2, sup))
    lk = np.log([v[0] for v in vals]); l2 = np.log([v[2] for v in vals]); ls = np.log([v[3] for v in vals])
    print("  d=%d slopes: E L^2 %.2f  sup_x E_x L %.2f" % (d, np.polyfit(lk, l2, 1)[0], np.polyfit(lk, ls, 1)[0]))

def paths_d2(N, K, rho, signs, dt, w0amp):
    steps = int(round(1 / dt))
    x = rng.random(N)
    S = np.zeros((N, K * K)); Lc = np.zeros((N, K * K)); Qp = np.zeros(N)
    rows = np.arange(N)
    for s in range(steps):
        t = s * dt
        cid = cell_ids(x[:, None], t, K, 2)
        w = w0amp * np.sin(2 * np.pi * x) + rho * K * signs[cid]
        dx = w * dt + np.sqrt(dt) * rng.standard_normal(N)
        np.add.at(S, (rows, cid), dx)
        np.add.at(Lc, (rows, cid), dt)
        Qp += w * w * dt
        x = (x + dx) % 1.0
    return S, Lc, Qp

def estimate(S, Lc, K):
    N = S.shape[0]; A = np.arange(N // 2); B = np.arange(N // 2, N)
    G = Lc[A].mean(0)
    nB = len(B)
    U = (S[B].sum(0) ** 2 - (S[B] ** 2).sum(0)) / (nB * (nB - 1))
    ok = K * K * G >= FLOOR
    return np.sum(U[ok] / G[ok]), int(np.sum(~ok))

print("ESTIMATOR RATE d=2  (expected slopes: m=0.25 -> -1/3, m=1 -> -1/2; cells kept when normalized occupation >= %.2f)" % FLOOR)
for m_s, cK, reps in [(0.25, 0.35, 8), (1.0, 1.0, 8)]:
    out = []
    for N in [100, 200, 400, 800, 1600]:
        K = max(2, int(round(cK * N ** (2 / (4 * m_s + 2)))))
        rho = 0.5 * K ** (-m_s - 1)
        signs = rng.choice([-1.0, 1.0], size=K * K)
        dt = min(1e-4, 1 / (10 * K * K))
        _, Lref, Qtrue = paths_d2(4000, K, rho, signs, dt, 0.3)
        Q = Qtrue.mean()
        errs = []
        dropped = 0
        for r in range(reps):
            S, Lc, _ = paths_d2(N, K, rho, signs, dt, 0.3)
            value, removed = estimate(S, Lc, K)
            errs.append(value - Q)
            dropped += removed
        rmse = float(np.sqrt(np.mean(np.square(errs))))
        out.append((N, K, Q, rmse))
        print("  m=%.2f N=%4d K=%2d  Q=%.4f  rmse=%.4f  smallest reference occupation=%.3f  dropped cells=%d" % (m_s, N, K, Q, rmse, K * K * Lref.mean(0).min(), dropped))
    slope = np.polyfit(np.log([o[0] for o in out]), np.log([o[3] for o in out]), 1)[0]
    print("  m=%.2f fitted slope %.3f" % (m_s, slope))
