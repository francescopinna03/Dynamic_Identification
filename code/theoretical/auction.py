import numpy as np
from typing import Tuple


def softmax(logits, lambd):
    z = lambd * logits
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def signal_kernel(V: int, sigma_s: float) -> np.ndarray:
    S = V
    p = np.full((V, S), sigma_s / (S - 1))
    for v in range(V):
        p[v, v] = 1.0 - sigma_s
    return p


def payoff_tensor(V: int, B: int) -> np.ndarray:
    u = np.zeros((B, B, V))
    for i in range(B):
        for j in range(B):
            for v in range(V):
                if i > j:
                    u[i, j, v] = v - i
                elif i == j:
                    u[i, j, v] = 0.5 * (v - i)
    return u


def expected_payoffs(q: np.ndarray, chi: float,
                     p_v: np.ndarray, p_s_given_v: np.ndarray,
                     u_tensor: np.ndarray) -> np.ndarray:
    V = p_v.shape[0]
    S = V

    p_s = (p_v[:, None] * p_s_given_v).sum(axis=0)

    p_v_si = p_v[None, :] * p_s_given_v.T
    p_v_given_si = p_v_si / p_s[:, None]

    p_bj_given_v_correct = p_s_given_v @ q
    U_correct_at_v = np.einsum('vj,ijv->iv', p_bj_given_v_correct, u_tensor)
    U_correct = p_v_given_si @ U_correct_at_v.T

    qbar_given_si = p_v_given_si @ p_bj_given_v_correct
    U_cursed = np.einsum(
        'sv,sj,ijv->si', p_v_given_si, qbar_given_si, u_tensor
    )

    return (1 - chi) * U_correct + chi * U_cursed


def cva_qre_equilibrium(lambd: float, chi: float,
                        V: int = 3, B: int = 3,
                        sigma_s: float = 0.3,
                        damping: float = 0.5,
                        max_iter: int = 2000,
                        tol: float = 1e-10) -> Tuple[np.ndarray, int, float]:
    p_v = np.ones(V) / V
    p_s_given_v = signal_kernel(V, sigma_s)
    u_tensor = payoff_tensor(V, B)

    q = np.ones((V, B)) / B

    diff = np.inf
    for it in range(max_iter):
        U = expected_payoffs(q, chi, p_v, p_s_given_v, u_tensor)
        q_new = np.zeros_like(q)
        for s in range(V):
            q_new[s] = softmax(U[s], lambd)
        q_new = damping * q + (1.0 - damping) * q_new
        diff = float(np.abs(q_new - q).max())
        q = q_new
        if diff < tol:
            return q, it + 1, diff

    return q, max_iter, diff


def cva_fisher(lambd: float, chi: float, tau: float = 0.0,
               V: int = 3, B: int = 3, sigma_s: float = 0.3,
               eps: float = 1e-5) -> Tuple[np.ndarray, dict]:
    params = np.array([lambd, chi, tau], dtype=float)
    p_v = np.ones(V) / V
    p_s_given_v = signal_kernel(V, sigma_s)
    p_s = (p_v[:, None] * p_s_given_v).sum(axis=0)

    def q_at(p):
        q, _, _ = cva_qre_equilibrium(p[0], p[1], V=V, B=B, sigma_s=sigma_s)
        return q

    q0 = q_at(params)
    n = 3
    dq = np.zeros((n, V, B))
    for i in range(n):
        if i == 2:
            dq[i] = 0.0
            continue
        plus = params.copy(); plus[i] += eps
        minus = params.copy(); minus[i] -= eps
        dq[i] = (q_at(plus) - q_at(minus)) / (2 * eps)

    I = np.zeros((n, n))
    for a in range(n):
        for b_idx in range(n):
            val = 0.0
            for s in range(V):
                for b in range(B):
                    if q0[s, b] > 1e-12:
                        val += (p_s[s] * dq[a, s, b] * dq[b_idx, s, b]
                                / q0[s, b])
            I[a, b_idx] = val

    info = dict(q=q0, p_s=p_s, dq=dq)
    return I, info


def perfect_signal_collapse(lambd: float = 1.8,
                            chi_grid=(0.0, 0.2, 0.5, 1.0),
                            V: int = 3, B: int = 3,
                            atol: float = 1e-9) -> dict:
    qs = []
    for chi in chi_grid:
        q, _, diff = cva_qre_equilibrium(
            lambd, float(chi), V=V, B=B, sigma_s=0.0
        )
        if diff > 1e-8:
            raise RuntimeError(f"QRE did not converge at chi={chi}: diff={diff}")
        qs.append(q)
    max_ccp_gap = max(float(np.max(np.abs(q - qs[0]))) for q in qs[1:])
    I, _ = cva_fisher(lambd, 0.2, V=V, B=B, sigma_s=0.0)
    chi_row_norm = float(np.linalg.norm(I[1, :]))
    return {
        'max_ccp_gap': max_ccp_gap,
        'chi_fisher_row_norm': chi_row_norm,
        'passed': bool(max_ccp_gap <= atol and chi_row_norm <= 1e-7),
    }


def diagnose_equilibrium(lambd: float, chi: float,
                         V: int = 3, B: int = 3,
                         sigma_s: float = 0.3):
    q, n_iter, diff = cva_qre_equilibrium(lambd, chi, V, B, sigma_s)
    print(f"  Iterations to convergence: {n_iter}, "
          f"maximum residual: {diff:.2e}")
    print(f"  CCP receiver-side q(b | s):")
    print(f"    s \\ b:  ", "  ".join(f"b={b}" for b in range(B)))
    for s in range(V):
        row = "  ".join(f"{q[s, b]:.4f}" for b in range(B))
        print(f"    s={s}:    {row}")


def main():
    print("=" * 78)
    print("G_4: discretized first-price common-value auction")
    print("Setup: V = S = B = 3, sigma_s = 0.3, eta_0 = (1.80, 0.2)")
    print("=" * 78)

    V, B = 3, 3
    sigma_s = 0.3
    lambd, chi = 1.80, 0.2

    print(f"\nKernel p(s | v) with sigma_s = {sigma_s}:")
    p_s_given_v = signal_kernel(V, sigma_s)
    for v in range(V):
        print(f"  v={v}: " + "  ".join(f"{p_s_given_v[v, s]:.3f}"
                                       for s in range(V)))

    print(f"\nQRE equilibrium diagnostics at (lambda, chi) = ({lambd}, {chi}):")
    diagnose_equilibrium(lambd, chi, V, B, sigma_s)

    print(f"\nQRE equilibrium diagnostics at chi = 0 (no cursing):")
    diagnose_equilibrium(lambd, 0.0, V, B, sigma_s)

    print(f"\nQRE equilibrium diagnostics at chi = 1 (fully cursed):")
    diagnose_equilibrium(lambd, 1.0, V, B, sigma_s)

    print(f"\n" + "-" * 78)
    print("Receiver-side Fisher matrix for G_4")
    print("-" * 78)
    I_G4, info = cva_fisher(lambd, chi, tau=0.0, V=V, B=B, sigma_s=sigma_s)
    print(f"\nFisher G_4 (3x3, expected zero tau row and column):")
    for row in I_G4:
        print("  " + "  ".join(f"{x:+.4f}" for x in row))
    print(f"\nEigenvalues: {np.linalg.eigvalsh(I_G4)}")
    print(f"Rank (eigenvalues > 1e-10): "
          f"{np.sum(np.linalg.eigvalsh(I_G4) > 1e-10)}")

    I_lc = I_G4[:2, :2]
    print(f"\nG_4 submatrix for (lambda, chi):")
    print(f"  {I_lc[0, 0]:+.4f}  {I_lc[0, 1]:+.4f}")
    print(f"  {I_lc[1, 0]:+.4f}  {I_lc[1, 1]:+.4f}")
    print(f"  Eigenvalues: {np.linalg.eigvalsh(I_lc)}")
    print(f"  Determinant: {np.linalg.det(I_lc):.6e}")

    collapse = perfect_signal_collapse(lambd=lambd, V=V, B=B)
    print("\nPerfect-signal limit test:")
    print(f"  maximum CCP gap across chi values: {collapse['max_ccp_gap']:.3e}")
    print(f"  chi Fisher row norm:                {collapse['chi_fisher_row_norm']:.3e}")
    print(f"  PASS: {collapse['passed']}")
    if not collapse['passed']:
        raise AssertionError("cursing does not collapse under a perfect signal")


if __name__ == "__main__":
    main()
