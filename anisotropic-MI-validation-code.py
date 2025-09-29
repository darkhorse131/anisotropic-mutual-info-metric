"""
GOLD MASTER — Reference Implementation for Anisotropic Mutual Information and the Emergent Spatial Metric - SM §S8
=================================================

This is the minimal, *validated* code to accompany §S8. It:
  • builds the covariance of the anisotropic quadratic lattice (Eq. S45),
  • computes *exact* Gaussian MI for this equal‑time model using the stable C=0 specialization,
  • verifies the X≈2 scaling prerequisite,
  • performs *single‑radius* MI tomography with ±ε averaging, DT2 (polarization) and PSD projection (DT1),
  • compares the reconstructed metric against the continuum expectation g = Φ diag(1/kx, 1/ky).

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time
import scipy.linalg
from scipy.linalg import sqrtm, eigh

# ==============================================================================
# 1) Simulation Parameters (kept identical to the validated run)
# ==============================================================================

LATTICE_SIZE = 256   # L (large to suppress wrap-around within the local window)
KX, KY       = 1.0, 4.0
MASS         = 0.01  # m > 0 (ξ = 1/m = 100). Ensures a clean contact window.

CELL_SIZE    = 1     # ℓ   (use ℓ=1 to eliminate cell form-factor contamination)
RHO          = 10    # ρ   (choose within ℓ ≪ ρ ≪ ξ ≪ L)

C_VAC        = 1.0   # overall scale (length units)

# ---- Optional strict single-radius variant (small edit #2) --------------------
# When True, use the *normalized* pair-sum direction u = (e1+e2)/√2 so that |ε|=ρ
# for all probes. On a discrete lattice, ρ·u requires integer components; if not
# available, the code will *automatically* fall back to the default pair-sum.
STRICT_SINGLE_RADIUS = False


# ==============================================================================
# 2) Core Physics: Covariance on the L×L torus (Eq. S45)
# ==============================================================================

def build_covariance_matrix(L, kx, ky, m):
    """
    Ground-state equal-time covariances via FFT.
    We keep ω(0,0)=m > 0; no manual zeroing.

    Returns:
        cov_qq, cov_pp  (real-space two-point functions)
    """
    print(f"Building {L}x{L} covariance matrix...")
    t0 = time.time()

    # Momentum grid (matrix convention; 'ij' keeps axes explicit)
    k = np.fft.fftfreq(L) * 2 * np.pi
    kx_grid, ky_grid = np.meshgrid(k, k, indexing='ij')

    # Dispersion (discrete Laplacian)
    omega_sq = m**2 + 2*kx*(1 - np.cos(kx_grid)) + 2*ky*(1 - np.cos(ky_grid))
    omega_sq[0, 0] = m**2
    omega_k = np.sqrt(omega_sq)

    # Momentum-space correlators
    qk = 1.0 / (2.0 * omega_k)     # <q q>
    pk = 0.5 * omega_k             # <p p>

    # Back to real space (numpy IFFT is normalized by 1/L^2)
    cov_qq = np.fft.ifft2(qk).real
    cov_pp = np.fft.ifft2(pk).real

    print(f"Covariance matrix built in {time.time()-t0:.2f} seconds.")
    return cov_qq, cov_pp


# ==============================================================================
# 3) Exact MI for C=0 blocks — stable symplectic spectrum
# ==============================================================================

def get_sub_covariance_blocks(coords, full_cov_qq, full_cov_pp, L):
    """
    Extract A (qq) and B (pp) blocks on the chosen sites.

    *** Indexing consistency note (small edit #1) ***
    This code indexes the covariance tables as `cov[dx, dy]`, i.e., the first index
    corresponds to the *x-displacement* and the second to the *y-displacement*.
    NumPy arrays are "row, col" by default; we are explicitly choosing and using
    the (dx, dy) convention *consistently everywhere in this file*. If a reader
    prefers the (dy, dx) convention, they can swap the indices here.
    """
    n = len(coords)
    A = np.zeros((n, n))  # qq
    B = np.zeros((n, n))  # pp
    for i, (x1, y1) in enumerate(coords):
        for j, (x2, y2) in enumerate(coords):
            dx = (x1 - x2) % L
            dy = (y1 - y2) % L
            A[i, j] = full_cov_qq[dx, dy]
            B[i, j] = full_cov_pp[dx, dy]
    return A, B


def calculate_entropy_stable_C0(A, B):
    """
    Von Neumann entropy for a Gaussian state with block-diagonal Γ = diag(A, B).
    Uses ν_k^2 = eig( A B ) evaluated via the Hermitian surrogate M = √A · B · √A.

    Numerically stable: eigenvalues from `eigh(M)` (real symmetric).
    """
    if A.shape[0] == 0:
        return 0.0

    # Form the symmetric surrogate
    A_sqrt = sqrtm(A)
    M = A_sqrt @ B @ A_sqrt
    M = (M + M.T.conj()) / 2.0  # enforce Hermiticity

    eig_AB = eigh(M, eigvals_only=True)
    eig_AB = eig_AB[eig_AB > 1e-16]          # clip tiny negatives from roundoff
    if eig_AB.size == 0:
        return 0.0

    # Physical floor (uncertainty principle)
    eig_AB[eig_AB < 0.25] = 0.25
    nus = np.sqrt(eig_AB)

    S = 0.0
    for nu in nus:
        # f(ν) = (ν+1/2)ln(ν+1/2) − (ν−1/2)ln(ν−1/2), with safe branch at ν≈1/2
        S_plus = (nu + 0.5) * np.log(nu + 0.5)
        S_minus = 0.0 if nu <= 0.5 + 1e-12 else (nu - 0.5) * np.log(nu - 0.5)
        S += S_plus - S_minus
    return S


def get_mutual_information(cell_a, cell_b, cov_qq, cov_pp, L):
    """I(A:B) = S(A) + S(B) − S(A∪B) using the stable C=0 method."""
    A_a, B_a = get_sub_covariance_blocks(cell_a, cov_qq, cov_pp, L)
    A_b, B_b = get_sub_covariance_blocks(cell_b, cov_qq, cov_pp, L)
    A_ab, B_ab = get_sub_covariance_blocks(cell_a + cell_b, cov_qq, cov_pp, L)

    Sa  = calculate_entropy_stable_C0(A_a,  B_a)
    Sb  = calculate_entropy_stable_C0(A_b,  B_b)
    Sab = calculate_entropy_stable_C0(A_ab, B_ab)

    mi = Sa + Sb - Sab
    return max(0.0, mi)  # floor tiny negatives from roundoff


# ==============================================================================
# 4) Scaling analysis (prerequisite: X ≈ 2 on the axis)
# ==============================================================================

def run_scaling_analysis(cell_size, cov_qq, cov_pp, L):
    print("\n--- Running Scaling Analysis (Prerequisite Check) ---")

    r_min = cell_size * 2
    r_max = min(L // 4, int(1 / (MASS * 2)) if MASS > 0 else L // 4)
    if r_max <= r_min:
        print("ERROR: Insufficient scale separation for scaling analysis.")
        return False

    print(f"Analyzing MI scaling from r={r_min} to r={r_max}...")

    # Central cell (ℓ × ℓ)
    c = L // 2
    cell_a = []
    o = cell_size // 2
    for i in range(cell_size):
        for j in range(cell_size):
            cell_a.append(((c - o + i) % L, (c - o + j) % L))

    r_vals, mi_vals = [], []
    step = max(1, (r_max - r_min) // 20)
    for r in range(r_min, r_max + 1, step):
        vec = np.array([r, 0], dtype=int)
        cell_b = [((x + vec[0]) % L, (y + vec[1]) % L) for (x, y) in cell_a]
        mi = get_mutual_information(cell_a, cell_b, cov_qq, cov_pp, L)
        if mi > 0:
            r_vals.append(r); mi_vals.append(mi)

    if not mi_vals:
        print("ERROR: Could not calculate MI for scaling analysis.")
        return False

    log_r = np.log(r_vals)
    log_mi = np.log(mi_vals)
    slope, intercept = np.polyfit(log_r, log_mi, 1)
    X = -slope
    print(f"Measured scaling exponent X = {X:.4f} (Target: 2.0)")

    # Plot
    plt.figure(figsize=(7.0, 5.6))
    plt.loglog(r_vals, mi_vals, 'bo', label='Numerical MI Data')
    fit_mi = np.exp(intercept) * (np.asarray(r_vals, float)**slope)
    plt.loglog(r_vals, fit_mi, 'r--', label=f'Fit: I ~ r^({slope:.2f}) (X={X:.2f})')
    tgt = fit_mi[0] * (np.asarray(r_vals)/float(r_vals[0]))**(-2.0)
    plt.loglog(r_vals, tgt, 'k:', label='Target: X=2.0')
    plt.title(f'Mutual Information Scaling Analysis (l={CELL_SIZE}, m={MASS})')
    plt.xlabel('Separation r'); plt.ylabel('Mutual Information I(r)')
    plt.grid(True, which='both', ls='--', alpha=0.5); plt.legend()
    plt.show()

    ok = (1.75 < X < 2.25)
    print("-> SUCCESS: Scaling is consistent with the Riemannian regime (X approx 2)."
          if ok else f"-> FAILURE: Scaling (X={X:.2f}) is NOT consistent with X=2.")
    return ok


# ==============================================================================
# 5) Tomography (±ε averaging; DT2; PSD projection)
# ==============================================================================

def project_to_psd(g):
    """Nearest-PSD (Frobenius) via eigenvalue clipping."""
    vals, vecs = eigh(g)
    vals = np.maximum(vals, 1e-16)
    return vecs @ np.diag(vals) @ vecs.T


def run_mi_tomography(rho, cell_size, cov_qq, cov_pp, L):
    print(f"\n--- Running Robust MI Tomography (rho={rho}, l={cell_size}) ---")

    # Central cell A
    c = L // 2
    cell_a = []
    o = cell_size // 2
    for i in range(cell_size):
        for j in range(cell_size):
            cell_a.append(((c - o + i) % L, (c - o + j) % L))

    # Directions for the minimal closed-form
    e1 = np.array([rho, 0], dtype=int)
    e2 = np.array([0, rho], dtype=int)
    # Pair-sum (default): ε = ρ(e1+e2) (length √2 ρ)
    e1pe2_default = np.array([rho, rho], dtype=int)

    # Optional strict single-radius (small edit #2):
    # Try to use u = (e1+e2)/√2 so that |ε|=ρ. On the lattice we need integer
    # components for ρ/√2; if unavailable for this ρ, we fall back to default.
    use_strict = False
    if STRICT_SINGLE_RADIUS:
        n = int(round(rho / np.sqrt(2)))
        if n*n*2 == rho*rho:
            e1pe2_strict = np.array([n, n], dtype=int)
            use_strict = True
        else:
            print("Note: strict single-radius (ρ/√2 integer) not available for this ρ;"
                  " falling back to standard pair-sum (length √2 ρ).")
            e1pe2_strict = None

    # Helper to compute ȳ(v) := 0.5(1/I(+v) + 1/I(-v))
    def ybar(vec):
        cell_b_pos = [((x + vec[0]) % L, (y + vec[1]) % L) for (x, y) in cell_a]
        cell_b_neg = [((x - vec[0]) % L, (y - vec[1]) % L) for (x, y) in cell_a]
        I_pos = get_mutual_information(cell_a, cell_b_pos, cov_qq, cov_pp, L)
        I_neg = get_mutual_information(cell_a, cell_b_neg, cov_qq, cov_pp, L)
        if I_pos <= 0 or I_neg <= 0:
            raise RuntimeError("Non-positive MI encountered in tomography.")
        return 0.5 * (1.0/I_pos + 1.0/I_neg)

    # Measure ȳ along the required directions
    y_e1 = ybar(e1)
    y_e2 = ybar(e2)
    y_e1pe2 = ybar(e1pe2_strict if use_strict else e1pe2_default)

    print(f"  -> ȳ(e1)     = {y_e1:.6f}")
    print(f"  -> ȳ(e2)     = {y_e2:.6f}")
    print(f"  -> ȳ(e1+e2)  = {y_e1pe2:.6f} "
          f"({'strict |ε|=ρ' if use_strict else 'length √2·ρ'})")

    # DT2 (polarization residual)
    pol_res = y_e1pe2 - (y_e1 + y_e2) if not use_strict else \
              (y_e1pe2 - 0.5*y_e1 - 0.5*y_e2)
    avg_y   = (y_e1 + y_e2 + y_e1pe2) / 3.0
    norm_res = abs(pol_res) / max(avg_y, 1e-18)
    print(f"\nDIAGNOSTIC (DT2): Normalized Polarization Residual = {100*norm_res:.2f}%")

    # Closed-form reconstruction (symmetrized)
    rho2 = float(rho**2)
    if use_strict:
        # Strict single-radius identity (§S5.1)
        g11 = (C_VAC/rho2) * y_e1
        g22 = (C_VAC/rho2) * y_e2
        g12 = (C_VAC/rho2) * (y_e1pe2 - 0.5*y_e1 - 0.5*y_e2)
    else:
        # Default pair-sum (validated path)
        g11 = (C_VAC/rho2) * y_e1
        g22 = (C_VAC/rho2) * y_e2
        g12 = (C_VAC/(2.0*rho2)) * (y_e1pe2 - y_e1 - y_e2)

    g_hat = np.array([[g11, g12], [g12, g22]], dtype=float)
    g_psd = project_to_psd(g_hat)

    # Scale (gauge) inference from volume:  det g = Φ^2 det h,  det h = 1/(kx·ky)
    det_g = float(np.linalg.det(g_psd))
    det_h = 1.0 / (KX * KY)
    if det_g > 0:
        c_tilde_eff = C_VAC / np.sqrt(det_g / det_h)
        print(f"Effective C_tilde inferred from volume element: {c_tilde_eff:.4f}")
    else:
        c_tilde_eff = 1.0
        print("Warning: det(g_psd) <= 0. Cannot infer C_tilde.")

    return g_psd, c_tilde_eff, norm_res


# ==============================================================================
# 6) Expected metric (continuum target) and plotting
# ==============================================================================

def expected_metric(kx, ky, c_tilde):
    phi = C_VAC / c_tilde
    return phi * np.array([[1.0/kx, 0.0], [0.0, 1.0/ky]], float)


def plot_metric_comparison(g_hat, g_expected, title):
    fig, ax = plt.subplots(figsize=(7.4, 7.4))

    def plot_ellipse(g, label, style, color):
        vals, vecs = eigh(g)
        if np.any(vals <= 0):
            print(f"Warning: Cannot plot non-PSD metric for {label}."); return
        idx = np.argsort(vals); vals = vals[idx]; vecs = vecs[:, idx]
        width  = 2.0 / np.sqrt(vals[0])
        height = 2.0 / np.sqrt(vals[1])
        angle  = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        ax.add_patch(Ellipse((0,0), width, height, angle=angle,
                             fill=False, lw=3, ls=style, ec=color, label=label))

    plot_ellipse(g_expected, 'Expected (Continuum)', '-',  'C0')
    plot_ellipse(g_hat,      'Reconstructed',       '--', 'C3')

    try:
        max_dim = 1.2 / np.sqrt(np.min(eigh(g_expected)[0]))
    except Exception:
        max_dim = 5.0

    ax.set_xlim(-max_dim, max_dim); ax.set_ylim(-max_dim, max_dim)
    ax.set_aspect('equal', 'box'); ax.grid(True, ls='--', alpha=0.6)
    ax.set_title(title); ax.set_xlabel('x direction'); ax.set_ylabel('y direction')
    ax.legend(); plt.show()


# ==============================================================================
# 7) Main
# ==============================================================================

if __name__ == "__main__":
    print("======================================================================")
    print(" Rigorous Simulation for Emergent Anisotropic Metric (SM S8)")
    print(f" Parameters: L={LATTICE_SIZE}, kx={KX}, ky={KY}, m={MASS:.4f} (xi={1/MASS if MASS>0 else 'inf'})")
    print(f" Tomography: l={CELL_SIZE}, rho={RHO}, strict_single_radius={STRICT_SINGLE_RADIUS}")
    print("======================================================================\n")

    cov_qq, cov_pp = build_covariance_matrix(LATTICE_SIZE, KX, KY, MASS)

    # 1) Scaling pre-check (X ≈ 2 along an axis)
    ok = run_scaling_analysis(CELL_SIZE, cov_qq, cov_pp, LATTICE_SIZE)

    if ok:
        # 2) Tomography at a certified radius
        g_psd, c_tilde_eff, dt2 = run_mi_tomography(RHO, CELL_SIZE, cov_qq, cov_pp, LATTICE_SIZE)

        # 3) Expected target (same gauge-invariant shape; scale fixed from det g)
        g_exp = expected_metric(KX, KY, c_tilde_eff)

        print("\n--- Verification ---")
        np.set_printoptions(precision=6, suppress=True)
        print("Reconstructed Metric (g_psd):\n", g_psd)
        print("\nExpected Theoretical Metric (g_expected):\n", g_exp)

        # 4) Shape & orientation checks (gauge-invariant)
        vals_hat, vecs_hat = eigh(g_psd)
        vals_exp, _        = eigh(g_exp)
        angle_deg = np.degrees(np.arctan2(vecs_hat[1, 0], vecs_hat[0, 0]))
        misalignment = min(abs(angle_deg), abs(angle_deg-90), abs(angle_deg+90),
                           abs(angle_deg-180), abs(angle_deg+180))
        ratio_hat = float(np.max(vals_hat) / np.min(vals_hat))
        ratio_exp = float(np.max(vals_exp) / np.min(vals_exp))
        shape_err = 100.0 * abs(ratio_hat - ratio_exp) / max(ratio_exp, 1e-18)

        ORIENT_TOL = 5.0   # degrees
        SHAPE_TOL  = 10.0  # percent

        print("\n--- Analysis of Results ---")
        print(f"1. Orientation: misalignment = {misalignment:.2f}° (tol {ORIENT_TOL}°) -> "
              f"{'SUCCESS' if misalignment < ORIENT_TOL else 'FAILURE'}")
        print(f"2. Shape (anisotropy): recon = {ratio_hat:.4f}, expected = {ratio_exp:.4f}, "
              f"error = {shape_err:.2f}% (tol {SHAPE_TOL}%) -> "
              f"{'SUCCESS' if shape_err < SHAPE_TOL else 'FAILURE'}")
        print(f"   DT2 residual (normalized): {100*dt2:.2f}%")

        if (misalignment < ORIENT_TOL) and (shape_err < SHAPE_TOL):
            print("\n=======================================================================")
            print("=   OVERALL CONCLUSION: SUCCESS                                       =")
            print("=   The simulation validates the S8 model in the certified local window. =")
            print("=======================================================================")
        else:
            print("\n=======================================================================")
            print("=   OVERALL CONCLUSION: FAILURE                                       =")
            print("=   The reconstructed metric does not match within tolerance.          =")
            print("=   Revisit scaling/ρ or disable strict mode if enabled.               =")
            print("=======================================================================")

        plot_metric_comparison(g_psd, g_exp,
                               title=f"Metric Reconstruction Comparison\n"
                                     f"(kx={KX}, ky={KY}, rho={RHO}, l={CELL_SIZE}, m={MASS})")
    else:
        print("\nSimulation halted due to invalid scaling regime.")