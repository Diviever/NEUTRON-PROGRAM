"""
optimizer.py — supermirror optimizer (no GUI code)

  * m_reference(), critical_q()   reference layers / reflectivity for a given m, q_c of a material
  * StackTemplate                  fast batched Parratt for many thickness sets at once
  * hayter_mook_profiles()         q^4-law start profile
  * optimize_supermirror()         gradient descent (Adam) on the polynomial coefficients + N search
"""
import time

import numpy as np

from calculation import interlayer_potentials


# ==============================================================================
# 1. Supermirror optimizer: gradient descent on the thickness polynomials + N search
# ==============================================================================
POT_FACTOR = 4.0 * np.pi**2 / 81.8042   # kz^2 [Å^-2] per meV of potential (λ-independent)
OPT_Q_POINTS = 600                      # q grid over (0, m·q_c] used for ⟨R⟩

# Reference performance of real supermirrors: m -> (total layers, R range)
M_REFERENCE = [
    # m,   layers,  R_low, R_high
    (1.0,      1,   0.98,  1.00),   # anchor: plain Ni mirror (total reflection)
    (2.0,    150,   0.92,  0.95),
    (3.0,    400,   0.88,  0.91),
    (4.0,   1000,   0.80,  0.85),
    (6.0,   4000,   0.65,  0.72),
    (7.0,   8000,   0.55,  0.60),
]


def critical_q(V_ueV):
    """q_c (Å^-1) of a material with potential V (µeV)."""
    V_meV = max(float(V_ueV), 0.0) * 1e-3
    return 2.0 * np.sqrt(POT_FACTOR * V_meV)


def m_reference(m):
    """Interpolated (layers, R_low, R_high) for a given m (log-log for layers, linear for R)."""
    ms = np.array([r[0] for r in M_REFERENCE])
    lay = np.array([r[1] for r in M_REFERENCE], dtype=float)
    lo = np.array([r[2] for r in M_REFERENCE])
    hi = np.array([r[3] for r in M_REFERENCE])
    m = float(m)
    lm, lms = np.log(m), np.log(ms)
    if m > ms[-1]:   # extrapolate with the last segment's slope
        s_l = (np.log(lay[-1]) - np.log(lay[-2])) / (lms[-1] - lms[-2])
        layers = lay[-1] * np.exp(s_l * (lm - lms[-1]))
        s_lo = (lo[-1] - lo[-2]) / (ms[-1] - ms[-2])
        s_hi = (hi[-1] - hi[-2]) / (ms[-1] - ms[-2])
        r_lo = max(lo[-1] + s_lo * (m - ms[-1]), 0.2)
        r_hi = max(hi[-1] + s_hi * (m - ms[-1]), 0.25)
    else:
        layers = np.exp(np.interp(lm, lms, np.log(lay)))
        r_lo = float(np.interp(m, ms, lo))
        r_hi = float(np.interp(m, ms, hi))
    return float(layers), float(r_lo), float(r_hi)


class StackTemplate:
    """
    Fixed layer structure (N repeats, materials, interface sublayers) for fast
    batched reflectivity: many thickness sets are evaluated in one Parratt pass.
    Only bulk thicknesses vary; sublayer thicknesses and all potentials are fixed.
    """
    def __init__(self, pots_ueV, n_repeats, n_sublayers, d_sublayer, V_vac, V_sub, q):
        self.n_mat = len(pots_ueV)
        self.N = int(n_repeats)
        self.q = np.asarray(q, dtype=float)
        n_sub = int(n_sublayers)

        pots_meV = [p * 1e-3 for p in pots_ueV]
        V_vac_meV, V_sub_meV = V_vac * 1e-3, V_sub * 1e-3
        main_pots = [pots_meV[m] for _ in range(self.N) for m in range(self.n_mat)]

        # layer list (vacuum ... substrate), bulk column index or -1 for sublayers
        pots = [V_vac_meV]
        cols = [-2]                       # -2: semi-infinite media (thickness 0)
        pots += interlayer_potentials(V_vac_meV, main_pots[0], n_sub)
        cols += [-1] * n_sub
        for i, V in enumerate(main_pots):
            pots.append(V)
            cols.append(i)
            V_next = main_pots[i + 1] if i + 1 < len(main_pots) else V_sub_meV
            pots += interlayer_potentials(V, V_next, n_sub)
            cols += [-1] * n_sub
        pots.append(V_sub_meV)
        cols.append(-2)

        pots = np.round(np.array(pots, dtype=float), 12)
        self.u_pots, self.u_idx = np.unique(pots, return_inverse=True)
        self.cols = np.array(cols, dtype=int)
        kz2 = (self.q / 2.0)**2
        self.kz_u = np.sqrt(kz2[None, :] - POT_FACTOR * self.u_pots[:, None] + 0.0j)   # (U, Q)
        self.phi_sub_u = np.exp(2.0j * self.kz_u * float(d_sublayer))                  # (U, Q)
        self._r_cache = {}

    def _r(self, a, b):
        key = (a, b)
        r = self._r_cache.get(key)
        if r is None:
            ka, kb = self.kz_u[a], self.kz_u[b]
            r = (ka - kb) / (ka + kb)
            self._r_cache[key] = r
        return r

    def bulk_from_profiles(self, D):
        """D: (P, n_mat, N) thickness per material and repeat j (j=0 substrate side)
           -> (P, N*n_mat) bulk thickness in stack order (top first)."""
        D = np.clip(D, 0.0, None)
        P = D.shape[0]
        return D[:, :, ::-1].transpose(0, 2, 1).reshape(P, self.N * self.n_mat)

    def reflectivity(self, bulk):
        """bulk: (P, n_bulk) -> R: (P, Q)"""
        bulk = np.atleast_2d(bulk)
        P = bulk.shape[0]
        X = np.zeros((P, self.q.size), dtype=complex)
        u, cols = self.u_idx, self.cols
        for j in range(len(u) - 2, -1, -1):
            r = self._r(u[j], u[j + 1])
            c = cols[j + 1]
            if c == -2:                       # substrate: no phase
                X = (r + X) / (1.0 + r * X)
                continue
            if c == -1:
                phi = self.phi_sub_u[u[j + 1]]
            else:
                phi = np.exp(2.0j * self.kz_u[u[j + 1]][None, :] * bulk[:, c][:, None])
            Xp = X * phi
            X = (r + Xp) / (1.0 + r * Xp)
        return np.abs(X)**2


def poly_vander_t(N, degree):
    n = int(N)
    t = np.arange(n, dtype=float) / (n - 1) if n > 1 else np.zeros(1)
    return np.vander(t, degree + 1, increasing=True)          # (N, deg+1)


def fit_poly_t(d, degree):
    """Least-squares polynomial (in t = j/(N-1)) through the thickness samples d(j)."""
    n = len(d)
    t = np.arange(n, dtype=float) / (n - 1) if n > 1 else np.zeros(1)
    deg_eff = min(degree, max(n - 1, 0))
    c = np.polynomial.polynomial.polyfit(t, d, deg_eff) if n > 1 else np.array([d[0]])
    return np.concatenate([c, np.zeros(degree + 1 - len(c))])


def hayter_mook_profiles(pots_ueV, N, m, q_c):
    """
    Initial guess: bilayer k (k=0 top/vacuum side) reflects at q_k with q_k^4 spaced linearly
    between q_c and m*q_c (Hayter–Mook-type density ∝ q^3). Bragg condition with refraction:
        Λ_k = 2π / sqrt(q_k^2 - q_c,avg^2),  thickness split equally between the materials.
    Returns d(j) array (n_mat, N) with j = 0 at the substrate side.
    """
    n_mat = len(pots_ueV)
    V_avg = float(np.mean(pots_ueV)) * 1e-3
    qc_avg2 = 4.0 * POT_FACTOR * max(V_avg, 0.0)
    k = np.arange(N, dtype=float)
    frac = k / (N - 1) if N > 1 else np.zeros(1)
    q_top, q_bot = q_c, m * q_c
    qk = (q_top**4 + frac * (q_bot**4 - q_top**4)) ** 0.25
    lam = 2.0 * np.pi / np.sqrt(np.maximum(qk**2 - qc_avg2, 1e-10))
    d_k = lam / n_mat                    # k = 0 is the top
    d_j = d_k[::-1]                      # j = 0 is the substrate side
    return np.tile(d_j, (n_mat, 1))


def optimize_supermirror(cfg, callback=None, should_stop=None):
    """
    Gradient descent (Adam, forward-difference gradient evaluated in one batched
    Parratt pass) on the t-space polynomial coefficients of every material,
    wrapped in an outer search over the number of repeats N.

    cfg keys: pots (µeV list), coeffs (list of t-space coeff lists), m, q_c, target,
              degree, N_start, N_max, growth, max_iter, lr, d_min, init_hm (bool),
              n_sub, d_sub, V_vac, V_sub, q_points
    Stops when mean R over (0, m*q_c] >= target, or N_max is exceeded, or should_stop().
    """
    callback = callback or (lambda info: None)
    should_stop = should_stop or (lambda: False)

    pots = list(cfg['pots'])
    n_mat = len(pots)
    deg = int(cfg['degree'])
    n_c = deg + 1
    m, q_c, target = float(cfg['m']), float(cfg['q_c']), float(cfg['target'])
    q = np.linspace(1e-3, m * q_c, int(cfg.get('q_points', OPT_Q_POINTS)))
    d_min = float(cfg.get('d_min', 10.0))
    h = 0.02                                   # finite-difference step (Å)
    stall_window, stall_tol = 25, 2e-5
    last_emit = [0.0]

    # starting coefficients (t-space, fixed degree)
    theta_user = np.zeros((n_mat, n_c))
    for i, c in enumerate(cfg['coeffs']):
        c = np.asarray(c, dtype=float)[:n_c]
        theta_user[i, :len(c)] = c

    def loss_and_R(tmpl, V, thetas):
        """thetas: (P, n_mat, n_c) -> loss (P,), meanR (P,), R (P,Q)"""
        D = np.einsum('jc,pmc->pmj', V, thetas)                   # (P, n_mat, N)
        R = tmpl.reflectivity(tmpl.bulk_from_profiles(D))
        meanR = R.mean(axis=1)
        pen = np.mean(np.clip(d_min - D, 0.0, None)**2, axis=(1, 2)) / d_min**2
        return (1.0 - meanR) + 10.0 * pen, meanR, R

    def emit(info, force=False):
        now = time.time()
        if force or now - last_emit[0] > 1.0:
            last_emit[0] = now
            callback(info)

    N = max(int(cfg['N_start']), deg + 1, 2)
    theta = theta_user.copy()
    best = None                                   # overall best (N, theta, meanR)
    status = 'running'

    while True:
        tmpl = StackTemplate(pots, N, cfg['n_sub'], cfg['d_sub'], cfg['V_vac'], cfg['V_sub'], q)
        V = poly_vander_t(N, deg)

        # candidates for this N: previous coefficients (t-space is N independent) and a fresh HM guess
        cands = [theta]
        if cfg.get('init_hm', True):
            D_hm = hayter_mook_profiles(pots, N, m, q_c)
            cands.append(np.array([fit_poly_t(D_hm[i], deg) for i in range(n_mat)]))
        L0, mR0, _ = loss_and_R(tmpl, V, np.array(cands))
        i0 = int(np.argmin(L0))
        theta = cands[i0].copy()

        # Adam
        mom = np.zeros_like(theta)
        vel = np.zeros_like(theta)
        b1, b2, eps = 0.9, 0.999, 1e-12
        lr0 = float(cfg['lr'])
        stage_best = (float(L0[i0]), theta.copy(), float(mR0[i0]))
        history = []
        n_par = theta.size
        eye = np.eye(n_par).reshape(n_par, n_mat, n_c) * h

        for it in range(1, int(cfg['max_iter']) + 1):
            if should_stop():
                status = 'stopped'
                break
            batch = np.concatenate([theta[None], theta[None] + eye], axis=0)
            L, mR, R = loss_and_R(tmpl, V, batch)
            if L[0] < stage_best[0]:
                stage_best = (L[0], theta.copy(), mR[0])
            history.append(L[0])

            info = {'N': N, 'layers': N * n_mat, 'iter': it, 'meanR': float(mR[0]),
                    'best_meanR': float(stage_best[2]), 'coeffs': stage_best[1].tolist(),
                    'target': target, 'status': 'running'}
            if mR[0] >= target:
                status = 'success'
                break
            emit(info)

            g = ((L[1:] - L[0]) / h).reshape(theta.shape)
            mom = b1 * mom + (1 - b1) * g
            vel = b2 * vel + (1 - b2) * g * g
            mhat = mom / (1 - b1**it)
            vhat = vel / (1 - b2**it)
            lr = lr0 / (1.0 + it / 100.0)
            theta = theta - lr * mhat / (np.sqrt(vhat) + eps)

            if len(history) > stall_window and \
               history[-stall_window - 1] - min(history[-stall_window:]) < stall_tol:
                break

        theta = stage_best[1] if status != 'success' else theta
        stage_meanR = float(mR[0]) if status == 'success' else float(stage_best[2])
        if best is None or stage_meanR > best[2] or status == 'success':
            best = (N, theta.copy(), stage_meanR)

        emit({'N': N, 'layers': N * n_mat, 'iter': it, 'meanR': stage_meanR,
              'best_meanR': stage_meanR, 'coeffs': theta.tolist(), 'target': target,
              'status': 'stage_done'}, force=True)

        if status in ('success', 'stopped'):
            break
        N_next = max(N + 1, int(np.ceil(N * float(cfg['growth']))))
        if N_next > int(cfg['N_max']):
            status = 'n_max'
            break
        N = N_next

    if status == 'success':
        N_res, th_res, mR_res = N, theta, stage_meanR
    else:
        N_res, th_res, mR_res = best
    return {'status': status, 'N': N_res, 'layers': N_res * n_mat,
            'coeffs': np.asarray(th_res).tolist(), 'meanR': mR_res, 'target': target}
