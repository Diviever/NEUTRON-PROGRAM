import sys
import time
import numpy as np

# Universal PyQt5 / PyQt6 compatibility
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox, QSplitter,
        QScrollArea, QFrame, QLineEdit, QComboBox, QStackedWidget, QSizePolicy, QCheckBox
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox, QSplitter,
        QScrollArea, QFrame, QLineEdit, QComboBox, QStackedWidget, QSizePolicy, QCheckBox
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread

# Matplotlib Qt Canvas
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.figure import Figure
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


# ==============================================================================
# 1. Helpers: Locale-safe float conversion, Qt5/Qt6 enum access
# ==============================================================================
def safe_float(val, default=0.0):
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(str(val).strip().replace(',', '.'))
    except (ValueError, TypeError):
        return default


def qenum(owner, group, name):
    """Qt6 uses scoped enums (Qt.Orientation.Horizontal), Qt5 flat ones (Qt.Horizontal)."""
    scoped = getattr(owner, group, None)
    if scoped is not None and hasattr(scoped, name):
        return getattr(scoped, name)
    return getattr(owner, name)


# ==============================================================================
# 2. Thickness profile d(j) for one material across the N repeats
# ==============================================================================
POLY_MAX_DEGREE = 6
THICKNESS_MODES = ("linear", "polynomial")
_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def thickness_profile(spec, n_repeats):
    """
    Thickness of one material for every repeat j = 0 … N-1.
    j = 0 is the substrate side (deepest bilayer), j = N-1 the vacuum side (top).

    spec['mode'] == 'linear'      : d(t) = d_start + (d_end - d_start) * t,   t = j / (N-1)
    spec['mode'] == 'polynomial'  : d(x) = c0 + c1*x + c2*x^2 + ... ,
                                    x = t = j/(N-1)  (spec['arg'] == 't', default)
                                    x = j            (spec['arg'] == 'j')
                                    coefficients in spec['coeffs'] = [c0, c1, c2, ...]
    Returns the raw values (may be negative for a badly chosen polynomial).
    """
    n = int(n_repeats)
    if n <= 0:
        return np.array([], dtype=float)
    j = np.arange(n, dtype=float)
    t = j / (n - 1) if n > 1 else np.zeros(1)

    mode = spec.get('mode', 'linear')
    if mode == 'polynomial':
        coeffs = np.asarray(spec.get('coeffs', [0.0]), dtype=float)
        if coeffs.size == 0:
            return np.zeros(n)
        x = j if spec.get('arg', 't') == 'j' else t
        return np.polynomial.polynomial.polyval(x, coeffs)

    d0 = float(spec.get('d_start', 50.0))
    d1 = float(spec.get('d_end', 50.0))
    return d0 + (d1 - d0) * t


# ==============================================================================
# 3. Vectorized Parratt Calculation Engine
# ==============================================================================
def interlayer_potentials(V_left, V_right, n):
    if n <= 0:
        return []
    if n == 1:
        return [0.5 * (V_left + V_right)]
    return [V_left + (V_right - V_left) * (k + 1) / (n + 1) for k in range(n)]


def build_stack_from_layers(layers, n_repeats, n_sublayers, d_sublayer, V_vac, V_sub):
    if not layers or n_repeats <= 0:
        return np.array([]), np.array([])

    n_mat = len(layers)
    thk_arrays = []
    pot_values = []

    for lay in layers:
        # d(j) from the layer's function; negative values are clipped to 0 Å.
        # Reversed so that the stack is written from the vacuum side (j = N-1) downwards.
        thk_seq = np.clip(thickness_profile(lay, n_repeats), 0.0, None)[::-1]
        thk_arrays.append(thk_seq)
        pot_values.append(lay['pot'] * 1e-3)  # Convert µeV -> meV

    V_vac_meV = V_vac * 1e-3
    V_sub_meV = V_sub * 1e-3

    main_thks = [thk_arrays[m][k] for k in range(n_repeats) for m in range(n_mat)]
    print(main_thks)
    main_pots = [pot_values[m] for _ in range(n_repeats) for m in range(n_mat)]

    potentials, thicknesses = [], []

    # Vacuum -> First Layer
    potentials += interlayer_potentials(V_vac_meV, main_pots[0], n_sublayers)
    thicknesses += [d_sublayer] * n_sublayers

    # Bulk layers + interfaces
    n_total_bulk = len(main_pots)
    for i, (V, d) in enumerate(zip(main_pots, main_thks)):
        potentials.append(V)
        thicknesses.append(d)

        V_next = main_pots[i + 1] if i + 1 < n_total_bulk else V_sub_meV
        potentials += interlayer_potentials(V, V_next, n_sublayers)
        thicknesses += [d_sublayer] * n_sublayers

    return np.array(thicknesses, dtype=float), np.array(potentials, dtype=complex)


def calculate_parratt(thicknesses, potentials, q_vals, V_vac, V_sub, lam_A=5.0):
    """
    Parratt recursion. kz is computed only for the distinct potential values
    (materials + interface grading steps), so memory stays O(unique·Q) even for
    stacks with tens of thousands of (sub)layers.
    """
    if len(thicknesses) == 0:
        return np.zeros_like(q_vals)

    q = np.asarray(q_vals, dtype=float)
    k0 = 2.0 * np.pi / lam_A
    E_meV = 81.8042 / (lam_A**2)
    pot_factor = (k0**2) / E_meV

    V_vac_meV = V_vac * 1e-3
    V_sub_meV = V_sub * 1e-3

    all_pots = np.concatenate(([V_vac_meV], potentials, [V_sub_meV])).astype(complex)
    all_thks = np.concatenate(([0.0], thicknesses, [0.0]))
    total_layers = len(all_pots)

    u_pots, u_idx = np.unique(np.round(all_pots, 12), return_inverse=True)
    kz_inc_sq = (q / 2.0)**2
    kz_u = np.sqrt(kz_inc_sq[None, :] - (pot_factor * u_pots[:, None]) + 0.0j)

    X = np.zeros(len(q), dtype=complex)

    for j in range(total_layers - 2, -1, -1):
        kz_j = kz_u[u_idx[j]]
        kz_next = kz_u[u_idx[j + 1]]
        d_next = all_thks[j + 1]

        r = (kz_j - kz_next) / (kz_j + kz_next)
        phi = np.exp(2.0j * kz_next * d_next)
        X = (r + X * phi) / (1.0 + r * X * phi)

    return np.abs(X)**2


# ==============================================================================
# 4. Supermirror optimizer: gradient descent on the thickness polynomials + N search
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
    import time
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


# ==============================================================================
# 5. Per-material card: name, V, thickness function d(j) and a mini graph
# ==============================================================================
LAYER_COLORS = ['#0052cc', '#c81e1e', '#1a7f37', '#8250df', '#bf8700', '#0a7ea4', '#cf222e']


class MiniCanvas(FigureCanvas):
    """Small matplotlib canvas that lets the mouse wheel scroll the surrounding list."""
    def wheelEvent(self, event):
        event.ignore()


class LayerCard(QFrame):
    changed = pyqtSignal()
    remove_requested = pyqtSignal(object)

    def __init__(self, name="Layer", d_start=50.0, d_end=50.0, pot=0.100,
                 n_repeats=30, color='#0052cc', parent=None):
        super().__init__(parent)
        self.setObjectName("LayerCard")
        self.setFrameShape(qenum(QFrame, 'Shape', 'StyledPanel'))
        self.n_repeats = int(n_repeats)
        self.color = color
        self._building = True
        self._arg_prev = 't'

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        # ---- Header: material, potential, mode, remove ----
        head = QHBoxLayout()
        swatch = QLabel("■")
        swatch.setStyleSheet(f"color: {color}; font-size: 13pt; background: transparent;")
        head.addWidget(swatch)
        head.addWidget(QLabel("Material:"))
        self.ed_name = QLineEdit(str(name))
        self.ed_name.setMaximumWidth(80)
        head.addWidget(self.ed_name)
        head.addWidget(QLabel("V (µeV):"))
        self.ed_pot = QLineEdit(f"{pot:.4f}")
        self.ed_pot.setMaximumWidth(75)
        head.addWidget(self.ed_pot)
        head.addStretch(1)
        head.addWidget(QLabel("d(j):"))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Linear", "Polynomial"])
        self.cmb_mode.setCurrentIndex(0)  # default: linear
        head.addWidget(self.cmb_mode)
        self.btn_remove = QPushButton("✕")
        self.btn_remove.setFixedWidth(28)
        self.btn_remove.setToolTip("Remove this material from the unit cell")
        head.addWidget(self.btn_remove)
        outer.addLayout(head)

        # ---- Body: parameters (left) + mini graph (right) ----
        body = QHBoxLayout()
        body.setSpacing(8)
        self.stack = QStackedWidget()

        # Linear page
        lin_page = QWidget()
        lg = QGridLayout(lin_page)
        lg.setContentsMargins(0, 0, 0, 0)
        lg.addWidget(QLabel("d_start (Å):"), 0, 0)
        self.ed_d_start = QLineEdit(f"{d_start:.2f}")
        self.ed_d_start.setToolTip("Thickness at j = 0 (substrate side)")
        lg.addWidget(self.ed_d_start, 0, 1)
        lg.addWidget(QLabel("d_end (Å):"), 1, 0)
        self.ed_d_end = QLineEdit(f"{d_end:.2f}")
        self.ed_d_end.setToolTip("Thickness at j = N-1 (vacuum side)")
        lg.addWidget(self.ed_d_end, 1, 1)
        lin_formula = QLabel("d(t) = d_start + (d_end − d_start)·t\nt = j / (N−1),  j = 0 … N−1")
        lin_formula.setObjectName("Formula")
        lg.addWidget(lin_formula, 2, 0, 1, 2)
        lg.setRowStretch(3, 1)
        self.stack.addWidget(lin_page)

        # Polynomial page
        poly_page = QWidget()
        pv = QVBoxLayout(poly_page)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(3)
        top = QHBoxLayout()
        top.addWidget(QLabel("Degree:"))
        self.spin_deg = QSpinBox()
        self.spin_deg.setRange(0, POLY_MAX_DEGREE)
        self.spin_deg.setValue(2)
        top.addWidget(self.spin_deg)
        top.addWidget(QLabel("x ="))
        self.cmb_arg = QComboBox()
        self.cmb_arg.addItems(["t = j/(N−1) ∈ [0,1]", "j (repeat index)"])
        self.cmb_arg.setToolTip("Polynomial argument. Switching converts the coefficients "
                                "so the curve stays the same for the current N.")
        top.addWidget(self.cmb_arg)
        top.addStretch(1)
        pv.addLayout(top)

        self.lbl_formula = QLabel()
        self.lbl_formula.setObjectName("Formula")
        self.lbl_formula.setWordWrap(True)
        pv.addWidget(self.lbl_formula)

        cgrid = QGridLayout()
        cgrid.setHorizontalSpacing(4)
        cgrid.setVerticalSpacing(2)
        self.coef_labels, self.coef_edits = [], []
        for k in range(POLY_MAX_DEGREE + 1):
            lbl = QLabel(f"c{k}:")
            ed = QLineEdit("0")
            ed.setMaximumWidth(90)
            ed.setToolTip(f"Coefficient of x{str(k).translate(_SUPERSCRIPT)} (Å)")
            r, c = divmod(k, 2)
            cgrid.addWidget(lbl, r, 2 * c)
            cgrid.addWidget(ed, r, 2 * c + 1)
            self.coef_labels.append(lbl)
            self.coef_edits.append(ed)
        pv.addLayout(cgrid)
        pv.addStretch(1)
        self.stack.addWidget(poly_page)

        body.addWidget(self.stack, 1)

        # Mini graph d(j)
        self.mini_fig = Figure(figsize=(2.5, 1.6), dpi=80, facecolor='#ffffff')
        self.mini_canvas = MiniCanvas(self.mini_fig)
        self.mini_canvas.setFixedSize(240, 150)
        self.mini_ax = self.mini_fig.add_subplot(111)
        body.addWidget(self.mini_canvas, 0)
        outer.addLayout(body)

        self.lbl_warn = QLabel("")
        self.lbl_warn.setStyleSheet("color: #cf222e; font-weight: bold; background: transparent;")
        self.lbl_warn.setVisible(False)
        outer.addWidget(self.lbl_warn)

        # ---- Signals ----
        for ed in [self.ed_name, self.ed_pot, self.ed_d_start, self.ed_d_end] + self.coef_edits:
            ed.textChanged.connect(self._emit_changed)
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.cmb_arg.currentIndexChanged.connect(self._on_arg_changed)
        self.spin_deg.valueChanged.connect(self._on_degree_changed)
        self.btn_remove.clicked.connect(lambda: self.remove_requested.emit(self))

        self._building = False
        self._update_degree_widgets()
        self.redraw()

    # ---------------- Public API ----------------
    def set_n_repeats(self, n):
        self.n_repeats = int(n)
        self.redraw()

    def mode(self):
        return THICKNESS_MODES[self.cmb_mode.currentIndex()]

    def arg(self):
        return 'j' if self.cmb_arg.currentIndex() == 1 else 't'

    def coefficients(self):
        deg = self.spin_deg.value()
        return [safe_float(self.coef_edits[k].text(), 0.0) for k in range(deg + 1)]

    def set_polynomial(self, coeffs, arg='t'):
        """Programmatic setup (e.g. from an optimizer): switch to polynomial mode with given c0..cn."""
        coeffs = list(coeffs)[:POLY_MAX_DEGREE + 1]
        self._building = True
        self.cmb_arg.setCurrentIndex(1 if arg == 'j' else 0)
        self._arg_prev = 'j' if arg == 'j' else 't'
        self.spin_deg.setValue(max(0, len(coeffs) - 1))
        for k, ed in enumerate(self.coef_edits):
            ed.setText(self._fmt(coeffs[k]) if k < len(coeffs) else "0")
        self.cmb_mode.setCurrentIndex(1)
        self.stack.setCurrentIndex(1)
        self._building = False
        self._update_degree_widgets()
        self._emit_changed()

    def get_spec(self):
        return {
            'name': self.ed_name.text().strip(),
            'pot': safe_float(self.ed_pot.text(), 0.0),
            'mode': self.mode(),
            'd_start': safe_float(self.ed_d_start.text(), 50.0),
            'd_end': safe_float(self.ed_d_end.text(), 50.0),
            'coeffs': self.coefficients(),
            'arg': self.arg(),
        }

    # ---------------- Internal ----------------
    @staticmethod
    def _fmt(v):
        return f"{v:.8g}"

    def _all_coeffs(self):
        return [safe_float(ed.text(), 0.0) for ed in self.coef_edits]

    def _set_all_coeffs(self, coeffs):
        for ed, c in zip(self.coef_edits, coeffs):
            ed.setText(self._fmt(c))

    def _emit_changed(self, *_):
        if self._building:
            return
        self.redraw()
        self.changed.emit()

    def _on_mode_changed(self, idx):
        self.stack.setCurrentIndex(idx)
        if idx == 1 and not self._building and all(c == 0.0 for c in self._all_coeffs()):
            # First switch to polynomial: start from the current linear profile
            d0 = safe_float(self.ed_d_start.text(), 50.0)
            d1 = safe_float(self.ed_d_end.text(), 50.0)
            slope = d1 - d0
            if self.arg() == 'j':
                slope /= max(self.n_repeats - 1, 1)
            self._building = True
            self._set_all_coeffs([d0, slope] + [0.0] * (POLY_MAX_DEGREE - 1))
            self._building = False
        self._emit_changed()

    def _on_arg_changed(self, idx):
        new_arg = 'j' if idx == 1 else 't'
        if new_arg != self._arg_prev and not self._building:
            # Keep the curve identical: x_j = s * t with s = N-1
            s = float(max(self.n_repeats - 1, 1))
            coeffs = self._all_coeffs()
            if new_arg == 'j':
                coeffs = [c / s**k for k, c in enumerate(coeffs)]
            else:
                coeffs = [c * s**k for k, c in enumerate(coeffs)]
            self._building = True
            self._set_all_coeffs(coeffs)
            self._building = False
        self._arg_prev = new_arg
        self._update_degree_widgets()
        self._emit_changed()

    def _on_degree_changed(self, _):
        self._update_degree_widgets()
        self._emit_changed()

    def _update_degree_widgets(self):
        deg = self.spin_deg.value()
        for k in range(POLY_MAX_DEGREE + 1):
            visible = k <= deg
            self.coef_labels[k].setVisible(visible)
            self.coef_edits[k].setVisible(visible)
        x = 'j' if self.arg() == 'j' else 't'
        terms = []
        for k in range(deg + 1):
            if k == 0:
                terms.append("c0")
            elif k == 1:
                terms.append(f"c1·{x}")
            else:
                terms.append(f"c{k}·{x}{str(k).translate(_SUPERSCRIPT)}")
        self.lbl_formula.setText("d = " + " + ".join(terms))

    def redraw(self):
        spec = self.get_spec()
        n = max(self.n_repeats, 1)
        d = thickness_profile(spec, n)
        j = np.arange(n)

        ax = self.mini_ax
        ax.clear()
        ax.plot(j, d, '-o' if n <= 150 else '-', color=self.color, lw=1.4, ms=2.5)
        n_neg = int(np.sum(d < 0))
        if n_neg:
            ax.axhline(0.0, color='#cf222e', lw=0.8, ls='--')
            ax.fill_between(j, d, 0.0, where=d < 0, color='#cf222e', alpha=0.25)
            self.lbl_warn.setText(f"⚠ d < 0 in {n_neg} of {n} repeats — clipped to 0 Å in the stack")
            self.lbl_warn.setVisible(True)
        else:
            self.lbl_warn.setVisible(False)

        dmin, dmax = (float(np.min(d)), float(np.max(d))) if d.size else (0.0, 0.0)
        ax.set_title(f"{spec['name'] or '—'}: {dmin:.1f} … {dmax:.1f} Å", fontsize=8, pad=3)
        ax.set_xlabel("repeat j  (0 = substrate side)", fontsize=7, labelpad=1)
        ax.set_ylabel("d (Å)", fontsize=7, labelpad=1)
        ax.tick_params(labelsize=6.5, length=2, pad=1)
        ax.set_xlim(-0.5, n - 0.5)
        ypad = 0.1 * (dmax - dmin) if dmax > dmin else max(abs(dmax) * 0.1, 1.0)
        ax.set_ylim(dmin - ypad, dmax + ypad)
        ax.grid(True, ls='--', alpha=0.4, lw=0.6)
        try:
            self.mini_fig.tight_layout(pad=0.4)
        except Exception:
            pass
        self.mini_canvas.draw_idle()


# ==============================================================================
# 6. Background worker: runs optimize_supermirror() off the GUI thread
# ==============================================================================
class OptimizerWorker(QObject):
    progress = pyqtSignal(object)
    finished = pyqtSignal(object)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            result = optimize_supermirror(self.cfg, callback=self.progress.emit,
                                          should_stop=lambda: self._stop)
        except Exception as exc:          # report instead of killing the thread silently
            result = {'status': 'error', 'error': repr(exc)}
        self.finished.emit(result)


# ==============================================================================
# 7. Main GUI Application (Embedded Matplotlib)
# ==============================================================================
class MultilayerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neutron Multilayer Reflectivity Designer")
        self.resize(1400, 820)
        self.cards = []
        self.m = 2.0                      # supermirror m: target range is 0 … m·q_c
        self._opt_thread = None
        self._opt_worker = None
        self._last_apply = 0.0
        self._last_calc_time = 0.0

        # Debounced recalculation (typing in a field doesn't recompute on every keystroke)
        self._calc_timer = QTimer(self)
        self._calc_timer.setSingleShot(True)
        self._calc_timer.setInterval(150)
        self._calc_timer.timeout.connect(self.update_calculation)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(qenum(Qt, 'Orientation', 'Horizontal'))
        main_layout.addWidget(splitter)

        # ----------------- Left Panel: Controls & Layer cards -----------------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Multilayer Global Settings
        global_box = QGroupBox("Multilayer & Interface Settings")
        self.global_box = global_box
        g_layout = QVBoxLayout(global_box)

        # N Repetitions
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Repetitions (N):"))
        self.spin_repeats = QSpinBox()
        self.spin_repeats.setRange(1, 20000)
        self.spin_repeats.setValue(30)
        self.spin_repeats.valueChanged.connect(self.on_repeats_changed)
        h1.addWidget(self.spin_repeats)
        g_layout.addLayout(h1)

        # Sublayer Slices
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Interlayer Sublayers (n):"))
        self.spin_sublayers = QSpinBox()
        self.spin_sublayers.setRange(0, 50)
        self.spin_sublayers.setValue(5)
        self.spin_sublayers.valueChanged.connect(self.schedule_update)
        h2.addWidget(self.spin_sublayers)
        g_layout.addLayout(h2)

        # Sublayer Step Thickness
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Sublayer Step d (Å):"))
        self.spin_d_sub = QDoubleSpinBox()
        self.spin_d_sub.setRange(0.1, 20.0)
        self.spin_d_sub.setSingleStep(0.5)
        self.spin_d_sub.setValue(1.5)
        self.spin_d_sub.valueChanged.connect(self.schedule_update)
        h3.addWidget(self.spin_d_sub)
        g_layout.addLayout(h3)

        # Potentials
        h4 = QHBoxLayout()
        h4.addWidget(QLabel("V_sub (µeV):"))
        self.spin_vsub = QDoubleSpinBox()
        self.spin_vsub.setRange(-2.0, 5.0)
        self.spin_vsub.setDecimals(3)
        self.spin_vsub.setValue(0.090)
        self.spin_vsub.valueChanged.connect(self.schedule_update)
        h4.addWidget(self.spin_vsub)

        h4.addWidget(QLabel("V_vac (µeV):"))
        self.spin_vvac = QDoubleSpinBox()
        self.spin_vvac.setRange(-2.0, 5.0)
        self.spin_vvac.setDecimals(3)
        self.spin_vvac.setValue(0.0)
        self.spin_vvac.valueChanged.connect(self.schedule_update)
        h4.addWidget(self.spin_vvac)
        g_layout.addLayout(h4)

        left_layout.addWidget(global_box)

        # Supermirror optimizer (gradient descent on the polynomial coefficients + N search)
        self.opt_box = self._build_optimizer_box()
        left_layout.addWidget(self.opt_box)

        # Unit Cell: one card per material, each with its own thickness function
        cells_box = QGroupBox("Periodic Unit Cell Layers — thickness functions d(j)")
        self.cells_box = cells_box
        c_layout = QVBoxLayout(cells_box)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(qenum(Qt, 'ScrollBarPolicy', 'ScrollBarAlwaysOff'))
        self.cards_container = QWidget()
        self.cards_container.setObjectName("CardsContainer")
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(2, 2, 2, 2)
        self.cards_layout.setSpacing(6)
        self.cards_layout.addStretch(1)
        self.scroll.setWidget(self.cards_container)
        c_layout.addWidget(self.scroll)

        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("+ Add Layer")
        self.btn_add.clicked.connect(lambda: self.add_layer_row())
        self.btn_del = QPushButton("- Remove Last Layer")
        self.btn_del.clicked.connect(lambda: self.remove_layer_row())
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_del)
        c_layout.addLayout(btn_layout)

        left_layout.addWidget(cells_box, 1)
        splitter.addWidget(left_panel)

        # ----------------- Right Panel: Embedded Matplotlib -----------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(8, 8), dpi=100, facecolor='#ffffff')
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        splitter.addWidget(right_panel)

        # Subplots: ax1 = Reflectivity, ax2 = Potential Profile
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2)
        self.fig.tight_layout(pad=3.0)

        splitter.setSizes([580, 820])

        # Load Default Materials & trigger first plot
        self.load_default_layers()
        self.on_m_changed(self.spin_m.value())
        QTimer.singleShot(50, self.update_calculation)

    # ----------------- Optimizer panel -----------------
    def _build_optimizer_box(self):
        box = QGroupBox("Supermirror Optimizer — gradient descent on d(j) coefficients")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)

        def dspin(lo, hi, val, step, dec, tip=""):
            w = QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setDecimals(dec)
            w.setSingleStep(step)
            w.setValue(val)
            w.setToolTip(tip)
            return w

        def ispin(lo, hi, val, tip=""):
            w = QSpinBox()
            w.setRange(lo, hi)
            w.setValue(val)
            w.setToolTip(tip)
            return w

        self.spin_m = dspin(1.01, 10.0, self.m, 0.1, 2,
                            "Supermirror m: high reflectivity is required for 0 < q ≤ m·q_c")
        self.spin_m.valueChanged.connect(self.on_m_changed)
        self.spin_target = dspin(0.05, 1.0, 0.92, 0.005, 3,
                                 "Stop when the mean reflectivity over (0, m·q_c] reaches this value")
        self.spin_target.valueChanged.connect(self.schedule_update)
        self.lbl_mref = QLabel("")
        self.lbl_mref.setObjectName("Formula")
        self.lbl_mref.setWordWrap(True)

        self.spin_opt_deg = ispin(1, POLY_MAX_DEGREE, 4, "Polynomial degree used for every material")
        self.spin_iter = ispin(5, 5000, 150, "Maximum gradient-descent iterations for each N")
        self.spin_nstart = ispin(2, 20000, 50, "First number of repeats N tried (filled from m)")
        self.spin_nmax = ispin(2, 20000, 150, "Largest N the search may reach (filled from m)")
        self.spin_growth = dspin(1.01, 3.0, 1.15, 0.05, 2, "N is multiplied by this factor when a stage stalls")
        self.spin_lr = dspin(0.01, 50.0, 2.0, 0.5, 2, "Adam step size in Å (decays as 1/(1+it/100))")
        self.spin_dmin = dspin(0.0, 100.0, 10.0, 1.0, 1, "Layers thinner than this are penalized")
        self.chk_hm = QCheckBox("Try Hayter–Mook q⁴-law start profile at each N")
        self.chk_hm.setChecked(True)

        self.btn_opt_start = QPushButton("▶ Start optimization")
        self.btn_opt_start.clicked.connect(lambda: self.start_optimization())
        self.btn_opt_stop = QPushButton("■ Stop")
        self.btn_opt_stop.setEnabled(False)
        self.btn_opt_stop.clicked.connect(lambda: self.stop_optimization())

        self.lbl_opt_status = QLabel("Idle.")
        self.lbl_opt_status.setWordWrap(True)
        self.lbl_opt_status.setObjectName("OptStatus")

        r = 0
        grid.addWidget(QLabel("m:"), r, 0); grid.addWidget(self.spin_m, r, 1)
        grid.addWidget(QLabel("Target ⟨R⟩:"), r, 2); grid.addWidget(self.spin_target, r, 3)
        r += 1
        grid.addWidget(self.lbl_mref, r, 0, 1, 4)
        r += 1
        grid.addWidget(QLabel("Poly degree:"), r, 0); grid.addWidget(self.spin_opt_deg, r, 1)
        grid.addWidget(QLabel("Max iter / N:"), r, 2); grid.addWidget(self.spin_iter, r, 3)
        r += 1
        grid.addWidget(QLabel("N start:"), r, 0); grid.addWidget(self.spin_nstart, r, 1)
        grid.addWidget(QLabel("N max:"), r, 2); grid.addWidget(self.spin_nmax, r, 3)
        r += 1
        grid.addWidget(QLabel("N growth ×:"), r, 0); grid.addWidget(self.spin_growth, r, 1)
        grid.addWidget(QLabel("Step (Å):"), r, 2); grid.addWidget(self.spin_lr, r, 3)
        r += 1
        grid.addWidget(QLabel("Min d (Å):"), r, 0); grid.addWidget(self.spin_dmin, r, 1)
        grid.addWidget(self.chk_hm, r, 2, 1, 2)
        r += 1
        bl = QHBoxLayout()
        bl.addWidget(self.btn_opt_start)
        bl.addWidget(self.btn_opt_stop)
        grid.addLayout(bl, r, 0, 1, 4)
        r += 1
        grid.addWidget(self.lbl_opt_status, r, 0, 1, 4)
        return box

    def current_q_c(self):
        """q_c of the material with the highest potential (Ni in the default stack)."""
        pots = [c.get_spec()['pot'] for c in self.cards]
        if not pots or max(pots) <= 0:
            return 0.0215, "Ni"
        i = int(np.argmax(pots))
        return critical_q(pots[i]), (self.cards[i].get_spec()['name'] or "top")

    def on_m_changed(self, value):
        self.m = float(value)
        layers, r_lo, r_hi = m_reference(self.m)
        n_mat = max(len(self.cards), 1)
        n_est = layers / n_mat
        self.spin_target.blockSignals(True)
        self.spin_target.setValue(round(r_lo, 3))
        self.spin_target.blockSignals(False)
        self.spin_nstart.setValue(max(2, int(round(0.7 * n_est))))
        self.spin_nmax.setValue(max(3, int(np.ceil(2.0 * n_est))))
        q_c, name = self.current_q_c()
        self.lbl_mref.setText(
            f"Reference for m = {self.m:.2f}: ~{layers:,.0f} layers (≈{n_est:,.0f} repeats), "
            f"R ≈ {100*r_lo:.0f}–{100*r_hi:.0f} %.  Range 0 … m·q_c = {self.m*q_c:.4f} Å⁻¹ "
            f"(q_c {name} = {q_c:.4f} Å⁻¹).")
        self.schedule_update()

    def build_opt_config(self):
        deg = self.spin_opt_deg.value()
        n_now = self.spin_repeats.value()
        coeffs = []
        for card in self.cards:
            d_now = thickness_profile(card.get_spec(), n_now)   # current curve -> t-space fit
            coeffs.append(fit_poly_t(d_now, deg).tolist())
        q_c, _ = self.current_q_c()
        return {
            'pots': [c.get_spec()['pot'] for c in self.cards],
            'coeffs': coeffs,
            'm': self.m,
            'q_c': q_c,
            'target': self.spin_target.value(),
            'degree': deg,
            'N_start': self.spin_nstart.value(),
            'N_max': max(self.spin_nmax.value(), self.spin_nstart.value()),
            'growth': self.spin_growth.value(),
            'max_iter': self.spin_iter.value(),
            'lr': self.spin_lr.value(),
            'd_min': self.spin_dmin.value(),
            'init_hm': self.chk_hm.isChecked(),
            'n_sub': self.spin_sublayers.value(),
            'd_sub': self.spin_d_sub.value(),
            'V_vac': self.spin_vvac.value(),
            'V_sub': self.spin_vsub.value(),
            'q_points': OPT_Q_POINTS,
        }

    def set_controls_enabled(self, enabled):
        for w in (self.global_box, self.cells_box, self.spin_m, self.spin_target, self.spin_opt_deg,
                  self.spin_iter, self.spin_nstart, self.spin_nmax, self.spin_growth, self.spin_lr,
                  self.spin_dmin, self.chk_hm, self.btn_opt_start):
            w.setEnabled(enabled)
        self.btn_opt_stop.setEnabled(not enabled)

    def start_optimization(self):
        if self._opt_thread is not None:
            return
        if not self.cards:
            self.lbl_opt_status.setText("Add at least one material first.")
            return
        cfg = self.build_opt_config()
        self._opt_t0 = time.time()
        self.lbl_opt_status.setText(f"Starting: N = {cfg['N_start']}, target ⟨R⟩ ≥ {cfg['target']:.3f} "
                                    f"over (0, {cfg['m'] * cfg['q_c']:.4f}] Å⁻¹ …")
        self.set_controls_enabled(False)

        self._opt_thread = QThread(self)
        self._opt_worker = OptimizerWorker(cfg)
        self._opt_worker.moveToThread(self._opt_thread)
        self._opt_thread.started.connect(self._opt_worker.run)
        self._opt_worker.progress.connect(self.on_opt_progress)
        self._opt_worker.finished.connect(self.on_opt_finished)
        self._opt_worker.finished.connect(self._opt_thread.quit)
        self._opt_thread.finished.connect(self._on_opt_thread_done)
        self._opt_thread.start()

    def stop_optimization(self):
        if self._opt_worker is not None:
            self._opt_worker.stop()
            self.lbl_opt_status.setText(self.lbl_opt_status.text() + "\nStopping after the current step …")

    def _on_opt_thread_done(self):
        if self._opt_worker is not None:
            self._opt_worker.deleteLater()
        if self._opt_thread is not None:
            self._opt_thread.deleteLater()
        self._opt_worker = None
        self._opt_thread = None

    def apply_opt_state(self, N, coeffs):
        """Push N and the t-space coefficients into the GUI (cards switch to polynomial mode)."""
        self.spin_repeats.blockSignals(True)
        self.spin_repeats.setValue(int(N))
        self.spin_repeats.blockSignals(False)
        for card, c in zip(self.cards, coeffs):
            card.n_repeats = int(N)
            card.set_polynomial(c, arg='t')
        self.schedule_update()

    def on_opt_progress(self, info):
        elapsed = time.time() - getattr(self, '_opt_t0', time.time())
        stage = "stage finished" if info.get('status') == 'stage_done' else f"iter {info['iter']}"
        self.lbl_opt_status.setText(
            f"N = {info['N']} ({info['layers']} layers) · {stage} · "
            f"⟨R⟩ = {info['meanR']:.4f} (best {info['best_meanR']:.4f}) → target {info['target']:.3f} · "
            f"{elapsed:.0f} s")
        # Updating cards + main plot costs a full recalculation: throttle it
        now = time.time()
        if info.get('status') == 'stage_done' or now - self._last_apply > max(2.0, 3.0 * self._last_calc_time):
            self._last_apply = now
            self.apply_opt_state(info['N'], info['coeffs'])

    def on_opt_finished(self, res):
        elapsed = time.time() - getattr(self, '_opt_t0', time.time())
        self.set_controls_enabled(True)
        st = res.get('status')
        if st == 'error':
            self.lbl_opt_status.setText(f"✗ Optimizer error: {res.get('error')}")
            return
        self.apply_opt_state(res['N'], res['coeffs'])
        head = {'success': "✓ Target reached",
                'n_max': "✗ N max reached without meeting the target — best result applied",
                'stopped': "■ Stopped — best result so far applied"}.get(st, st)
        self.lbl_opt_status.setText(
            f"{head}: N = {res['N']} ({res['layers']} layers), ⟨R⟩ = {res['meanR']:.4f} "
            f"(target {res['target']:.3f}), {elapsed:.0f} s.")

    def closeEvent(self, event):
        if self._opt_worker is not None:
            self._opt_worker.stop()
        if self._opt_thread is not None:
            self._opt_thread.quit()
            self._opt_thread.wait(15000)
        super().closeEvent(event)

    # ----------------- Layer card operations -----------------
    def add_layer_row(self, name="Layer", d_start=50.0, d_end=50.0, pot=0.100):
        color = LAYER_COLORS[len(self.cards) % len(LAYER_COLORS)]
        card = LayerCard(name, d_start, d_end, pot,
                         n_repeats=self.spin_repeats.value(), color=color)
        card.changed.connect(self.schedule_update)
        card.remove_requested.connect(self.remove_card)
        # insert before the trailing stretch
        self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)
        self.cards.append(card)
        self.schedule_update()
        return card

    def remove_card(self, card):
        if card in self.cards:
            self.cards.remove(card)
            self.cards_layout.removeWidget(card)
            card.setParent(None)
            card.deleteLater()
            self.schedule_update()

    def remove_layer_row(self):
        if self.cards:
            self.remove_card(self.cards[-1])

    def load_default_layers(self):
        for card in list(self.cards):
            self.remove_card(card)
        # Default mode for every material is linear
        self.add_layer_row("Ni", d_start=40.0, d_end=200.0, pot=0.2410)
        self.add_layer_row("Ti", d_start=40.0, d_end=200.0, pot=-0.0490)

    def get_layers(self):
        return [card.get_spec() for card in self.cards]

    def on_repeats_changed(self, n):
        for card in self.cards:
            card.set_n_repeats(n)
        self.schedule_update()

    def schedule_update(self, *_):
        self._calc_timer.start()

    # ----------------- Live Calculation & Plotting -----------------
    def update_calculation(self):
        t_calc0 = time.time()
        layers = self.get_layers()
        if not layers:
            self.ax1.clear()
            self.ax2.clear()
            self.canvas.draw_idle()
            return

        n_rep = self.spin_repeats.value()
        n_sub = self.spin_sublayers.value()
        d_sub = self.spin_d_sub.value()
        v_vac = self.spin_vvac.value()
        v_sub = self.spin_vsub.value()

        # 1. Build Layer Stack
        thk_stack, pot_stack = build_stack_from_layers(
            layers, n_repeats=n_rep, n_sublayers=n_sub, d_sublayer=d_sub,
            V_vac=v_vac, V_sub=v_sub
        )

        if len(thk_stack) == 0:
            return

        # 2. Potential Profile Coordinates
        z_edges = np.concatenate(([0.0], np.cumsum(thk_stack)))
        total_z = float(z_edges[-1])
        z_stair = np.repeat(z_edges, 2)[1:-1]
        V_stair = np.repeat(pot_stack.real * 1e3, 2)  # meV -> µeV

        pad = 50.0
        z_plot = np.concatenate(([-pad, 0.0], z_stair, [total_z, total_z + pad]))
        V_plot = np.concatenate(([v_vac, v_vac], V_stair, [v_sub, v_sub]))

        # 3. Calculate Reflectivity R(q)
        q_c, qc_name = self.current_q_c()
        q_top = self.m * q_c
        q_max = max(0.09, 1.3 * q_top)
        # first OPT_Q_POINTS points = exactly the optimizer's grid over (0, m·q_c]
        q_in = np.linspace(1e-3, q_top, OPT_Q_POINTS)
        q_out = np.linspace(q_top, q_max, 700)[1:]
        q_range = np.concatenate((q_in, q_out))
        R = calculate_parratt(
            thicknesses=thk_stack,
            potentials=pot_stack,
            q_vals=q_range,
            V_vac=v_vac,
            V_sub=v_sub,
            lam_A=5.0
        )
        R_safe = np.clip(R, 1e-6, 1.2)
        in_range = np.arange(q_range.size) < OPT_Q_POINTS
        mean_R = float(np.mean(R[in_range])) if np.any(in_range) else float('nan')
        edge = (q_range >= 0.95 * q_top) & in_range
        edge_R = float(np.mean(R[edge])) if np.any(edge) else float('nan')
        target = self.spin_target.value()

        # 4. Render Plots with Matplotlib
        self.ax1.clear()
        self.ax2.clear()

        # --- Plot 1: Reflectivity ---
        ok = mean_R >= target
        self.ax1.axvspan(0.0, q_top, color='#1a7f37' if ok else '#bf8700', alpha=0.07,
                         label=f'target range 0 … m·$q_c$ = {q_top:.4f} Å⁻¹')
        self.ax1.semilogy(q_range, R_safe, color='#0052cc', lw=1.6, label='Parratt TMM')
        self.ax1.axvline(q_c, color='#e02424', linestyle='--', lw=1.2, alpha=0.7,
                         label=f'{qc_name} $q_c$ edge ({q_c:.4f} Å⁻¹)')
        self.ax1.axvline(q_top, color='#1a7f37', linestyle=':', lw=1.4, alpha=0.9, label=f'm·$q_c$ (m = {self.m:.2f})')
        self.ax1.hlines(target, 0.0, q_top, colors='#57606a', linestyles='--', lw=1.0, alpha=0.8,
                        label=f'target ⟨R⟩ = {target:.3f}')
        n_layers = n_rep * len(layers)
        self.ax1.set_title(f"Neutron Specular Reflectivity $R(q)$\n"
                           f"N = {n_rep} ({n_layers} layers)  ·  ⟨R⟩(0…m·$q_c$) = {mean_R:.4f} "
                           f"{'✓' if ok else '✗'} (target {target:.3f})  ·  R near m·$q_c$ ≈ {edge_R:.3f}",
                           fontsize=9.5, fontweight='bold', pad=6)
        self.ax1.set_xlabel("Momentum transfer $q$ (Å⁻¹)", fontsize=9)
        self.ax1.set_ylabel("Reflectivity $R$", fontsize=9)
        self.ax1.set_xlim(0.0, q_max)
        self.ax1.set_ylim(1e-5, 1.3)
        self.ax1.grid(True, which="both", linestyle="--", alpha=0.4)
        self.ax1.legend(loc='lower left', fontsize=7.5, framealpha=0.9)

        # --- Plot 2: Potential Profile ---
        self.ax2.plot(z_plot, V_plot, color='#c81e1e', lw=1.8, label='$V(z)$')
        self.ax2.fill_between(z_plot, V_plot, 0.0, color='#c81e1e', alpha=0.15)
        self.ax2.set_title(f"Discretized Potential Profile $V(z)$ ({len(thk_stack)} Sublayers)", fontsize=11, fontweight='bold', pad=8)
        self.ax2.set_xlabel("Depth $z$ (Å)", fontsize=9)
        self.ax2.set_ylabel("Potential $V$ (µeV)", fontsize=9)
        self.ax2.set_xlim(z_plot[0], z_plot[-1])

        ymin, ymax = float(np.min(V_plot)), float(np.max(V_plot))
        ypad = (ymax - ymin) * 0.2 if ymax > ymin else 0.1
        self.ax2.set_ylim(ymin - ypad, ymax + ypad)
        self.ax2.grid(True, linestyle="--", alpha=0.4)

        self.fig.tight_layout(pad=2.2)
        self.canvas.draw_idle()
        self._last_calc_time = time.time() - t_calc0


# ==============================================================================
# 8. Entry Point & Light UI Styling
# ==============================================================================
LIGHT_STYLE = """
QWidget {
    background-color: #f6f8fa;
    color: #24292f;
    font-size: 9pt;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    margin-top: 10px;
    font-weight: bold;
    padding: 12px 8px 8px 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: #0969da;
}
QScrollArea {
    border: none;
    background-color: #ffffff;
}
QWidget#CardsContainer {
    background-color: #ffffff;
}
QFrame#LayerCard {
    background-color: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 6px;
}
QFrame#LayerCard QLabel {
    font-weight: normal;
}
QLabel#OptStatus {
    color: #0969da;
    font-weight: bold;
}
QLabel#Formula {
    color: #57606a;
    font-style: italic;
}
QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 4px;
    padding: 3px 6px;
    font-weight: normal;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #0969da;
}
QPushButton {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 4px;
    padding: 5px 12px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #f3f4f6;
    border-color: #8c959f;
}
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(LIGHT_STYLE)

    window = MultilayerApp()
    window.show()
    sys.exit(app.exec() if hasattr(app, 'exec') else app.exec_())
