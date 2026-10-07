import sys
import numpy as np

# Universal PyQt5 / PyQt6 compatibility
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox, QSplitter,
        QScrollArea, QFrame, QLineEdit, QComboBox, QStackedWidget, QSizePolicy
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox, QSplitter,
        QScrollArea, QFrame, QLineEdit, QComboBox, QStackedWidget, QSizePolicy
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal

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
    if len(thicknesses) == 0:
        return np.zeros_like(q_vals)

    q = np.asarray(q_vals, dtype=float)
    k0 = 2.0 * np.pi / lam_A
    E_meV = 81.8042 / (lam_A**2)
    pot_factor = (k0**2) / E_meV

    V_vac_meV = V_vac * 1e-3
    V_sub_meV = V_sub * 1e-3

    all_pots = np.concatenate(([V_vac_meV], potentials, [V_sub_meV]))
    all_thks = np.concatenate(([0.0], thicknesses, [0.0]))
    total_layers = len(all_pots)

    kz_inc_sq = (q / 2.0)**2
    kz = np.sqrt(kz_inc_sq[None, :] - (pot_factor * all_pots[:, None]) + 0.0j)

    X = np.zeros(len(q), dtype=complex)

    for j in range(total_layers - 2, -1, -1):
        kz_j = kz[j]
        kz_next = kz[j + 1]
        d_next = all_thks[j + 1]

        r = (kz_j - kz_next) / (kz_j + kz_next)
        phi = np.exp(2.0j * kz_next * d_next)
        X = (r + X * phi) / (1.0 + r * X * phi)

    return np.abs(X)**2


# ==============================================================================
# 4. Per-material card: name, V, thickness function d(j) and a mini graph
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
        return f"{v:.6g}"

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
        ax.plot(j, d, '-o', color=self.color, lw=1.4, ms=2.5)
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
# 5. Main GUI Application (Embedded Matplotlib)
# ==============================================================================
class MultilayerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neutron Multilayer Reflectivity Designer")
        self.resize(1400, 820)
        self.cards = []

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
        g_layout = QVBoxLayout(global_box)

        # N Repetitions
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Repetitions (N):"))
        self.spin_repeats = QSpinBox()
        self.spin_repeats.setRange(1, 500)
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

        # Unit Cell: one card per material, each with its own thickness function
        cells_box = QGroupBox("Periodic Unit Cell Layers — thickness functions d(j)")
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
        QTimer.singleShot(50, self.update_calculation)

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
        q_range = np.linspace(0.001, 0.09, 1000)
        R = calculate_parratt(
            thicknesses=thk_stack,
            potentials=pot_stack,
            q_vals=q_range,
            V_vac=v_vac,
            V_sub=v_sub,
            lam_A=5.0
        )
        R_safe = np.clip(R, 1e-6, 1.2)

        # 4. Render Plots with Matplotlib
        self.ax1.clear()
        self.ax2.clear()

        # --- Plot 1: Reflectivity ---
        self.ax1.semilogy(q_range, R_safe, color='#0052cc', lw=2.0, label='Parratt TMM')
        self.ax1.axvline(0.0215, color='#e02424', linestyle='--', lw=1.2, alpha=0.7, label='Ni $q_c$ Edge (0.0215 Å⁻¹)')
        self.ax1.set_title("Neutron Specular Reflectivity $R(q)$", fontsize=11, fontweight='bold', pad=8)
        self.ax1.set_xlabel("Momentum transfer $q$ (Å⁻¹)", fontsize=9)
        self.ax1.set_ylabel("Reflectivity $R$", fontsize=9)
        self.ax1.set_xlim(0.0, 0.09)
        self.ax1.set_ylim(1e-5, 1.3)
        self.ax1.grid(True, which="both", linestyle="--", alpha=0.4)
        self.ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)

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


# ==============================================================================
# 6. Entry Point & Light UI Styling
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
