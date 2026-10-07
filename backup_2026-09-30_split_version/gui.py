"""
gui.py — user interface (PyQt5/PyQt6 + embedded matplotlib)

  * LayerCard        one material: name, V, d(j) function entries and a mini graph
  * OptimizerWorker  runs optimizer.optimize_supermirror() in a background QThread
  * MultilayerApp    main window: settings, optimizer panel, layer cards, R(q) and V(z) plots
"""
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

from calculation import (POLY_MAX_DEGREE, THICKNESS_MODES, thickness_profile,
                         build_stack_from_layers, calculate_parratt)
from optimizer import OPT_Q_POINTS, critical_q, m_reference, fit_poly_t, optimize_supermirror
from gui_style import LAYER_COLORS

_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


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
# 2. Per-material card: name, V, thickness function d(j) and a mini graph
# ==============================================================================


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
# 3. Background worker: runs optimize_supermirror() off the GUI thread
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
# 4. Main GUI Application (Embedded Matplotlib)
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
