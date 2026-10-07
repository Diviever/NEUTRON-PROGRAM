"""
gui.py — user interface (PyQt5/PyQt6 + embedded matplotlib)

    LayerCard       the entries of ONE Material object; typing changes the material directly
    MultilayerApp   main window: holds self.materials (list of Material), shows R(q) and V(z),
                    runs the optimizer loop in the program itself

Data flow:
    entries  ->  Material objects  ->  build_stack()  ->  reflectivity()  ->  plots
    optimizer changes the same Material objects  ->  show_material() puts the numbers back in the entries
"""
import time

import numpy as np

# Universal PyQt5 / PyQt6 compatibility
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox, QSplitter,
        QScrollArea, QFrame, QLineEdit, QComboBox, QStackedWidget
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox, QSplitter,
        QScrollArea, QFrame, QLineEdit, QComboBox, QStackedWidget
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.figure import Figure
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from calculation import (Material, build_stack, reflectivity, critical_q, q_grid,
                         mean_reflectivity)
from optimizer import reference_for_m, hayter_mook_start, GradientDescent, optimize
from gui_style import LAYER_COLORS

MAX_DEGREE = 6
SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


# ==============================================================================
# Helpers
# ==============================================================================
def safe_float(text, default):
    """Number from an entry ('1,5' and '1.5' both work); `default` if it is not a number."""
    try:
        return float(str(text).strip().replace(',', '.'))
    except ValueError:
        return default


def qenum(owner, group, name):
    """Qt6 uses scoped enums (Qt.Orientation.Horizontal), Qt5 flat ones (Qt.Horizontal)."""
    scoped = getattr(owner, group, None)
    if scoped is not None and hasattr(scoped, name):
        return getattr(scoped, name)
    return getattr(owner, name)


class MiniCanvas(FigureCanvas):
    """Small matplotlib canvas that lets the mouse wheel scroll the list of materials."""
    def wheelEvent(self, event):
        event.ignore()


# ==============================================================================
# One material card
# ==============================================================================
class LayerCard(QFrame):
    """
    Entries of one Material object.
        read_entries()   entries  -> material   (called whenever a number is typed)
        show_material()  material -> entries    (called after the program changed the material)
    """
    changed = pyqtSignal()                 # the material was edited -> recalculate
    remove_requested = pyqtSignal(object)

    def __init__(self, material, n_repeats, color):
        super().__init__()
        self.setObjectName("LayerCard")
        self.setFrameShape(qenum(QFrame, 'Shape', 'StyledPanel'))
        self.material = material
        self.n_repeats = n_repeats
        self.color = color
        self._showing = False              # True while show_material() fills the entries

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        # ---- header: name, potential, mode, remove ----
        head = QHBoxLayout()
        swatch = QLabel("■")
        swatch.setStyleSheet(f"color: {color}; font-size: 13pt; background: transparent;")
        head.addWidget(swatch)
        head.addWidget(QLabel("Material:"))
        self.ed_name = QLineEdit()
        self.ed_name.setMaximumWidth(80)
        head.addWidget(self.ed_name)
        head.addWidget(QLabel("V (µeV):"))
        self.ed_pot = QLineEdit()
        self.ed_pot.setMaximumWidth(75)
        head.addWidget(self.ed_pot)
        head.addStretch(1)
        head.addWidget(QLabel("d(t):"))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Linear", "Polynomial"])
        head.addWidget(self.cmb_mode)
        self.btn_remove = QPushButton("✕")
        self.btn_remove.setFixedWidth(28)
        self.btn_remove.setToolTip("Remove this material")
        head.addWidget(self.btn_remove)
        outer.addLayout(head)

        # ---- body: entries of the two modes (left) + graph (right) ----
        body = QHBoxLayout()
        self.pages = QStackedWidget()

        linear_page = QWidget()
        lg = QGridLayout(linear_page)
        lg.setContentsMargins(0, 0, 0, 0)
        self.ed_start = QLineEdit()
        self.ed_end = QLineEdit()
        lg.addWidget(QLabel("d_start (Å):"), 0, 0)
        lg.addWidget(self.ed_start, 0, 1)
        lg.addWidget(QLabel("d_end (Å):"), 1, 0)
        lg.addWidget(self.ed_end, 1, 1)
        formula = QLabel("d(t) = d_start + (d_end − d_start)·t\n"
                         "t = j/(N−1):  0 at the substrate, 1 at the vacuum")
        formula.setObjectName("Formula")
        lg.addWidget(formula, 2, 0, 1, 2)
        lg.setRowStretch(3, 1)
        self.pages.addWidget(linear_page)

        poly_page = QWidget()
        pv = QVBoxLayout(poly_page)
        pv.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Degree:"))
        self.spin_deg = QSpinBox()
        self.spin_deg.setRange(1, MAX_DEGREE)
        row.addWidget(self.spin_deg)
        row.addStretch(1)
        pv.addLayout(row)
        self.lbl_formula = QLabel()
        self.lbl_formula.setObjectName("Formula")
        pv.addWidget(self.lbl_formula)
        grid = QGridLayout()
        self.coef_labels, self.coef_edits = [], []
        for k in range(MAX_DEGREE + 1):
            label, entry = QLabel(f"c{k}:"), QLineEdit()
            entry.setMaximumWidth(90)
            grid.addWidget(label, k // 2, 2 * (k % 2))
            grid.addWidget(entry, k // 2, 2 * (k % 2) + 1)
            self.coef_labels.append(label)
            self.coef_edits.append(entry)
        pv.addLayout(grid)
        pv.addStretch(1)
        self.pages.addWidget(poly_page)
        body.addWidget(self.pages, 1)

        self.fig = Figure(figsize=(2.5, 1.6), dpi=80, facecolor='#ffffff')
        self.canvas = MiniCanvas(self.fig)
        self.canvas.setFixedSize(240, 150)
        self.ax = self.fig.add_subplot(111)
        body.addWidget(self.canvas)
        outer.addLayout(body)

        self.lbl_warn = QLabel("")
        self.lbl_warn.setStyleSheet("color: #cf222e; font-weight: bold; background: transparent;")
        outer.addWidget(self.lbl_warn)

        # ---- connections ----
        for entry in [self.ed_name, self.ed_pot, self.ed_start, self.ed_end] + self.coef_edits:
            entry.textChanged.connect(self.read_entries)
        self.cmb_mode.currentIndexChanged.connect(self.on_mode_changed)
        self.spin_deg.valueChanged.connect(self.on_degree_changed)
        self.btn_remove.clicked.connect(lambda: self.remove_requested.emit(self))

        self.show_material()

    # ---- entries -> material ----
    def read_entries(self, *_):
        if self._showing:
            return
        mat = self.material
        mat.name = self.ed_name.text().strip()
        mat.potential = safe_float(self.ed_pot.text(), mat.potential)
        mat.linear[0] = safe_float(self.ed_start.text(), mat.linear[0])
        mat.linear[1] = safe_float(self.ed_end.text(), mat.linear[1])
        for k in range(len(mat.poly)):
            mat.poly[k] = safe_float(self.coef_edits[k].text(), mat.poly[k])
        self.draw_graph()
        self.changed.emit()

    def on_mode_changed(self, index):
        if self._showing:
            return
        if index == 1:
            self.material.switch_to_poly()        # polynomial starts from the current straight line
        else:
            self.material.mode = "linear"
        self.show_material()
        self.changed.emit()

    def on_degree_changed(self, degree):
        if self._showing:
            return
        self.material.set_degree(degree)
        self.show_material()
        self.changed.emit()

    # ---- material -> entries ----
    def show_material(self):
        mat = self.material
        degree = len(mat.poly) - 1
        self._showing = True
        self.ed_name.setText(mat.name)
        self.ed_pot.setText(f"{mat.potential:.4f}")
        self.ed_start.setText(f"{mat.linear[0]:.2f}")
        self.ed_end.setText(f"{mat.linear[1]:.2f}")
        self.cmb_mode.setCurrentIndex(1 if mat.mode == "poly" else 0)
        self.pages.setCurrentIndex(1 if mat.mode == "poly" else 0)
        self.spin_deg.setValue(degree)
        for k, (label, entry) in enumerate(zip(self.coef_labels, self.coef_edits)):
            label.setVisible(k <= degree)
            entry.setVisible(k <= degree)
            entry.setText(f"{mat.poly[k]:.8g}" if k <= degree else "0")
        terms = ["c0", "c1·t"] + [f"c{k}·t{str(k).translate(SUPERSCRIPT)}" for k in range(2, degree + 1)]
        self.lbl_formula.setText("d(t) = " + " + ".join(terms))
        self._showing = False
        self.draw_graph()

    def draw_graph(self):
        """Thickness of this material in every repeat j = 0 … N−1."""
        N = max(self.n_repeats, 1)
        j = np.arange(N)
        d = self.material.thickness(j / (N - 1) if N > 1 else 0.0 * j)

        self.ax.clear()
        self.ax.plot(j, d, '-o' if N <= 150 else '-', color=self.color, lw=1.4, ms=2.5)
        negative = int(np.sum(d < 0))
        if negative:
            self.ax.axhline(0.0, color='#cf222e', lw=0.8, ls='--')
            self.lbl_warn.setText(f"⚠ d < 0 in {negative} of {N} repeats — built as 0 Å")
        self.lbl_warn.setVisible(negative > 0)
        self.ax.set_title(f"{self.material.name or '—'}: {d.min():.1f} … {d.max():.1f} Å", fontsize=8, pad=3)
        self.ax.set_xlabel("repeat j  (0 = substrate side)", fontsize=7, labelpad=1)
        self.ax.set_ylabel("d (Å)", fontsize=7, labelpad=1)
        self.ax.tick_params(labelsize=6.5, length=2, pad=1)
        self.ax.grid(True, ls='--', alpha=0.4, lw=0.6)
        try:
            self.fig.tight_layout(pad=0.4)
        except Exception:
            pass
        self.canvas.draw_idle()


# ==============================================================================
# Main window
# ==============================================================================
class MultilayerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neutron Multilayer Reflectivity Designer")
        self.resize(1400, 820)

        self.materials = []          # list of Material — the one place where the structure lives
        self.cards = []              # one LayerCard per material (same order)
        self.m = 2.0                 # supermirror m: high reflectivity wanted for 0 < q ≤ m·q_c
        self.running = False         # True while the optimizer loop runs
        self.mean_R = 0.0            # ⟨R⟩ of the structure shown in the plot
        self.calc_seconds = 0.0      # duration of the last plot update

        # recalculate 150 ms after the last change (not on every keystroke)
        self.calc_timer = QTimer(self)
        self.calc_timer.setSingleShot(True)
        self.calc_timer.setInterval(150)
        self.calc_timer.timeout.connect(self.update_calculation)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        splitter = QSplitter(qenum(Qt, 'Orientation', 'Horizontal'))
        main_layout.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.build_settings_box())
        left_layout.addWidget(self.build_optimizer_box())
        left_layout.addWidget(self.build_materials_box(), 1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(8, 8), dpi=100, facecolor='#ffffff')
        self.canvas = FigureCanvas(self.fig)
        right_layout.addWidget(NavigationToolbar(self.canvas, self))
        right_layout.addWidget(self.canvas)
        self.ax1 = self.fig.add_subplot(2, 1, 1)     # R(q)
        self.ax2 = self.fig.add_subplot(2, 1, 2)     # V(z)
        splitter.addWidget(right)
        splitter.setSizes([580, 820])

        # default structure: Ni / Ti, linear thickness laws
        self.add_material(Material("Ni", 0.2410, linear=(40.0, 200.0)))
        self.add_material(Material("Ti", -0.0490, linear=(40.0, 200.0)))
        self.on_m_changed(self.spin_m.value())
        QTimer.singleShot(50, self.update_calculation)

    # ------------------------------------------------------------------ panels
    def build_settings_box(self):
        box = QGroupBox("Multilayer & Interface Settings")
        self.settings_box = box
        grid = QGridLayout(box)
        self.spin_repeats = self.int_spin(1, 20000, 30)
        self.spin_repeats.valueChanged.connect(self.on_repeats_changed)
        self.spin_sublayers = self.int_spin(0, 50, 5)
        self.spin_d_sub = self.float_spin(0.1, 20.0, 1.5, 0.5, 2)
        self.spin_vsub = self.float_spin(-2.0, 5.0, 0.090, 0.01, 3)
        self.spin_vvac = self.float_spin(-2.0, 5.0, 0.0, 0.01, 3)
        for w in (self.spin_sublayers, self.spin_d_sub, self.spin_vsub, self.spin_vvac):
            w.valueChanged.connect(self.schedule_update)
        grid.addWidget(QLabel("Repetitions (N):"), 0, 0); grid.addWidget(self.spin_repeats, 0, 1)
        grid.addWidget(QLabel("Interface sublayers (n):"), 1, 0); grid.addWidget(self.spin_sublayers, 1, 1)
        grid.addWidget(QLabel("Sublayer step d (Å):"), 1, 2); grid.addWidget(self.spin_d_sub, 1, 3)
        grid.addWidget(QLabel("V_sub (µeV):"), 2, 0); grid.addWidget(self.spin_vsub, 2, 1)
        grid.addWidget(QLabel("V_vac (µeV):"), 2, 2); grid.addWidget(self.spin_vvac, 2, 3)
        return box

    def build_optimizer_box(self):
        box = QGroupBox("Supermirror Optimizer — gradient descent on the polynomial coefficients")
        grid = QGridLayout(box)
        self.spin_m = self.float_spin(1.01, 10.0, self.m, 0.1, 2)
        self.spin_m.valueChanged.connect(self.on_m_changed)
        self.spin_target = self.float_spin(0.05, 1.0, 0.92, 0.005, 3)
        self.spin_target.valueChanged.connect(self.schedule_update)
        self.lbl_reference = QLabel("")
        self.lbl_reference.setObjectName("Formula")
        self.lbl_reference.setWordWrap(True)
        self.spin_nstart = self.int_spin(2, 20000, 50)
        self.spin_nmax = self.int_spin(2, 20000, 150)
        self.spin_growth = self.float_spin(1.01, 3.0, 1.15, 0.05, 2)
        self.spin_iter = self.int_spin(5, 5000, 150)
        self.spin_step = self.float_spin(0.01, 50.0, 2.0, 0.5, 2)
        self.spin_dmin = self.float_spin(0.0, 100.0, 10.0, 1.0, 1)
        self.btn_hm = QPushButton("Hayter–Mook start profile")
        self.btn_hm.setToolTip("Set all materials to a q⁴-law supermirror profile for the current m")
        self.btn_hm.clicked.connect(lambda: self.set_hayter_mook())
        self.btn_start = QPushButton("▶ Start optimization")
        self.btn_start.clicked.connect(lambda: self.start_optimization())
        self.btn_stop = QPushButton("■ Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(lambda: self.stop_optimization())
        self.lbl_status = QLabel("Idle.")
        self.lbl_status.setObjectName("OptStatus")
        self.lbl_status.setWordWrap(True)

        rows = [
            ("m:", self.spin_m, "Target ⟨R⟩:", self.spin_target),
            ("N start:", self.spin_nstart, "N max:", self.spin_nmax),
            ("N growth ×:", self.spin_growth, "Max iter / N:", self.spin_iter),
            ("Step (Å):", self.spin_step, "Min d (Å):", self.spin_dmin),
        ]
        grid.addWidget(self.lbl_reference, 0, 0, 1, 4)
        for r, (l1, w1, l2, w2) in enumerate(rows, start=1):
            grid.addWidget(QLabel(l1), r, 0); grid.addWidget(w1, r, 1)
            grid.addWidget(QLabel(l2), r, 2); grid.addWidget(w2, r, 3)
        buttons = QHBoxLayout()
        for b in (self.btn_hm, self.btn_start, self.btn_stop):
            buttons.addWidget(b)
        grid.addLayout(buttons, 5, 0, 1, 4)
        grid.addWidget(self.lbl_status, 6, 0, 1, 4)
        self.optimizer_inputs = [self.spin_m, self.spin_target, self.spin_nstart, self.spin_nmax,
                                 self.spin_growth, self.spin_iter, self.spin_step, self.spin_dmin,
                                 self.btn_hm, self.btn_start]
        return box

    def build_materials_box(self):
        box = QGroupBox("Materials of the repeating unit — thickness laws d(t)")
        self.materials_box = box
        layout = QVBoxLayout(box)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(qenum(Qt, 'ScrollBarPolicy', 'ScrollBarAlwaysOff'))
        container = QWidget()
        container.setObjectName("CardsContainer")
        self.cards_layout = QVBoxLayout(container)
        self.cards_layout.setContentsMargins(2, 2, 2, 2)
        self.cards_layout.setSpacing(6)
        self.cards_layout.addStretch(1)
        scroll.setWidget(container)
        layout.addWidget(scroll)
        buttons = QHBoxLayout()
        btn_add = QPushButton("+ Add Material")
        btn_add.clicked.connect(lambda: self.add_material(Material("Layer", 0.100, linear=(50.0, 50.0))))
        btn_del = QPushButton("- Remove Last Material")
        btn_del.clicked.connect(lambda: self.remove_card(self.cards[-1]) if self.cards else None)
        buttons.addWidget(btn_add)
        buttons.addWidget(btn_del)
        layout.addLayout(buttons)
        return box

    @staticmethod
    def int_spin(lo, hi, value):
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(value)
        return w

    @staticmethod
    def float_spin(lo, hi, value, step, decimals):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(decimals)
        w.setSingleStep(step)
        w.setValue(value)
        return w

    # ------------------------------------------------------------------ materials
    def add_material(self, material):
        card = LayerCard(material, self.spin_repeats.value(), LAYER_COLORS[len(self.cards) % len(LAYER_COLORS)])
        card.changed.connect(self.schedule_update)
        card.remove_requested.connect(self.remove_card)
        self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)
        self.materials.append(material)
        self.cards.append(card)
        self.schedule_update()

    def remove_card(self, card):
        self.materials.remove(card.material)
        self.cards.remove(card)
        self.cards_layout.removeWidget(card)
        card.setParent(None)
        card.deleteLater()
        self.schedule_update()

    def on_repeats_changed(self, N):
        for card in self.cards:
            card.n_repeats = N
            card.draw_graph()
        self.schedule_update()

    def show_structure(self, N):
        """Put N and all material values (e.g. changed by the optimizer) into the entries, then recalculate."""
        self.spin_repeats.blockSignals(True)
        self.spin_repeats.setValue(N)
        self.spin_repeats.blockSignals(False)
        for card in self.cards:
            card.n_repeats = N
            card.show_material()
        self.update_calculation()

    # ------------------------------------------------------------------ target
    def target_edge(self):
        """(q_c, q_top = m·q_c, name) — q_c of the material with the highest potential."""
        if not self.materials:
            return 0.0215, 0.0215 * self.m, "Ni"
        top = max(self.materials, key=lambda mat: mat.potential)
        q_c = critical_q(top.potential) if top.potential > 0 else 0.0215
        return q_c, self.m * q_c, top.name

    def on_m_changed(self, value):
        self.m = float(value)
        layers, r_low, r_high = reference_for_m(self.m)
        repeats = layers / max(len(self.materials), 1)
        self.spin_target.setValue(round(r_low, 3))
        self.spin_nstart.setValue(max(2, round(0.7 * repeats)))
        self.spin_nmax.setValue(max(3, int(np.ceil(2.0 * repeats))))
        q_c, q_top, name = self.target_edge()
        self.lbl_reference.setText(
            f"Reference for m = {self.m:.2f}: ~{layers:,.0f} layers (≈{repeats:,.0f} repeats), "
            f"R ≈ {100 * r_low:.0f}–{100 * r_high:.0f} %.  Target range 0 … m·q_c = {q_top:.4f} Å⁻¹ "
            f"(q_c {name} = {q_c:.4f} Å⁻¹).")
        self.schedule_update()

    # ------------------------------------------------------------------ calculation + plots
    def schedule_update(self, *_):
        self.calc_timer.start()

    def update_calculation(self):
        started = time.time()
        self.ax1.clear()
        self.ax2.clear()
        if not self.materials:
            self.canvas.draw_idle()
            return

        N = self.spin_repeats.value()
        q_c, q_top, qc_name = self.target_edge()
        target = self.spin_target.value()

        # METHOD 1 and METHOD 2 — the same calls the optimizer uses
        potentials, thicknesses = build_stack(self.materials, N, self.spin_vvac.value(), self.spin_vsub.value(),
                                              self.spin_sublayers.value(), self.spin_d_sub.value())
        q = q_grid(q_top)
        R = reflectivity(potentials, thicknesses, q)
        self.mean_R = mean_reflectivity(R, q, q_top)
        near_edge = (q >= 0.95 * q_top) & (q <= q_top)
        edge_R = float(np.mean(R[near_edge]))
        ok = self.mean_R >= target

        # --- R(q) ---
        a = self.ax1
        a.axvspan(0.0, q_top, color='#1a7f37' if ok else '#bf8700', alpha=0.07,
                  label=f'target range 0 … m·$q_c$ = {q_top:.4f} Å⁻¹')
        a.semilogy(q, np.clip(R, 1e-6, 1.2), color='#0052cc', lw=1.6, label='Parratt')
        a.axvline(q_c, color='#e02424', ls='--', lw=1.2, alpha=0.7, label=f'{qc_name} $q_c$ ({q_c:.4f} Å⁻¹)')
        a.axvline(q_top, color='#1a7f37', ls=':', lw=1.4, label=f'm·$q_c$ (m = {self.m:.2f})')
        a.hlines(target, 0.0, q_top, colors='#57606a', linestyles='--', lw=1.0, label=f'target ⟨R⟩ = {target:.3f}')
        a.set_title(f"Neutron Specular Reflectivity $R(q)$\n"
                    f"N = {N} ({N * len(self.materials)} layers)  ·  ⟨R⟩(0…m·$q_c$) = {self.mean_R:.4f} "
                    f"{'✓' if ok else '✗'} (target {target:.3f})  ·  R near m·$q_c$ ≈ {edge_R:.3f}",
                    fontsize=9.5, fontweight='bold', pad=6)
        a.set_xlabel("Momentum transfer $q$ (Å⁻¹)", fontsize=9)
        a.set_ylabel("Reflectivity $R$", fontsize=9)
        a.set_xlim(0.0, q[-1])
        a.set_ylim(1e-5, 1.3)
        a.grid(True, which="both", ls="--", alpha=0.4)
        a.legend(loc='lower left', fontsize=7.5, framealpha=0.9)

        # --- V(z): each layer drawn from its top edge to its bottom edge; 50 Å of vacuum/substrate shown ---
        edges = np.concatenate(([-50.0], np.cumsum(thicknesses[:-1]), [thicknesses.sum() + 50.0]))
        b = self.ax2
        b.stairs(potentials, edges, baseline=0.0, fill=True, color='#c81e1e', alpha=0.15)
        b.stairs(potentials, edges, color='#c81e1e', lw=1.5)
        b.set_title(f"Potential Profile $V(z)$ ({len(potentials) - 2} layers incl. interface steps)",
                    fontsize=11, fontweight='bold', pad=8)
        b.set_xlabel("Depth $z$ (Å)", fontsize=9)
        b.set_ylabel("Potential $V$ (µeV)", fontsize=9)
        b.set_xlim(edges[0], edges[-1])
        span = potentials.max() - potentials.min()
        b.set_ylim(potentials.min() - 0.2 * span - 0.01, potentials.max() + 0.2 * span + 0.01)
        b.grid(True, ls="--", alpha=0.4)

        self.fig.tight_layout(pad=2.2)
        self.canvas.draw_idle()
        self.calc_seconds = time.time() - started

    # ------------------------------------------------------------------ optimizer
    def set_hayter_mook(self):
        q_c, _, _ = self.target_edge()
        hayter_mook_start(self.materials, self.m, q_c)
        self.show_structure(self.spin_repeats.value())

    def start_optimization(self):
        if not self.materials or self.running:
            return
        q_c, q_top, _ = self.target_edge()
        N_start = self.spin_nstart.value()
        gd = GradientDescent(self.materials, N_start,
                             self.spin_vvac.value(), self.spin_vsub.value(),
                             self.spin_sublayers.value(), self.spin_d_sub.value(),
                             q_grid(q_top), q_top,
                             step=self.spin_step.value(), d_min=self.spin_dmin.value())
        self.running = True
        self.set_inputs_enabled(False)
        self.opt_started = time.time()
        self.last_refresh = 0.0

        status = optimize(gd, self.spin_target.value(), N_start,
                          max(self.spin_nmax.value(), N_start), self.spin_growth.value(),
                          self.spin_iter.value(), self.on_optimizer_step)

        self.running = False
        self.set_inputs_enabled(True)
        self.show_structure(gd.N)                      # recalculates -> self.mean_R of the final structure
        head = {"success": "✓ Target reached",
                "n_max": "✗ N max reached without meeting the target — best result of the last N kept",
                "stopped": "■ Stopped — best result of the current N kept"}[status]
        self.lbl_status.setText(f"{head}: N = {gd.N} ({gd.N * len(self.materials)} layers), "
                                f"⟨R⟩ = {self.mean_R:.4f} (target {self.spin_target.value():.3f}), "
                                f"{time.time() - self.opt_started:.0f} s.")

    def on_optimizer_step(self, N, iteration, mean_R, best_R):
        """Called by optimize() every iteration. Returns False when Stop was pressed."""
        self.lbl_status.setText(
            f"N = {N} ({N * len(self.materials)} layers) · iteration {iteration} · ⟨R⟩ = {mean_R:.4f} "
            f"(best at this N {best_R:.4f}) → target {self.spin_target.value():.3f} · "
            f"{time.time() - self.opt_started:.0f} s")
        # redraw entries and plots now and then (a redraw takes as long as one reflectivity calculation)
        if time.time() - self.last_refresh > max(1.5, 4 * self.calc_seconds):
            self.show_structure(N)
            self.last_refresh = time.time()
        QApplication.processEvents()                   # keep the window alive, let Stop be clicked
        return self.running

    def stop_optimization(self):
        self.running = False

    def set_inputs_enabled(self, enabled):
        for w in [self.settings_box, self.materials_box] + self.optimizer_inputs:
            w.setEnabled(enabled)
        self.btn_stop.setEnabled(not enabled)

    def closeEvent(self, event):
        self.running = False
        super().closeEvent(event)
