"""
gui.py — user interface (PyQt5/PyQt6 + embedded matplotlib)

    PARAMETERS      table of every adjustable number: label, start value, limits, where it is shown
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
        QScrollArea, QFrame, QLineEdit, QComboBox, QStackedWidget,
        QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox, QSplitter,
        QScrollArea, QFrame, QLineEdit, QComboBox, QStackedWidget,
        QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView
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

from calculation import (Material, Interface, build_stack, reflectivity, critical_q, q_grid, mean_reflectivity,
                         energy_from_k, k_from_energy, glancing_angle, q_from_angle, H_OVER_M)
from optimizer import M_REFERENCE, reference_for_m, hayter_mook_start, GradientDescent, optimize
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
# Every adjustable number of the program: shown and edited in the app
# ==============================================================================
# decimals = 0 gives an integer entry, a tuple as start value a choice list. The group decides where it appears.
#  key              label                          start           min      max       step     dec  group       tooltip
PARAMETERS = [
    ("N",              "Repetitions N",              30,             1,       20000,    1,       0,   "structure", "How many times the unit cell (all materials) is repeated"),
    ("V_sub",          "V_sub (µeV)",                0.090,          -2.0,    5.0,      0.01,    3,   "structure", "Potential of the substrate"),
    ("V_vac",          "V_vac (µeV)",                0.0,            -2.0,    5.0,      0.01,    3,   "structure", "Potential of the medium above the mirror"),

    ("profile",        "Profile",                    ("erf", "linear"), None, None,   None,    None, "interface", "erf: Gaussian roughness · linear: straight ramp with the same rms width"),
    ("sigma",          "Roughness σ (Å)",            3.0,            0.0,     1000.0,   0.5,     2,   "interface", "rms width of every boundary transition; 0 = sharp. σ ≫ period → fully intermixed average medium"),
    ("n_slices",       "Slices per transition",      10,             1,       200,      1,       0,   "interface", "The smooth profile is cut into this many slices across each transition (erf: ±4σ)"),
    ("chaos",          "Chaos 0…1",                  0.0,            0.0,     1.0,      0.05,    2,   "interface", "Every boundary gets its own width σ·(1 + chaos·u), u random in [−1, 1]"),
    ("seed",           "Random seed",                1,              0,       999999,   1,       0,   "interface", "Same seed = same random structure (reproducible curves); change it for another realization"),

    ("energy",         "Energy E (meV)",             energy_from_k(2 * np.pi / 5.0), 1e-4, 1e5, 0.1, 4, "beam", "Neutron energy — editing it updates k"),
    ("k",              "Momentum k = p/ħ (Å⁻¹)",     2 * np.pi / 5.0, 1e-3,   100.0,    0.01,    5,   "beam",      "Neutron wave vector — editing it updates E"),

    ("q_min",          "q min (Å⁻¹)",                0.001,          1e-4,    0.1,      0.001,   4,   "plot",      "Lowest q of the calculation grid"),
    ("q_max_factor",   "q max / (m·q_c)",            1.3,            1.0,     5.0,      0.1,     2,   "plot",      "Highest q of the grid, as a multiple of the target edge m·q_c"),
    ("q_points",       "q points",                   800,            50,      20000,    50,      0,   "plot",      "Number of q values (plots and optimizer use the same grid)"),
    ("edge_fraction",  "Edge window from (× m·q_c)", 0.95,           0.5,     0.999,    0.01,    3,   "plot",      "'R near m·q_c' = mean R between this fraction of m·q_c and m·q_c"),
    ("R_scale",        "R axis",                     ("log", "linear"), None, None,   None,    None, "plot",      "Reflectivity axis: logarithmic or linear"),
    ("log_R_min",      "R axis min (log₁₀)",         -5,             -12,     -1,       1,       0,   "plot",      "Lower limit of the reflectivity axis (log scale only)"),
    ("z_pad",          "Vacuum / substrate shown (Å)", 50.0,         0.0,     10000.0,  10.0,    0,   "plot",      "Width of vacuum and substrate drawn in the V(z) plot"),

    ("m",              "m",                          2.0,            1.01,    10.0,     0.1,     2,   "target",    "High reflectivity is wanted for 0 < q ≤ m·q_c"),
    ("target",         "Target ⟨R⟩",                 0.92,           0.05,    1.0,      0.005,   3,   "target",    "The search stops when ⟨R⟩ over (0, m·q_c] reaches this (filled from the reference table)"),

    ("N_start",        "N start",                    50,             2,       20000,    1,       0,   "search",    "First N tried (filled from the reference table)"),
    ("N_max",          "N max",                      150,            2,       20000,    1,       0,   "search",    "Largest N allowed (filled from the reference table)"),
    ("growth",         "N growth ×",                 1.15,           1.01,    3.0,      0.05,    2,   "search",    "N is multiplied by this when a stage ends without reaching the target"),
    ("max_iter",       "Max iterations / N",         150,            5,       5000,     10,      0,   "search",    "Gradient-descent iterations allowed for one N"),
    ("step",           "Adam step (Å)",              2.0,            0.01,    50.0,     0.5,     2,   "search",    "How far every coefficient moves in one iteration (at the start)"),
    ("d_min",          "Min layer d (Å)",            10.0,           0.0,     100.0,    1.0,     1,   "search",    "Layers thinner than this are penalized"),

    ("penalty_weight", "Thin-layer penalty weight",  10.0,           0.0,     1000.0,   1.0,     2,   "advanced",  "Weight of the d < d_min penalty in the loss"),
    ("h",              "Gradient probe h (Å)",       0.02,           1e-4,    5.0,      0.01,    4,   "advanced",  "Coefficient change used to measure the gradient"),
    ("step_decay",     "Step decay (iterations)",    100,            1,       1000000,  10,      0,   "advanced",  "step = step₀ / (1 + iteration / decay)"),
    ("beta1",          "Adam β1",                    0.9,            0.0,     0.9999,   0.01,    4,   "advanced",  "Memory of the gradient (momentum)"),
    ("beta2",          "Adam β2",                    0.999,          0.0,     0.99999,  0.001,   5,   "advanced",  "Memory of the squared gradient (step normalization)"),
    ("patience",       "Patience (iterations)",      20,             1,       10000,    1,       0,   "advanced",  "A stage ends after this many iterations without gain"),
    ("min_gain",       "Min ⟨R⟩ gain",               1e-5,           0.0,     0.1,      1e-5,    6,   "advanced",  "Smaller improvements of ⟨R⟩ do not count as gain"),
    ("N_start_factor", "N start = × reference",      0.7,            0.05,    10.0,     0.05,    2,   "advanced",  "N start filled as this factor × reference repeats when m changes"),
    ("N_max_factor",   "N max = × reference",        2.0,            0.1,     20.0,     0.1,     2,   "advanced",  "N max filled as this factor × reference repeats when m changes"),
    ("refresh_s",      "Refresh plots every (s)",    1.5,            0.1,     600.0,    0.5,     1,   "advanced",  "During optimization the plots are redrawn at most this often"),
]

GROUP_TITLES = {
    "structure": "Multilayer",
    "interface": "Interfaces (roughness)",
    "beam":      "Neutron beam",
    "plot":      "q grid & plots",
    "target":    "Supermirror target",
    "search":    "Search",
    "advanced":  "Advanced optimizer settings",
}
REDRAW_GROUPS = ("structure", "interface", "beam", "plot", "target")     # changing these recalculates the plots


# ==============================================================================
# Main window
# ==============================================================================
class MultilayerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neutron Multilayer Reflectivity Designer")
        self.resize(1400, 860)

        self.materials = []                                  # list of Material — the structure
        self.cards = []                                      # one LayerCard per material (same order)
        self.params = {}                                     # key -> entry widget, for every PARAMETERS row
        self.reference = [row[:] for row in M_REFERENCE]     # editable copy of the reference table
        self.m = 2.0                                         # supermirror m
        self.running = False                                 # True while the optimizer loop runs
        self.mean_R = 0.0                                    # ⟨R⟩ of the structure shown in the plot
        self.calc_seconds = 0.0                              # duration of the last plot update

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

        # ---- left: tabs + run controls ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.pages = [self.build_structure_tab(), self.build_beam_tab(),
                      self.build_optimizer_tab(), self.build_reference_tab()]
        for page, title in zip(self.pages, ["Structure", "Beam && plots", "Optimizer", "Reference table"]):
            self.tabs.addTab(page, title)
        left_layout.addWidget(self.tabs, 1)

        run_row = QHBoxLayout()
        self.btn_start = QPushButton("▶ Start optimization")
        self.btn_start.clicked.connect(lambda: self.start_optimization())
        self.btn_stop = QPushButton("■ Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(lambda: self.stop_optimization())
        run_row.addWidget(self.btn_start)
        run_row.addWidget(self.btn_stop)
        left_layout.addLayout(run_row)
        self.lbl_status = QLabel("Idle.")
        self.lbl_status.setObjectName("OptStatus")
        self.lbl_status.setWordWrap(True)
        left_layout.addWidget(self.lbl_status)
        splitter.addWidget(left)

        # ---- right: plots ----
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
        splitter.setSizes([600, 800])

        # ---- connections of the parameter entries (special handlers first, then the redraw) ----
        self.params["N"].valueChanged.connect(self.on_repeats_changed)
        self.params["m"].valueChanged.connect(self.on_m_changed)
        self.params["energy"].valueChanged.connect(self.on_energy_changed)
        self.params["k"].valueChanged.connect(self.on_k_changed)
        for key in ("N_start_factor", "N_max_factor"):
            self.params[key].valueChanged.connect(lambda *_: self.on_m_changed(self.v("m")))
        for key, label, start, lo, hi, step, dec, group, tip in PARAMETERS:
            if group in REDRAW_GROUPS:
                widget = self.params[key]
                changed = widget.currentIndexChanged if isinstance(widget, QComboBox) else widget.valueChanged
                changed.connect(self.schedule_update)

        # default structure: Ni / Ti, linear thickness laws
        self.add_material(Material("Ni", 0.2410, linear=(40.0, 200.0)))
        self.add_material(Material("Ti", -0.0490, linear=(40.0, 200.0)))
        self.on_m_changed(self.v("m"))
        QTimer.singleShot(50, self.update_calculation)

    # ------------------------------------------------------------------ parameters
    def v(self, key):
        """Current value of a parameter entry (text for a choice list)."""
        widget = self.params[key]
        return widget.currentText() if isinstance(widget, QComboBox) else widget.value()

    def set_v(self, key, value):
        """Set a parameter entry without triggering its own handlers."""
        widget = self.params[key]
        widget.blockSignals(True)
        widget.setValue(value)
        widget.blockSignals(False)

    def build_group(self, group):
        """Group box with the entries of all PARAMETERS rows of this group, two per row."""
        box = QGroupBox(GROUP_TITLES[group])
        grid = QGridLayout(box)
        rows = [p for p in PARAMETERS if p[7] == group]
        for i, (key, label, start, lo, hi, step, dec, _, tip) in enumerate(rows):
            if isinstance(start, tuple):
                widget = QComboBox()
                widget.addItems(start)
            elif dec == 0:
                widget = QSpinBox()
                widget.setRange(int(lo), int(hi))
                widget.setSingleStep(int(step))
                widget.setValue(int(round(start)))
            else:
                widget = QDoubleSpinBox()
                widget.setDecimals(dec)
                widget.setRange(lo, hi)
                widget.setSingleStep(step)
                widget.setValue(start)
            widget.setToolTip(tip)
            text = QLabel(label + ":")
            text.setToolTip(tip)
            grid.addWidget(text, i // 2, 2 * (i % 2))
            grid.addWidget(widget, i // 2, 2 * (i % 2) + 1)
            self.params[key] = widget
        return box, grid

    def add_full_row(self, grid, widget):
        grid.addWidget(widget, grid.rowCount(), 0, 1, 4)

    # ------------------------------------------------------------------ tabs
    def build_structure_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        for group in ("structure", "interface"):
            box, _ = self.build_group(group)
            layout.addWidget(box)
        layout.addWidget(self.build_materials_box(), 1)
        return page

    def build_beam_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        box, grid = self.build_group("beam")
        self.lbl_beam = QLabel("")
        self.lbl_beam.setObjectName("Formula")
        self.lbl_beam.setWordWrap(True)
        self.add_full_row(grid, self.lbl_beam)
        layout.addWidget(box)
        box, _ = self.build_group("plot")
        layout.addWidget(box)
        layout.addStretch(1)
        return page

    def build_optimizer_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        box, grid = self.build_group("target")
        self.lbl_reference = QLabel("")
        self.lbl_reference.setObjectName("Formula")
        self.lbl_reference.setWordWrap(True)
        self.add_full_row(grid, self.lbl_reference)
        layout.addWidget(box)
        box, grid = self.build_group("search")
        btn_hm = QPushButton("Hayter–Mook start profile")
        btn_hm.setToolTip("Set all materials to a q⁴-law supermirror profile for the current m")
        btn_hm.clicked.connect(lambda: self.set_hayter_mook())
        self.add_full_row(grid, btn_hm)
        layout.addWidget(box)
        box, _ = self.build_group("advanced")
        layout.addWidget(box)
        layout.addStretch(1)
        return page

    def build_reference_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel("Typical total layer count and reflectivity of real supermirrors. When m changes, "
                      "Target ⟨R⟩ = R low, N start and N max are filled from this table "
                      "(layers interpolated log-log, R linearly; the last row pair is extended above).")
        note.setObjectName("Formula")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.ref_table = QTableWidget(0, 4)
        self.ref_table.setHorizontalHeaderLabels(["m", "Total layers", "R low", "R high"])
        self.ref_table.horizontalHeader().setSectionResizeMode(qenum(QHeaderView, 'ResizeMode', 'Stretch'))
        for row in self.reference:
            self.append_reference_row(row)
        self.ref_table.itemChanged.connect(self.read_reference_table)
        layout.addWidget(self.ref_table, 1)
        buttons = QHBoxLayout()
        btn_add = QPushButton("+ Add row")
        btn_add.clicked.connect(lambda: self.append_reference_row(["", "", "", ""]))
        btn_del = QPushButton("− Remove selected row")
        btn_del.clicked.connect(lambda: self.remove_reference_row())
        buttons.addWidget(btn_add)
        buttons.addWidget(btn_del)
        layout.addLayout(buttons)
        self.lbl_ref_status = QLabel("")
        self.lbl_ref_status.setObjectName("Formula")
        layout.addWidget(self.lbl_ref_status)
        return page

    def build_materials_box(self):
        box = QGroupBox("Materials of the repeating unit — thickness laws d(t)")
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

    # ------------------------------------------------------------------ materials
    def add_material(self, material):
        card = LayerCard(material, self.v("N"), LAYER_COLORS[len(self.cards) % len(LAYER_COLORS)])
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

    def show_structure(self, N):
        """Put N and all material values (e.g. changed by the optimizer) into the entries, then recalculate."""
        self.set_v("N", N)
        for card in self.cards:
            card.n_repeats = N
            card.show_material()
        self.update_calculation()

    # ------------------------------------------------------------------ beam: E <-> k
    def on_energy_changed(self, E):
        self.set_v("k", k_from_energy(E))

    def on_k_changed(self, k):
        self.set_v("energy", energy_from_k(k))

    # ------------------------------------------------------------------ reference table
    def append_reference_row(self, values):
        self.ref_table.blockSignals(True)
        r = self.ref_table.rowCount()
        self.ref_table.insertRow(r)
        for c, value in enumerate(values):
            self.ref_table.setItem(r, c, QTableWidgetItem("" if value == "" else f"{value:g}"))
        self.ref_table.blockSignals(False)

    def remove_reference_row(self):
        if self.ref_table.currentRow() >= 0:
            self.ref_table.removeRow(self.ref_table.currentRow())
            self.read_reference_table()

    def read_reference_table(self, *_):
        """Table -> self.reference (complete rows only, sorted by m), then refill the target values."""
        rows = []
        for r in range(self.ref_table.rowCount()):
            cells = [self.ref_table.item(r, c) for c in range(4)]
            values = [safe_float(cell.text(), None) if cell else None for cell in cells]
            if None not in values and values[0] > 0 and values[1] > 0:
                rows.append(values)
        if len(rows) < 2:
            self.lbl_ref_status.setText("⚠ At least two complete rows are needed — table not applied.")
            return
        self.reference = sorted(rows)
        self.lbl_ref_status.setText(f"Applied: {len(rows)} rows.")
        self.on_m_changed(self.v("m"))

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
        layers, r_low, r_high = reference_for_m(self.m, self.reference)
        repeats = layers / max(len(self.materials), 1)
        self.set_v("target", round(r_low, 3))
        self.set_v("N_start", max(2, round(self.v("N_start_factor") * repeats)))
        self.set_v("N_max", max(3, int(np.ceil(self.v("N_max_factor") * repeats))))
        self.lbl_reference.setText(
            f"Reference for m = {self.m:.2f}: ~{layers:,.0f} layers (≈{repeats:,.0f} repeats), "
            f"R ≈ {100 * r_low:.0f}–{100 * r_high:.0f} %.")
        self.schedule_update()

    def interface(self):
        """Interface object built from the entries of the 'Interfaces' group."""
        return Interface(profile=self.v("profile"), sigma=self.v("sigma"), n_slices=self.v("n_slices"),
                         chaos=self.v("chaos"), seed=self.v("seed"))

    def q_values(self, q_top):
        return q_grid(q_top, self.v("q_min"), self.v("q_max_factor"), self.v("q_points"))

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

        N = self.v("N")
        k = self.v("k")
        q_c, q_top, qc_name = self.target_edge()
        target = self.v("target")

        # METHOD 1 and METHOD 2 — the same calls the optimizer uses
        potentials, thicknesses = build_stack(self.materials, N, self.v("V_vac"), self.v("V_sub"),
                                              self.interface())
        q = self.q_values(q_top)
        R = reflectivity(potentials, thicknesses, q)
        self.mean_R = mean_reflectivity(R, q, q_top)
        near_edge = q >= self.v("edge_fraction") * q_top
        edge_R = mean_reflectivity(R[near_edge], q[near_edge], q_top)
        ok = self.mean_R >= target

        # beam info: wavelength, speed, critical angles
        theta_c, theta_top = glancing_angle(q_c, k), glancing_angle(q_top, k)
        self.lbl_beam.setText(
            f"λ = {2 * np.pi / k:.3f} Å   ·   v = {H_OVER_M * k / (2 * np.pi):.1f} m/s\n"
            f"θ_c {qc_name} = {theta_c:.3f}°   ·   θ at m·q_c = {theta_top:.3f}°")

        # --- R(q), with the glancing angle of this beam on the top axis ---
        a = self.ax1
        a.axvspan(0.0, q_top, color='#1a7f37' if ok else '#bf8700', alpha=0.07,
                  label=f'target range 0 … m·$q_c$ = {q_top:.4f} Å⁻¹')
        log_scale = self.v("R_scale") == "log"
        if log_scale:
            a.semilogy(q, np.clip(R, 10.0 ** self.v("log_R_min") / 10, 1.2), color='#0052cc', lw=1.6, label='Parratt')
        else:
            a.plot(q, R, color='#0052cc', lw=1.6, label='Parratt')
        a.axvline(q_c, color='#e02424', ls='--', lw=1.2, alpha=0.7, label=f'{qc_name} $q_c$ ({q_c:.4f} Å⁻¹)')
        a.axvline(q_top, color='#1a7f37', ls=':', lw=1.4, label=f'm·$q_c$ (m = {self.m:.2f})')
        a.hlines(target, 0.0, q_top, colors='#57606a', linestyles='--', lw=1.0, label=f'target ⟨R⟩ = {target:.3f}')
        a.set_title(f"N = {N} ({N * len(self.materials)} layers)  ·  ⟨R⟩(0…m·$q_c$) = {self.mean_R:.4f} "
                    f"{'✓' if ok else '✗'} (target {target:.3f})  ·  R near m·$q_c$ ≈ {edge_R:.3f}",
                    fontsize=9.5, fontweight='bold', pad=6)
        a.set_xlabel("Momentum transfer $q$ (Å⁻¹)", fontsize=9)
        a.set_ylabel("Reflectivity $R$", fontsize=9)
        a.set_xlim(0.0, q[-1])
        a.set_ylim(*((10.0 ** self.v("log_R_min"), 1.3) if log_scale else (0.0, 1.05)))
        a.grid(True, which="both", ls="--", alpha=0.4)
        a.legend(loc='lower left' if log_scale else 'best', fontsize=7.5, framealpha=0.9)
        top_axis = a.secondary_xaxis('top', functions=(lambda x: glancing_angle(x, k),
                                                       lambda th: q_from_angle(th, k)))
        top_axis.set_xlabel(f"Glancing angle θ (°) at E = {self.v('energy'):.3f} meV", fontsize=8)
        top_axis.tick_params(labelsize=8)

        # --- V(z): each layer drawn from its top edge to its bottom edge ---
        pad = self.v("z_pad")
        edges = np.concatenate(([-pad], np.cumsum(thicknesses[:-1]), [thicknesses.sum() + pad]))
        b = self.ax2
        b.stairs(potentials, edges, baseline=0.0, fill=True, color='#c81e1e', alpha=0.15)
        b.stairs(potentials, edges, color='#c81e1e', lw=1.5)
        b.set_title(f"Potential Profile $V(z)$ ({len(potentials) - 2} slices, σ = {self.v('sigma'):g} Å)",
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
        self.show_structure(self.v("N"))

    def start_optimization(self):
        if not self.materials or self.running:
            return
        q_c, q_top, _ = self.target_edge()
        N_start = self.v("N_start")
        gd = GradientDescent(self.materials, N_start,
                             self.v("V_vac"), self.v("V_sub"), self.interface(),
                             self.q_values(q_top), q_top,
                             step=self.v("step"), step_decay=self.v("step_decay"),
                             d_min=self.v("d_min"), penalty_weight=self.v("penalty_weight"),
                             h=self.v("h"), beta1=self.v("beta1"), beta2=self.v("beta2"))
        self.running = True
        self.set_inputs_enabled(False)
        self.opt_started = time.time()
        self.last_refresh = 0.0

        status = optimize(gd, self.v("target"), N_start, max(self.v("N_max"), N_start),
                          self.v("growth"), self.v("max_iter"), self.v("patience"), self.v("min_gain"),
                          self.on_optimizer_step)

        self.running = False
        self.set_inputs_enabled(True)
        self.show_structure(gd.N)                      # recalculates -> self.mean_R of the final structure
        head = {"success": "✓ Target reached",
                "n_max": "✗ N max reached without meeting the target — best result of the last N kept",
                "stopped": "■ Stopped — best result of the current N kept"}[status]
        self.lbl_status.setText(f"{head}: N = {gd.N} ({gd.N * len(self.materials)} layers), "
                                f"⟨R⟩ = {self.mean_R:.4f} (target {self.v('target'):.3f}), "
                                f"{time.time() - self.opt_started:.0f} s.")

    def on_optimizer_step(self, N, iteration, mean_R, best_R):
        """Called by optimize() every iteration. Returns False when Stop was pressed."""
        self.lbl_status.setText(
            f"N = {N} ({N * len(self.materials)} layers) · iteration {iteration} · ⟨R⟩ = {mean_R:.4f} "
            f"(best at this N {best_R:.4f}) → target {self.v('target'):.3f} · "
            f"{time.time() - self.opt_started:.0f} s")
        # redraw entries and plots now and then (a redraw takes as long as one reflectivity calculation)
        if time.time() - self.last_refresh > max(self.v("refresh_s"), 4 * self.calc_seconds):
            self.show_structure(N)
            self.last_refresh = time.time()
        QApplication.processEvents()                   # keep the window alive, let Stop be clicked
        return self.running

    def stop_optimization(self):
        self.running = False

    def set_inputs_enabled(self, enabled):
        """During optimization all tabs stay viewable but not editable; only Stop works."""
        for page in self.pages:
            page.setEnabled(enabled)
        self.btn_start.setEnabled(enabled)
        self.btn_stop.setEnabled(not enabled)

    def closeEvent(self, event):
        self.running = False
        super().closeEvent(event)
