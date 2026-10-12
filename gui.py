"""
gui.py — user interface (PyQt5/PyQt6; the plots are drawn by plots.py with pyqtgraph)

    PARAMETERS      table of every adjustable number: label, start value, limits, where it is shown
    DropZone        area that accepts a dragged file (.xlsx structure, .xy measured curve)
    LayerCard       the entries of ONE Material object; typing changes the material directly
    MultilayerApp   main window: holds self.materials (list of Material), shows R(q) and V(z),
                    runs the optimizer loop in the program itself

Data flow:
    entries  ->  Material objects  ->  build_stack()  ->  neutron_reflectivity() / xray_reflectivity()  ->  plots
                                        (the depth plots always show the graded profile: build_stack(graded=True))
    optimizer changes the same Material objects  ->  show_material() puts the numbers back in the entries
    measured .xy curves  ->  read_xy()  ->  x converted to q with the beam's k  ->  points on the R(q) plots

Colors: gui_style.colors holds the colors of the current theme (light / dark);
apply_theme() switches it together with the Qt stylesheet.
"""
from pathlib import Path
import time

import numpy as np

# Universal PyQt5 / PyQt6 compatibility
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox, QSplitter,
        QScrollArea, QFrame, QLineEdit, QComboBox, QStackedWidget,
        QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QAbstractItemView
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QIcon, QPalette, QColor
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox, QSplitter,
        QScrollArea, QFrame, QLineEdit, QComboBox, QStackedWidget,
        QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QAbstractItemView
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QIcon, QPalette, QColor

from calculation import (Material, Interface, build_stack, neutron_reflectivity, xray_reflectivity,
                         PROFILE_NAMES, profile_weights,
                         critical_q, xray_critical_q, q_grid, mean_reflectivity,
                         energy_from_k, k_from_energy, glancing_angle, q_from_angle, H_OVER_M,
                         xray_wavelength, xray_energy)
from optimizer import M_REFERENCE, reference_for_m, hayter_mook_start, GradientDescent, optimize
from gui_style import THEMES, colors
from plots import PlotPanel, MiniPlot, style_plot, DASH, DOT
from structure_file import (LIBRARY_DIR, load_library, sheet_names, read_layers, periodic_materials,
                            read_thickness_file, read_xy)

MAX_DEGREE = 6

# Default X-ray permittivities at Cu Kα1 (λ = 1.540562 Å, 8.048 keV): ε = (1 − δ + iβ)², with
# δ, β = (r_e λ² / 2π)·n_atoms·(f1, f2), f1 = Z + f′, f2 = f″ (International Tables Vol. C, Cu Kα1)
#   Ni  8.908 g/cm³  f′ = −3.0029  f″ = 0.5091  -> θ_c = 0.400°
#   Ti  4.506 g/cm³  f′ =  0.2191  f″ = 1.8069  -> θ_c = 0.297°
#   SiO2 (fused silica, the substrate of V_sub = 0.090 µeV) 2.20 g/cm³ -> θ_c = 0.216°
EPS_NI = 0.999951363 + 9.9055e-07j
EPS_TI = 0.999973185 + 2.1806e-06j
EPS_SIO2 = 0.999985752 + 1.8523e-07j
SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
CURVE_FILES = (".xy", ".dat", ".txt", ".csv")
Q_LABEL = "Momentum transfer <i>q</i> (Å<sup>−1</sup>)"           # axis titles of the plots (HTML)
Z_LABEL = "Depth <i>z</i> (Å) from the top of the first slice (a rough surface starts above the nominal one)"             # measured curves; .xy is what Multifitting writes


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


def dropped_path(event, extensions=(".xlsx",)):
    """Path of the first file with one of these extensions in a drag-and-drop event, or None."""
    if not event.mimeData().hasUrls():
        return None
    for url in event.mimeData().urls():
        path = url.toLocalFile()
        if path.lower().endswith(tuple(extensions)):
            return path
    return None


def program_icon():
    """The program icon: icon.png (made from icon.svg) next to this file, else icon.svg itself."""
    folder = Path(__file__).resolve().parent
    for name in ("icon.png", "icon.ico", "icon.svg"):
        if (folder / name).exists():
            return QIcon(str(folder / name))
    return QIcon()


def mix_text(weights):
    """'100 % erf' or '70 % erf + 30 % exp': the normalized mix of a boundary profile."""
    w = profile_weights(weights)
    return " + ".join(f"{100 * wk:.0f} % {name}" for wk, name in zip(w, PROFILE_NAMES) if wk > 0)


class DropZone(QLabel):
    """Area that accepts a dragged file with one of the given extensions and reports its path."""
    file_dropped = pyqtSignal(str)

    def __init__(self, text, extensions=(".xlsx",)):
        super().__init__(text)
        self.extensions = extensions
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)
        self.setWordWrap(True)
        self.setAlignment(qenum(Qt, 'AlignmentFlag', 'AlignCenter'))
        self.setMinimumHeight(70)

    def dragEnterEvent(self, event):
        if dropped_path(event, self.extensions):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        path = dropped_path(event, self.extensions)
        if path:
            event.acceptProposedAction()
            self.file_dropped.emit(path)


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
    file_dropped = pyqtSignal(object, str) # (this card, path of an .xlsx with this material's thicknesses)

    def __init__(self, material, n_repeats, color, library=None):
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
        self.swatch = QLabel("■")
        self.swatch.setStyleSheet(f"color: {color}; font-size: 13pt; background: transparent;")
        head.addWidget(self.swatch)
        head.addWidget(QLabel("Material:"))
        self.cmb_name = QComboBox()                          # library materials; any other name can be typed
        self.cmb_name.setEditable(True)
        self.cmb_name.setInsertPolicy(qenum(QComboBox, 'InsertPolicy', 'NoInsert'))
        self.cmb_name.setMinimumWidth(105)
        self.cmb_name.setToolTip("Choose a material of the library (fills V and ε), or type any name")
        self.set_library(library or {})
        head.addWidget(self.cmb_name)
        head.addWidget(QLabel("V (µeV):"))
        self.ed_pot = QLineEdit()
        self.ed_pot.setMaximumWidth(75)
        head.addWidget(self.ed_pot)
        head.addStretch(1)
        head.addWidget(QLabel("d(t):"))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Linear", "Polynomial", "User-determined"])
        head.addWidget(self.cmb_mode)
        self.btn_remove = QPushButton("✕")
        self.btn_remove.setFixedWidth(28)
        self.btn_remove.setToolTip("Remove this material")
        head.addWidget(self.btn_remove)
        outer.addLayout(head)

        # ---- X-ray permittivity: ε = real + i·imag ----
        eps_row = QHBoxLayout()
        eps_row.addWidget(QLabel("X-ray  ε ="))
        self.ed_eps_re = QLineEdit()
        self.ed_eps_re.setMaximumWidth(115)
        self.ed_eps_re.setToolTip("Real part of the dielectric permittivity at the X-ray energy in use")
        eps_row.addWidget(self.ed_eps_re)
        eps_row.addWidget(QLabel("+ i"))
        self.ed_eps_im = QLineEdit()
        self.ed_eps_im.setMaximumWidth(115)
        self.ed_eps_im.setToolTip("Imaginary part (absorption, > 0)")
        eps_row.addWidget(self.ed_eps_im)
        eps_row.addStretch(1)
        outer.addLayout(eps_row)

        # ---- upper boundary of every layer of this material: width σ and shape (mix of erf, lin, exp) ----
        rough_row = QHBoxLayout()
        rough_row.addWidget(QLabel("Upper boundary:  σ (Å)"))
        self.ed_sigma = QLineEdit()
        self.ed_sigma.setMaximumWidth(55)
        self.ed_sigma.setToolTip("rms roughness of the upper boundary of every layer of this material (0 = sharp)")
        rough_row.addWidget(self.ed_sigma)
        rough_row.addSpacing(10)
        tips = {"erf": "error-function step: Gaussian roughness, smooth on both sides",
                "lin": "straight ramp: a uniform mixing zone of width √12·σ",
                "exp": "exponential step: steep in the middle with long tails"}
        self.profile_edits = []
        for name in PROFILE_NAMES:
            rough_row.addWidget(QLabel(name))
            entry = QLineEdit()
            entry.setMaximumWidth(42)
            entry.setToolTip(f"Weight 0 … 1 of the {tips[name]}. The boundary is the weighted mix of the three "
                             f"steps (all with the same σ); the weights are divided by their sum.")
            rough_row.addWidget(entry)
            self.profile_edits.append(entry)
        self.lbl_mix = QLabel("")
        self.lbl_mix.setObjectName("Formula")
        rough_row.addWidget(self.lbl_mix)
        rough_row.addStretch(1)
        outer.addLayout(rough_row)

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

        user_page = QWidget()                                # the thickness in every period, e.g. from a file
        uv = QVBoxLayout(user_page)
        uv.setContentsMargins(0, 0, 0, 0)
        uv.addWidget(QLabel("Thickness in each period (Å), top period first:"))
        self.ed_user = QLineEdit()
        self.ed_user.setToolTip("Values separated by spaces or ';'. With N equal to their number they are used as they are, "
                                "for another N they are interpolated.")
        uv.addWidget(self.ed_user)
        self.user_drop = DropZone("Drop an .xlsx with this material's thicknesses (nm)")
        self.user_drop.setMinimumHeight(40)
        self.user_drop.file_dropped.connect(lambda path: self.file_dropped.emit(self, path))
        uv.addWidget(self.user_drop)
        self.lbl_user = QLabel("")
        self.lbl_user.setObjectName("Formula")
        self.lbl_user.setWordWrap(True)
        uv.addWidget(self.lbl_user)
        uv.addStretch(1)
        self.pages.addWidget(user_page)
        body.addWidget(self.pages, 1)

        self.plot = MiniPlot()                               # thickness in every period
        body.addWidget(self.plot)
        outer.addLayout(body)

        self.lbl_warn = QLabel("")
        self.lbl_warn.setStyleSheet(f"color: {colors['warning']}; font-weight: bold; background: transparent;")
        outer.addWidget(self.lbl_warn)

        # ---- connections ----
        for entry in [self.ed_pot, self.ed_eps_re, self.ed_eps_im, self.ed_sigma,
                      self.ed_start, self.ed_end, self.ed_user] + self.profile_edits + self.coef_edits:
            entry.textChanged.connect(self.read_entries)
        self.cmb_name.editTextChanged.connect(self.read_entries)
        self.cmb_name.activated.connect(self.on_library_choice)
        self.cmb_mode.currentIndexChanged.connect(self.on_mode_changed)
        self.spin_deg.valueChanged.connect(self.on_degree_changed)
        self.btn_remove.clicked.connect(lambda: self.remove_requested.emit(self))

        self.show_material()

    def set_color(self, color):
        """New line color (after a theme change) and redraw."""
        self.color = color
        self.swatch.setStyleSheet(f"color: {color}; font-size: 13pt; background: transparent;")
        self.lbl_warn.setStyleSheet(f"color: {colors['warning']}; font-weight: bold; background: transparent;")
        self.draw_graph()

    def set_library(self, library):
        """Fill the name list with the materials of the library, keeping the name shown."""
        self.library = library
        shown = self.cmb_name.currentText()
        self.cmb_name.blockSignals(True)
        self.cmb_name.clear()
        self.cmb_name.addItems(list(library))
        self.cmb_name.setEditText(shown)
        self.cmb_name.blockSignals(False)

    def on_library_choice(self, index):
        """A library material was picked from the list: take its name, V and ε (σ and thicknesses stay)."""
        name = self.cmb_name.itemText(index)
        if name not in self.library:
            return
        self.material.name = name
        self.material.potential, self.material.epsilon = self.library[name]
        self.show_material()
        self.changed.emit()

    # ---- entries -> material ----
    def read_entries(self, *_):
        if self._showing:
            return
        mat = self.material
        mat.name = self.cmb_name.currentText().strip()
        mat.potential = safe_float(self.ed_pot.text(), mat.potential)
        mat.epsilon = complex(safe_float(self.ed_eps_re.text(), mat.epsilon.real),
                              safe_float(self.ed_eps_im.text(), mat.epsilon.imag))
        mat.sigma = max(safe_float(self.ed_sigma.text(), mat.sigma), 0.0)
        for k, entry in enumerate(self.profile_edits):
            mat.profile[k] = min(max(safe_float(entry.text(), mat.profile[k]), 0.0), 1.0)
        self.lbl_mix.setText("= " + mix_text(mat.profile))
        mat.linear[0] = safe_float(self.ed_start.text(), mat.linear[0])
        mat.linear[1] = safe_float(self.ed_end.text(), mat.linear[1])
        for k in range(len(mat.poly)):
            mat.poly[k] = safe_float(self.coef_edits[k].text(), mat.poly[k])
        user = [safe_float(x, None) for x in self.ed_user.text().replace(";", " ").split()]
        if user and None not in user:
            mat.user = np.array(user)
        self.draw_graph()
        self.changed.emit()

    def on_mode_changed(self, index):
        if self._showing:
            return
        if index == 1:
            self.material.switch_to_poly()        # polynomial starts from the current thicknesses
        elif index == 2:
            self.material.switch_to_user(self.n_repeats)   # keeps a loaded array, else the current thicknesses
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
        self.cmb_name.setEditText(mat.name)
        self.ed_pot.setText(f"{mat.potential:.4f}")
        self.ed_eps_re.setText(f"{mat.epsilon.real:.9f}")
        self.ed_eps_im.setText(f"{mat.epsilon.imag:.4e}")
        self.ed_sigma.setText(f"{mat.sigma:g}")
        for weight, entry in zip(mat.profile, self.profile_edits):
            entry.setText(f"{weight:g}")
        self.lbl_mix.setText("= " + mix_text(mat.profile))
        self.ed_start.setText(f"{mat.linear[0]:.2f}")
        self.ed_end.setText(f"{mat.linear[1]:.2f}")
        page = {"linear": 0, "poly": 1, "user": 2}[mat.mode]
        self.cmb_mode.setCurrentIndex(page)
        self.pages.setCurrentIndex(page)
        self.ed_user.setText(" ".join(f"{d:g}" for d in mat.user))
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
        """The thickness array used in the stack: this material's thickness in every period, top period first."""
        N = max(self.n_repeats, 1)
        period = np.arange(1, N + 1)
        d = self.material.thicknesses(N)

        n_user = self.material.user.size
        if self.material.mode == "user":
            self.lbl_user.setText(f"{n_user} values" + (" — used as they are." if n_user == N else
                                  f" — N = {N}, so they are interpolated (set N = {n_user} to use them as they are)."))
        negative = int(np.sum(d < 0))
        missing = int(np.sum(d == 0))
        if negative:
            self.lbl_warn.setText(f"⚠ d < 0 in {negative} of {N} periods — those layers are left out")
        self.lbl_warn.setVisible(negative > 0)
        title = f"{self.material.name or '—'}: {d.min():.1f} … {d.max():.1f} Å"
        if missing:
            title += f"  ({missing} × 0)"
        self.plot.draw(period, d, self.color, title)


# ==============================================================================
# Every adjustable number of the program: shown and edited in the app
# ==============================================================================
# decimals = 0 gives an integer entry, a tuple as start value a choice list. The group decides where it appears.
#  key              label                          start           min      max       step     dec  group       tooltip
PARAMETERS = [
    ("V_vac",          "V_vac (µeV)",                0.0,            -2.0,    5.0,      0.01,    3,   "structure", "Potential of the medium above the mirror"),
    ("V_sub",          "V_sub (µeV)",                0.090,          -2.0,    5.0,      0.01,    3,   "structure", "Potential of the substrate"),
    ("eps_vac_re",     "ε_vac real",                 1.0,            0.5,     1.5,      1e-6,    9,   "structure", "X-ray permittivity of the medium above the mirror (real part)"),
    ("eps_sub_re",     "ε_sub real",                 EPS_SIO2.real,  0.5,     1.5,      1e-6,    9,   "structure", "X-ray permittivity of the substrate (real part); default: fused silica at Cu Kα"),
    ("eps_vac_im",     "ε_vac imag",                 0.0,            0.0,     0.1,      1e-7,    10,  "structure", "X-ray permittivity of the medium above the mirror (imaginary part)"),
    ("eps_sub_im",     "ε_sub imag",                 EPS_SIO2.imag,  0.0,     0.1,      1e-7,    10,  "structure", "X-ray permittivity of the substrate (imaginary part)"),
    ("N",              "Repetitions N",              30,             1,       20000,    1,       0,   "structure", "How many times the unit cell (all materials) is repeated"),

    ("sigma_sub",      "Substrate σ (Å)",            3.0,            0.0,     1000.0,   0.5,     2,   "interface", "rms roughness of the substrate surface (boundary between the last layer and the substrate); 0 = sharp"),
    ("sub_erf",        "Substrate erf",              1.0,            0.0,     1.0,      0.1,     2,   "interface", "Weight 0 … 1 of the error-function step in the substrate boundary"),
    ("sub_lin",        "Substrate lin",              0.0,            0.0,     1.0,      0.1,     2,   "interface", "Weight 0 … 1 of the linear ramp in the substrate boundary"),
    ("sub_exp",        "Substrate exp",              0.0,            0.0,     1.0,      0.1,     2,   "interface", "Weight 0 … 1 of the exponential step in the substrate boundary"),
    ("rough_method",   "Roughness method",           ("factor (fast, as Multifitting)", "slices (graded profile, slow)"), None, None, None, None, "interface",
     "factor: sharp layers, the reflection of every boundary × the Fourier transform of its roughness profile — one step per layer, "
     "same as Multifitting, good while σ is well below the layer thicknesses. slices: the blurred profile itself cut into thin slices — "
     "exact also for overlapping transitions / intermixing, but tens of times slower"),
    ("slice_d",        "Slice thickness d (Å)",      2.0,            0.05,    100.0,    0.5,     2,   "interface", "The smooth transitions are cut into slices of at most d. Equal slices act as a grating with a false peak at q = 2π/d, so the largest d accepted is π/(4·q max) of the plots (the peak then lies beyond 8·q max)"),
    ("chaos",          "Chaos 0…1",                  0.0,            0.0,     1.0,      0.05,    2,   "interface", "Every boundary gets its own width σ·(1 + chaos·u), u random in [−1, 1]"),
    ("seed",           "Random seed",                1,              0,       999999,   1,       0,   "interface", "Same seed = same random structure (reproducible curves); change it for another realization"),

    ("file_order",     "First row of the sheet is",  ("top (ambient side)", "bottom (substrate side)"), None, None, None, None, "file", "Order of the rows in the structure file"),
    ("file_sigma",     "σ upper of loaded layers (Å)", 3.0,          0.0,     1000.0,   0.5,     2,   "file",      "Roughness given to every layer read from the file (each card can be changed afterwards)"),

    ("energy",         "Energy E (meV)",             energy_from_k(2 * np.pi / 5.0), 1e-4, 1e5, 0.1, 4, "beam", "Neutron energy — editing it updates k"),
    ("k",              "Momentum k = p/ħ (Å⁻¹)",     2 * np.pi / 5.0, 1e-3,   100.0,    0.01,    5,   "beam",      "Neutron wave vector — editing it updates E"),

    ("E_x",            "X-ray energy E (keV)",       xray_energy(1.540562), 0.1, 500.0, 0.01,   5,   "xray",      "Photon energy — editing it updates λ. The materials' ε must belong to this energy"),
    ("lambda_x",       "X-ray wavelength λ (Å)",     1.540562,       0.01,    200.0,    0.001,   6,   "xray",      "Photon wavelength — editing it updates E (default: Cu Kα1)"),
    ("polarization",   "Polarization",               ("s", "p"),     None,    None,     None,    None, "xray",     "s: E field parallel to the surface · p: in the plane of incidence"),
    ("xq_max",         "X-ray q max (Å⁻¹)",          0.25,           0.01,    10.0,     0.01,    3,   "xray",      "Highest q of the X-ray curve"),
    ("xq_points",      "X-ray q points",             3000,           100,     100000,   100,     0,   "xray",      "Number of q values of the X-ray curve"),

    ("q_min",          "q min (Å⁻¹)",                0.001,          1e-4,    0.1,      0.001,   4,   "plot",      "Lowest q of the calculation grid"),
    ("q_max_factor",   "q max / (m·q_c)",            1.3,            1.0,     50.0,     0.1,     2,   "plot",      "Highest q of the grid, as a multiple of the target edge m·q_c"),
    ("q_points",       "q points",                   800,            50,      20000,    50,      0,   "plot",      "Number of q values (plots and optimizer use the same grid)"),
    ("edge_fraction",  "Edge window from (× m·q_c)", 0.95,           0.5,     0.999,    0.01,    3,   "plot",      "'R near m·q_c' = mean R between this fraction of m·q_c and m·q_c"),
    ("R_scale",        "R axis",                     ("log", "linear"), None, None,   None,    None, "plot",      "Reflectivity axis: logarithmic or linear"),
    ("log_decades",    "Log range (decades)",        9,              1,       30,       1,       0,   "plot",      "Log axis: R is shown from its maximum down to its minimum, but at most this many decades"),
    ("z_pad",          "Vacuum / substrate shown (Å)", 50.0,         0.0,     10000.0,  10.0,    0,   "plot",      "Width of vacuum and substrate drawn in the V(z) plot"),
    ("theme",          "Theme",                      ("light", "dark"), None, None,   None,    None, "plot",      "Colors of the whole program and the plots; dark uses the colors of the program icon"),

    ("xray_unit",      "X-ray file: column 1 is",    ("θ (°)", "2θ (°)", "q (Å⁻¹)", "q (nm⁻¹)"), None, None, None, None, "data", "Meaning of the first column; θ and 2θ are turned into q with the X-ray wavelength λ"),
    ("neutron_unit",   "Neutron file: column 1 is",  ("θ (°)", "2θ (°)", "q (Å⁻¹)", "q (nm⁻¹)"), None, None, None, None, "data", "Meaning of the first column; θ and 2θ are turned into q with the neutron k"),
    ("xray_factor",    "X-ray points × factor",      1.0,            1e-9,    1e9,      0.1,     4,   "data",      "The measured intensities are multiplied by this (after the normalization)"),
    ("neutron_factor", "Neutron points × factor",    1.0,            1e-9,    1e9,      0.1,     4,   "data",      "The measured intensities are multiplied by this (after the normalization)"),
    ("xray_background",    "X-ray background",       0.0,            0.0,     1.0,      1e-6,    10,  "data",      "Added to the calculated X-ray R in the plot only, as Multifitting's 'Background': the detector cannot see less than about one count. ⟨R⟩ and the optimizer use R without it"),
    ("neutron_background", "Neutron background",     0.0,            0.0,     1.0,      1e-6,    10,  "data",      "Added to the calculated neutron R in the plot only (as Multifitting's 'Background'). ⟨R⟩ and the optimizer use R without it"),
    ("data_norm",      "Normalize measured",         ("max = 1", "as it is"), None, None, None, None, "data", "max = 1: counts are divided by their maximum, so they compare with R; as it is: the numbers of the file"),

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
    "interface": "Interfaces (σ and shape of each boundary belong to the medium below it: the cards, the substrate here)",
    "file":      "Structure from a file (.xlsx: column A = material, column B = thickness in nm)",
    "beam":      "Neutron beam",
    "xray":      "X-ray beam",
    "plot":      "q grid & plots",
    "data":      "Measured curves to compare (.xy as Multifitting writes it: column 1 = θ, 2θ or q, column 2 = intensity)",
    "target":    "Supermirror target",
    "search":    "Search",
    "advanced":  "Advanced optimizer settings",
}
REDRAW_GROUPS = ("structure", "interface", "beam", "xray", "plot", "data", "target")     # changing these recalculates the plots


# ==============================================================================
# Main window
# ==============================================================================
class MultilayerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neutron Multilayer Reflectivity Designer")
        self.setWindowIcon(program_icon())
        self.resize(1650, 900)

        self.materials = []                                  # list of Material — the structure
        self.cards = []                                      # one LayerCard per material (same order)
        self.params = {}                                     # key -> entry widget, for every PARAMETERS row
        self.reference = [row[:] for row in M_REFERENCE]     # editable copy of the reference table
        self.m = 2.0                                         # supermirror m
        self.running = False                                 # True while the optimizer loop runs
        self.mean_R = 0.0                                    # ⟨R⟩ of the structure shown in the plot
        self.calc_seconds = 0.0                              # duration of the last plot update
        self.measured = {"xray": None, "neutron": None}      # measured curves: (file name, x[], y[]) or None
        self.native_style = QApplication.style().objectName() or "Fusion"   # Qt style of the light theme (the system one)

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
        self.library = {}                                    # material library: {name: (potential, ε)}
        self.file_path = None                                # structure file currently loaded
        self.sheets = []                                     # its sheet names
        self.pages = [self.build_structure_tab(), self.build_file_tab(), self.build_beam_tab(),
                      self.build_optimizer_tab(), self.build_reference_tab()]
        for page, title in zip(self.pages, ["Structure", "File && library", "Beams && plots", "Optimizer",
                                            "Reference table"]):
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
        self.plots = PlotPanel()                             # neutron R(q), X-ray R(q), V(z), ε(z)
        right_layout.addWidget(self.plots)
        hint = QLabel("Plots: drag = move · wheel = zoom (over an axis: that axis only) · "
                      "right-click = View All / Export (PNG, SVG, CSV) · 'A' corner button = full view again")
        hint.setObjectName("Formula")
        right_layout.addWidget(hint)
        splitter.addWidget(right)
        splitter.setSizes([580, 1070])

        # ---- connections of the parameter entries (special handlers first, then the redraw) ----
        self.params["N"].valueChanged.connect(self.on_repeats_changed)
        self.params["m"].valueChanged.connect(self.on_m_changed)
        self.params["energy"].valueChanged.connect(self.on_energy_changed)
        self.params["k"].valueChanged.connect(self.on_k_changed)
        self.params["E_x"].valueChanged.connect(self.on_xray_energy_changed)
        self.params["lambda_x"].valueChanged.connect(self.on_xray_wavelength_changed)
        for key in ("N_start_factor", "N_max_factor"):
            self.params[key].valueChanged.connect(lambda *_: self.on_m_changed(self.v("m")))
        self.params["file_order"].currentIndexChanged.connect(lambda *_: self.load_sheet())
        self.params["file_sigma"].valueChanged.connect(lambda *_: self.load_sheet())
        self.params["theme"].currentIndexChanged.connect(lambda *_: self.apply_theme())
        self.setAcceptDrops(True)                            # an .xlsx can be dropped anywhere on the window
        self.reload_library()
        for key, label, start, lo, hi, step, dec, group, tip in PARAMETERS:
            if group in REDRAW_GROUPS:
                widget = self.params[key]
                changed = widget.currentIndexChanged if isinstance(widget, QComboBox) else widget.valueChanged
                changed.connect(self.schedule_update)

        # default structure: Ni / Ti, linear thickness laws
        self.add_material(Material("Ni", 0.2410, linear=(40.0, 200.0), epsilon=EPS_NI, sigma=3.0))
        self.add_material(Material("Ti", -0.0490, linear=(40.0, 200.0), epsilon=EPS_TI, sigma=3.0))
        self.on_m_changed(self.v("m"))
        self.apply_theme()
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
        box, _ = self.build_group("structure")
        layout.addWidget(box)
        box, grid = self.build_group("interface")
        self.lbl_slices = QLabel("")
        self.lbl_slices.setObjectName("Formula")
        self.lbl_slices.setWordWrap(True)
        self.add_full_row(grid, self.lbl_slices)
        layout.addWidget(box)
        layout.addWidget(self.build_materials_box(), 1)
        return page

    def build_file_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        box, grid = self.build_group("file")
        self.drop_zone = DropZone("Drop an .xlsx structure file here (or anywhere on the window)")
        self.drop_zone.file_dropped.connect(self.open_structure_file)
        self.add_full_row(grid, self.drop_zone)
        row = QHBoxLayout()
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(lambda: self.browse_structure_file())
        row.addWidget(btn_browse)
        row.addWidget(QLabel("Sheet:"))
        self.cmb_sheet = QComboBox()
        self.cmb_sheet.currentIndexChanged.connect(lambda *_: self.load_sheet())
        row.addWidget(self.cmb_sheet, 1)
        grid.addLayout(row, grid.rowCount(), 0, 1, 4)
        self.lbl_file = QLabel("No file loaded. The rows become the distinct materials in table order, each with its "
                               "thickness in every period (neighbouring equal rows are merged); N = number of periods.")
        self.lbl_file.setObjectName("Formula")
        self.lbl_file.setWordWrap(True)
        self.add_full_row(grid, self.lbl_file)
        layout.addWidget(box)

        box = QGroupBox(f"Material library — one text file per material in {LIBRARY_DIR.name}/ "
                        f"(row 1: V in µeV, row 2: ε real imaginary)")
        lib = QVBoxLayout(box)
        self.lib_table = QTableWidget(0, 4)
        self.lib_table.setHorizontalHeaderLabels(["Material", "V (µeV)", "ε real", "ε imag"])
        self.lib_table.horizontalHeader().setSectionResizeMode(qenum(QHeaderView, 'ResizeMode', 'Stretch'))
        self.lib_table.setEditTriggers(qenum(QAbstractItemView, 'EditTrigger', 'NoEditTriggers'))
        lib.addWidget(self.lib_table, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Ambient:"))
        self.cmb_ambient = QComboBox()
        self.cmb_ambient.currentIndexChanged.connect(lambda *_: self.use_library_medium("ambient"))
        row.addWidget(self.cmb_ambient, 1)
        row.addWidget(QLabel("Substrate:"))
        self.cmb_substrate = QComboBox()
        self.cmb_substrate.currentIndexChanged.connect(lambda *_: self.use_library_medium("substrate"))
        row.addWidget(self.cmb_substrate, 1)
        btn_reload = QPushButton("Reload library")
        btn_reload.clicked.connect(lambda: self.reload_library())
        row.addWidget(btn_reload)
        lib.addLayout(row)
        layout.addWidget(box, 1)
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
        box, grid = self.build_group("xray")
        self.lbl_xray = QLabel("")
        self.lbl_xray.setObjectName("Formula")
        self.lbl_xray.setWordWrap(True)
        self.add_full_row(grid, self.lbl_xray)
        layout.addWidget(box)
        box, _ = self.build_group("plot")
        layout.addWidget(box)
        box, grid = self.build_group("data")
        columns = QHBoxLayout()
        self.data_labels = {}
        for which, title in (("xray", "X-ray"), ("neutron", "neutron")):
            column = QVBoxLayout()
            zone = DropZone(f"Drop the measured {title} curve here (.xy)", CURVE_FILES)
            zone.setMinimumHeight(55)
            zone.file_dropped.connect(lambda path, w=which: self.load_measured(w, path))
            column.addWidget(zone)
            buttons = QHBoxLayout()
            for text, action, tip in (("Browse…", self.browse_measured, "Choose the file"),
                                      ("Fit q range", self.fit_q_to_measured, "Extend the q axis of this plot to the last measured point"),
                                      ("1-count background", self.background_from_measured,
                                       "Set this plot's background to the smallest measured value above 0 (one detector count)"),
                                      ("Clear", self.clear_measured, "Remove the measured points from the plot")):
                button = QPushButton(text)
                button.setToolTip(tip)
                button.clicked.connect(lambda _=False, a=action, w=which: a(w))
                buttons.addWidget(button)
            column.addLayout(buttons)
            label = QLabel("No file.")
            label.setObjectName("Formula")
            label.setWordWrap(True)
            column.addWidget(label)
            self.data_labels[which] = label
            columns.addLayout(column)
        grid.addLayout(columns, grid.rowCount(), 0, 1, 4)
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
        btn_add.clicked.connect(lambda: self.add_material(Material("Layer", 0.100, linear=(50.0, 50.0), sigma=3.0)))
        btn_del = QPushButton("- Remove Last Material")
        btn_del.clicked.connect(lambda: self.remove_card(self.cards[-1]) if self.cards else None)
        buttons.addWidget(btn_add)
        buttons.addWidget(btn_del)
        layout.addLayout(buttons)
        return box

    # ------------------------------------------------------------------ materials
    def add_material(self, material):
        card = LayerCard(material, self.v("N"), colors["layers"][len(self.cards) % len(colors["layers"])],
                         self.library)
        card.changed.connect(self.schedule_update)
        card.remove_requested.connect(self.remove_card)
        card.file_dropped.connect(self.load_card_file)
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

    def replace_structure(self, materials, N):
        """Remove all material cards and show the given materials instead, with N repeats."""
        for card in list(self.cards):
            self.remove_card(card)
        self.set_v("N", N)
        for material in materials:
            self.add_material(material)

    # ------------------------------------------------------------------ material library
    def reload_library(self):
        """Read materials/*.txt into self.library and show it in the table and the two pickers."""
        try:
            self.library = load_library()
        except Exception as error:                       # a malformed file: report, keep the old library
            self.lbl_file.setText(f"⚠ Material library could not be read: {error}")
            return
        self.lib_table.setRowCount(0)
        for name, (potential, epsilon) in self.library.items():
            r = self.lib_table.rowCount()
            self.lib_table.insertRow(r)
            for c, text in enumerate([name, f"{potential:.4f}", f"{epsilon.real:.9f}", f"{epsilon.imag:.4e}"]):
                self.lib_table.setItem(r, c, QTableWidgetItem(text))
        for combo in (self.cmb_ambient, self.cmb_substrate):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(["— keep the entries —"] + list(self.library))
            combo.blockSignals(False)
        for card in self.cards:                              # the name lists of the material cards
            card.set_library(self.library)

    def use_library_medium(self, which):
        """Fill V and ε of the ambient or the substrate (Structure tab) from the chosen library material."""
        combo = self.cmb_ambient if which == "ambient" else self.cmb_substrate
        if combo.currentIndex() <= 0:
            return
        potential, epsilon = self.library[combo.currentText()]
        prefix = ("V_vac", "eps_vac") if which == "ambient" else ("V_sub", "eps_sub")
        self.params[prefix[0]].setValue(potential)
        self.params[prefix[1] + "_re"].setValue(epsilon.real)
        self.params[prefix[1] + "_im"].setValue(epsilon.imag)

    # ------------------------------------------------------------------ structure file
    def browse_structure_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Structure file", "", "Excel files (*.xlsx)")
        if path:
            self.open_structure_file(path)

    def open_structure_file(self, path):
        """List the sheets of a dropped / chosen .xlsx file and load the first sheet that has layers."""
        try:
            sheets = sheet_names(path)
        except ImportError:
            self.lbl_file.setText("⚠ Reading .xlsx files needs the openpyxl package:  pip install openpyxl")
            return
        except Exception as error:
            self.lbl_file.setText(f"⚠ Cannot read {path}: {error}")
            return
        self.file_path = path
        self.sheets = [name for name, _ in sheets]
        self.cmb_sheet.blockSignals(True)
        self.cmb_sheet.clear()
        self.cmb_sheet.addItems([f"{name}   ({n} layers)" for name, n in sheets])
        first_with_layers = next((i for i, (_, n) in enumerate(sheets) if n > 0), 0)
        self.cmb_sheet.setCurrentIndex(first_with_layers)
        self.cmb_sheet.blockSignals(False)
        self.load_sheet()

    def load_sheet(self):
        """Replace the current structure by the chosen sheet, converted to a periodic one: the distinct materials
        in table order, each with its thickness in every period (user mode), N = number of periods."""
        if not self.file_path or not self.sheets or self.running:
            return
        sheet = self.sheets[self.cmb_sheet.currentIndex()]
        file_name = self.file_path.replace("\\", "/").split("/")[-1]
        rows = read_layers(self.file_path, sheet)
        if not rows:
            self.lbl_file.setText(f"⚠ Sheet '{sheet}' of {file_name} has no rows with a material and a thickness.")
            return
        try:
            materials, N, merged = periodic_materials(rows, self.library, self.v("file_sigma"),
                                                      first_row_is_top=self.v("file_order").startswith("top"))
        except KeyError as missing:
            self.lbl_file.setText(f"⚠ No library file for: {missing.args[0]}. Add materials/<name>.txt "
                                  f"(row 1: V in µeV, row 2: ε real imaginary) and press Reload library.")
            return
        self.replace_structure(materials, N)
        zeros = sum(int(np.sum(m.user == 0)) for m in materials)
        self.lbl_file.setText(
            f"Loaded sheet '{sheet}' of {file_name}: {len(rows)} rows ({sum(d for _, d in rows):.1f} nm) → "
            f"materials [{', '.join(m.name for m in materials)}] × N = {N} periods"
            + (f", {merged} neighbouring equal rows merged" if merged else "")
            + (f", {zeros} missing layers set to 0" if zeros else "")
            + f". First row = {self.v('file_order')}.")
        self.tabs.setCurrentIndex(0)                         # show the material cards

    def load_card_file(self, card, path):
        """A file dropped on a card in user mode: that material's thickness in every period; N follows."""
        file_name = path.replace("\\", "/").split("/")[-1]
        try:
            values = read_thickness_file(path, card.material.name, self.v("file_order").startswith("top"))
        except ImportError:
            card.lbl_user.setText("⚠ Reading .xlsx files needs the openpyxl package:  pip install openpyxl")
            return
        except (KeyError, ValueError) as problem:
            card.lbl_user.setText(f"⚠ {file_name}: {problem.args[0]}")
            return
        card.material.user = 10.0 * np.array(values)        # nm -> Å
        card.material.mode = "user"
        self.params["N"].setValue(len(values))               # the number of periods follows the file
        card.show_material()
        card.lbl_user.setText(f"{len(values)} values from {file_name} (top period first).")
        self.schedule_update()

    def dragEnterEvent(self, event):
        if dropped_path(event):
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = dropped_path(event)
        if path:
            event.acceptProposedAction()
            self.open_structure_file(path)

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

    def on_xray_energy_changed(self, E):
        self.set_v("lambda_x", xray_wavelength(E))

    def on_xray_wavelength_changed(self, wavelength):
        self.set_v("E_x", xray_energy(wavelength))

    # ------------------------------------------------------------------ measured curves (.xy)
    def load_measured(self, which, path):
        """Read a measured curve for the X-ray ('xray') or the neutron ('neutron') plot."""
        file_name = path.replace("\\", "/").split("/")[-1]
        try:
            x, y = read_xy(path)
        except (OSError, ValueError) as problem:
            self.data_labels[which].setText(f"⚠ {file_name}: {problem}")
            return
        self.measured[which] = (file_name, x, y)
        self.schedule_update()

    def browse_measured(self, which):
        path, _ = QFileDialog.getOpenFileName(self, "Measured curve", "",
                                              "Curves (*.xy *.dat *.txt *.csv);;All files (*)")
        if path:
            self.load_measured(which, path)

    def clear_measured(self, which):
        self.measured[which] = None
        self.data_labels[which].setText("No file.")
        self.schedule_update()

    def measured_curve(self, which):
        """The measured curve as (q[], R[]) on the axes of the plot, or None.
        Column 1 -> q:  q = 2k·sin θ  (θ = column 1, or column 1 / 2 for 2θ),  or q itself.
        Column 2 -> R:  divided by its maximum ("max = 1") and multiplied by the factor."""
        if self.measured[which] is None:
            return None
        _, x, y = self.measured[which]
        k = 2 * np.pi / self.v("lambda_x") if which == "xray" else self.v("k")
        unit = self.v(f"{which}_unit")
        if unit == "θ (°)":
            q = q_from_angle(x, k)
        elif unit == "2θ (°)":
            q = q_from_angle(x / 2.0, k)
        elif unit == "q (Å⁻¹)":
            q = x
        else:                                               # q in nm⁻¹
            q = x / 10.0
        R = y / y.max() if self.v("data_norm") == "max = 1" and y.max() > 0 else y
        return q, R * self.v(f"{which}_factor")

    def fit_q_to_measured(self, which):
        """Extend (or shrink) the q axis of this plot to the last measured point."""
        curve = self.measured_curve(which)
        if curve is None:
            return
        q_last = float(curve[0].max())
        if which == "xray":
            self.params["xq_max"].setValue(np.ceil(q_last * 1000) / 1000)          # rounded up: keeps the last point
        else:
            _, q_top, _ = self.target_edge()
            self.params["q_max_factor"].setValue(np.ceil(q_last / q_top * 100) / 100)

    def background_from_measured(self, which):
        """Background = the smallest measured value above 0, i.e. one detector count on the plot's scale."""
        curve = self.measured_curve(which)
        if curve is None or not np.any(curve[1] > 0):
            return
        self.params[f"{which}_background"].setValue(float(curve[1][curve[1] > 0].min()))

    def show_measured_info(self, which, curve, q_end):
        """Status line under a drop zone: points, range in the file and in q."""
        if curve is None:
            return
        name, x, _ = self.measured[which]
        q, R = curve
        text = (f"{name}: {x.size} points, column 1 = {x.min():g} … {x.max():g} {self.v(which + '_unit')}"
                f"  →  q = {q.min():.4f} … {q.max():.4f} Å⁻¹, R up to {R.max():.3g}")
        if q.max() > 1.001 * q_end:
            text += f"\n⚠ The plot ends at q = {q_end:.4f} Å⁻¹ — press 'Fit q range' to see all points."
        self.data_labels[which].setText(text)

    # ------------------------------------------------------------------ theme
    def apply_theme(self):
        """Light or dark: Qt stylesheet + palette, plot colors, material card colors."""
        name = self.v("theme")
        colors.clear()
        colors.update(THEMES[name])
        app = QApplication.instance()
        if colors["palette"]:                               # dark: Fusion style + dark palette
            app.setStyle("Fusion")
            palette = QPalette()
            for role, color in colors["palette"].items():
                palette.setColor(qenum(QPalette, 'ColorRole', role), QColor(color))
            disabled = qenum(QPalette, 'ColorGroup', 'Disabled')
            for role in ("WindowText", "Text", "ButtonText"):
                palette.setColor(disabled, qenum(QPalette, 'ColorRole', role), QColor(colors["disabled_text"]))
            app.setPalette(palette)
        else:                                               # light: the system style and palette
            app.setStyle(self.native_style)
            app.setPalette(app.style().standardPalette())
        app.setStyleSheet(colors["style"])

        self.plots.apply_theme()
        for i, card in enumerate(self.cards):
            card.set_color(colors["layers"][i % len(colors["layers"])])
        self.schedule_update()

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
        method = "slices" if self.v("rough_method").startswith("slices") else "factor"
        return Interface(method=method, slice_thickness=self.v("slice_d"), chaos=self.v("chaos"), seed=self.v("seed"))

    def ambient(self):
        """The medium above the stack as a Material (its σ is not used: the top layer's σ is the surface)."""
        return Material("ambient", self.v("V_vac"),
                        epsilon=complex(self.v("eps_vac_re"), self.v("eps_vac_im")))

    def substrate(self):
        """The substrate as a Material; its σ and profile describe the last boundary."""
        return Material("substrate", self.v("V_sub"),
                        epsilon=complex(self.v("eps_sub_re"), self.v("eps_sub_im")), sigma=self.v("sigma_sub"),
                        profile=(self.v("sub_erf"), self.v("sub_lin"), self.v("sub_exp")))

    def limit_slice_thickness(self, q_highest):
        """
        The largest slice thickness the entry accepts: d_max = π/(4·q_highest).
        Equal slices act as a grating with a false Bragg peak at q = 2π/d; with d ≤ d_max that peak lies
        beyond 8·q_highest, far outside the plots. A larger d in the entry is lowered to d_max.
        """
        d_max = max(np.floor(np.pi / (4.0 * q_highest) * 100.0) / 100.0, 0.05)    # rounded down to the entry's 0.01 Å
        entry = self.params["slice_d"]
        entry.blockSignals(True)                             # this calculation already uses the new value
        entry.setMaximum(d_max)
        entry.blockSignals(False)
        return d_max

    def slicing_summary(self, interface, d_max, q_highest):
        """Text: how many slices every kind of boundary gets (before chaos), and the largest d allowed."""
        if interface.method == "factor":
            text = ("Roughness by factors (the idea of Multifitting): sharp layers; the reflection and transmission of "
                    "every boundary are multiplied by factors made from the Fourier transform of its profile "
                    "(erf: r × exp(−2·kz₁kz₂·σ²), t·t′ × exp((kz₁ − kz₂)²σ²); lin and exp alike). No slices — the slice "
                    "thickness is not used. Good while σ is well below the layer thicknesses; for overlapping "
                    "transitions or intermixing use 'slices'.")
            if interface.chaos > 0:
                text += f"  Chaos: every boundary's σ varies by ±{100 * interface.chaos:.0f} %."
            return text
        media = [(f"{m.name or '—'} upper", m) for m in self.materials] + [("substrate", self.substrate())]
        parts = [f"{name} (σ {m.sigma:g} Å, {mix_text(m.profile)}): {interface.n_slices(m.sigma, m.profile) or 'sharp'}"
                 for name, m in media]
        text = (f"Slices per boundary (erf ±4σ, lin ±√3σ, exp ±7σ — the widest step used — cut into slices "
                f"≤ {interface.slice_thickness:g} Å):  " + "  ·  ".join(parts)
                + ".  The flat middle of every layer is one more slice."
                + f"  Largest d accepted for q up to {q_highest:.3f} Å⁻¹: {d_max:g} Å.")
        if interface.chaos > 0:
            text += f"  Chaos: every boundary's σ varies by ±{100 * interface.chaos:.0f} %, so these numbers vary too."
        return text

    def q_values(self, q_top):
        return q_grid(q_top, self.v("q_min"), self.v("q_max_factor"), self.v("q_points"))

    # ------------------------------------------------------------------ calculation + plots
    def schedule_update(self, *_):
        self.calc_timer.start()

    def update_calculation(self):
        started = time.time()
        self.plots.clear_all()
        if not self.materials:
            return

        # ---- q grids of both plots; they decide the largest slice thickness ----
        N = self.v("N")
        target = self.v("target")
        q_c, q_top, qc_name = self.target_edge()
        q = self.q_values(q_top)
        q_x = np.linspace(self.v("q_min"), self.v("xq_max"), self.v("xq_points"))
        q_highest = max(q[-1], q_x[-1])
        d_max = self.limit_slice_thickness(q_highest)
        interface = self.interface()

        # ---- neutrons: METHOD 1 + METHOD 2.1 (the same calls the optimizer uses) ----
        k = self.v("k")
        ambient, substrate = self.ambient(), self.substrate()
        potentials, thicknesses, roughness = build_stack(self.materials, N, ambient, substrate, interface)
        R = neutron_reflectivity(potentials, thicknesses, q, roughness)
        self.mean_R = mean_reflectivity(R, q, q_top)
        near_edge = q >= self.v("edge_fraction") * q_top
        edge_R = mean_reflectivity(R[near_edge], q[near_edge], q_top)
        ok = self.mean_R >= target

        # ---- X-rays: METHOD 1 with ε + METHOD 2.2 — the same structure (same slices) ----
        wavelength = self.v("lambda_x")
        k_x = 2 * np.pi / wavelength
        epsilons, _, roughness_x = build_stack(self.materials, N, ambient, substrate, interface, quantity="epsilon")
        R_x = xray_reflectivity(epsilons, thicknesses, q_x, wavelength, self.v("polarization"), roughness_x)
        n_steps = len(potentials) - 2                        # layers or slices the reflectivity went through
        densest = max(self.materials, key=lambda mat: 1.0 - mat.epsilon.real)
        q_cx = xray_critical_q(densest.epsilon, wavelength)

        # ---- beam info lines ----
        theta_c, theta_top = glancing_angle(q_c, k), glancing_angle(q_top, k)
        self.lbl_beam.setText(
            f"λ = {2 * np.pi / k:.3f} Å   ·   v = {H_OVER_M * k / (2 * np.pi):.1f} m/s\n"
            f"θ_c {qc_name} = {theta_c:.3f}°   ·   θ at m·q_c = {theta_top:.3f}°")
        self.lbl_xray.setText(
            f"k = {k_x:.4f} Å⁻¹   ·   θ_c {densest.name} = {glancing_angle(q_cx, k_x):.3f}° "
            f"(q_c = {q_cx:.4f} Å⁻¹)   ·   θ at q max = {glancing_angle(q_x[-1], k_x):.2f}°")

        # ---- measured curves converted to q (None when no file is loaded) ----
        measured_n, measured_x = self.measured_curve("neutron"), self.measured_curve("xray")
        self.show_measured_info("neutron", measured_n, q[-1])
        self.show_measured_info("xray", measured_x, q_x[-1])

        # ---- neutron R(q) ----
        panel, log_scale, decades = self.plots, self.v("R_scale") == "log", self.v("log_decades")
        a = panel.neutron
        panel.draw_reflectivity(a, q, R, k, colors["neutron"], log_scale, decades, self.v("neutron_background"),
                                measured_n, self.measured["neutron"] and self.measured["neutron"][0])
        panel.add_range(a, 0.0, q_top, colors["range_ok"] if ok else colors["range_low"], colors["range_alpha"],
                        f"target range 0 … m·q<sub>c</sub> = {q_top:.4f} Å⁻¹")
        panel.add_vline(a, q_c, colors["q_c"], DASH, f"{qc_name} q<sub>c</sub> ({q_c:.4f} Å⁻¹)")
        panel.add_vline(a, q_top, colors["edge_line"], DOT, f"m·q<sub>c</sub> (m = {self.m:.2f})", width=1.4, alpha=1.0)
        panel.add_segment(a, 0.0, q_top, target, colors["target"], DASH, f"target ⟨R⟩ = {target:.3f}")
        n_layers = sum(int(np.sum(m.thicknesses(N) > 0)) for m in self.materials)     # layers really in the stack
        style_plot(a, colors["axes"],
                   f"Neutrons · E = {self.v('energy'):.3f} meV · N = {N} ({n_layers} layers)<br>"
                   f"⟨R⟩(0…m·q<sub>c</sub>) = {self.mean_R:.4f} {'✓' if ok else '✗'} (target {target:.3f})"
                   f"  ·  R near m·q<sub>c</sub> ≈ {edge_R:.3f}",
                   Q_LABEL, "Reflectivity <i>R</i>")
        panel.place_legend(a, "lower left" if log_scale else "upper right")

        # ---- X-ray R(q) ----
        a = panel.xray
        panel.draw_reflectivity(a, q_x, R_x, k_x, colors["xray"], log_scale, decades, self.v("xray_background"),
                                measured_x, self.measured["xray"] and self.measured["xray"][0])
        panel.add_vline(a, q_cx, colors["q_c"], DASH, f"{densest.name} q<sub>c</sub> ({q_cx:.4f} Å⁻¹)")
        style_plot(a, colors["axes"],
                   f"X-rays · λ = {wavelength:.4f} Å ({self.v('E_x'):.3f} keV), {self.v('polarization')}-polarization<br>"
                   f"same structure, ε from the material cards", Q_LABEL, "Reflectivity <i>R</i>")
        panel.place_legend(a, "upper right")

        # ---- depth profiles of the same structure: always the graded (rough) profile ----
        if interface.method == "factor":
            potentials, thicknesses, _ = build_stack(self.materials, N, ambient, substrate, interface, graded=True)
            epsilons, _, _ = build_stack(self.materials, N, ambient, substrate, interface, quantity="epsilon", graded=True)
        self.lbl_slices.setText(self.slicing_summary(interface, d_max, q_highest))
        panel.draw_profile(panel.potential, potentials, thicknesses, self.v("z_pad"), colors["potential"])
        how = (f"{n_steps} slices" if interface.method == "slices" else
               f"{n_steps} sharp layers × roughness factor; graded profile shown")
        style_plot(panel.potential, colors["axes"], f"Neutron potential <i>V</i>(z) ({how})",
                   Z_LABEL, "Potential <i>V</i> (µeV)")
        panel.draw_profile(panel.epsilon, (1.0 - epsilons.real) * 1e5, thicknesses, self.v("z_pad"), colors["epsilon"])
        style_plot(panel.epsilon, colors["axes"], "X-ray permittivity: 1 − Re ε(z)", Z_LABEL, "(1 − Re ε) × 10<sup>5</sup>")
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
                             self.ambient(), self.substrate(), self.interface(),
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
