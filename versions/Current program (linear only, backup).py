import sys
import numpy as np

# Universal PyQt5 / PyQt6 compatibility
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTableWidget, QTableWidgetItem, QPushButton, QLabel, QSpinBox,
        QDoubleSpinBox, QGroupBox, QHeaderView, QSplitter
    )
    from PyQt6.QtCore import Qt, QTimer
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTableWidget, QTableWidgetItem, QPushButton, QLabel, QSpinBox,
        QDoubleSpinBox, QGroupBox, QHeaderView, QSplitter
    )
    from PyQt5.QtCore import Qt, QTimer

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
# 1. Helper: Locale-safe float conversion
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


# ==============================================================================
# 2. Vectorized Parratt Calculation Engine
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
        thk_seq = np.linspace(lay['d_start'], lay['d_end'], n_repeats)[::-1]
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
# 3. Main GUI Application (Embedded Matplotlib)
# ==============================================================================
class MultilayerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neutron Multilayer Reflectivity Designer")
        self.resize(1280, 780)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal if hasattr(Qt, 'Orientation') else Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ----------------- Left Panel: Controls & Table -----------------
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
        self.spin_repeats.valueChanged.connect(self.update_calculation)
        h1.addWidget(self.spin_repeats)
        g_layout.addLayout(h1)

        # Sublayer Slices
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Interlayer Sublayers (n):"))
        self.spin_sublayers = QSpinBox()
        self.spin_sublayers.setRange(0, 50)
        self.spin_sublayers.setValue(5)
        self.spin_sublayers.valueChanged.connect(self.update_calculation)
        h2.addWidget(self.spin_sublayers)
        g_layout.addLayout(h2)

        # Sublayer Step Thickness
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Sublayer Step d (Å):"))
        self.spin_d_sub = QDoubleSpinBox()
        self.spin_d_sub.setRange(0.1, 20.0)
        self.spin_d_sub.setSingleStep(0.5)
        self.spin_d_sub.setValue(1.5)
        self.spin_d_sub.valueChanged.connect(self.update_calculation)
        h3.addWidget(self.spin_d_sub)
        g_layout.addLayout(h3)

        # Potentials
        h4 = QHBoxLayout()
        h4.addWidget(QLabel("V_sub (µeV):"))
        self.spin_vsub = QDoubleSpinBox()
        self.spin_vsub.setRange(-2.0, 5.0)
        self.spin_vsub.setDecimals(3)
        self.spin_vsub.setValue(0.090)
        self.spin_vsub.valueChanged.connect(self.update_calculation)
        h4.addWidget(self.spin_vsub)

        h4.addWidget(QLabel("V_vac (µeV):"))
        self.spin_vvac = QDoubleSpinBox()
        self.spin_vvac.setRange(-2.0, 5.0)
        self.spin_vvac.setDecimals(3)
        self.spin_vvac.setValue(0.0)
        self.spin_vvac.valueChanged.connect(self.update_calculation)
        h4.addWidget(self.spin_vvac)
        g_layout.addLayout(h4)

        left_layout.addWidget(global_box)

        # Unit Cell Layer Table
        table_box = QGroupBox("Periodic Unit Cell Layers")
        t_layout = QVBoxLayout(table_box)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Material", "d_start (Å)", "d_end (Å)", "V (µeV)"])
        header = self.table.horizontalHeader()
        try:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        except AttributeError:
            header.setSectionResizeMode(QHeaderView.Stretch)

        self.table.itemChanged.connect(self.update_calculation)
        t_layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("+ Add Layer")
        self.btn_add.clicked.connect(self.add_layer_row)
        self.btn_del = QPushButton("- Remove Layer")
        self.btn_del.clicked.connect(self.remove_layer_row)
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_del)
        t_layout.addLayout(btn_layout)

        left_layout.addWidget(table_box)
        splitter.addWidget(left_panel)

        # ----------------- Right Panel: Embedded Matplotlib -----------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Create Matplotlib Figure & Subplots
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

        splitter.setSizes([440, 840])

        # Load Default Materials & trigger first plot
        self.load_default_layers()
        QTimer.singleShot(50, self.update_calculation)

    # ----------------- Table Operations -----------------
    def add_layer_row(self, name="Layer", d_start=50.0, d_end=50.0, pot=0.100):
        self.table.blockSignals(True)
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(name)))
        self.table.setItem(row, 1, QTableWidgetItem(f"{d_start:.1f}"))
        self.table.setItem(row, 2, QTableWidgetItem(f"{d_end:.1f}"))
        self.table.setItem(row, 3, QTableWidgetItem(f"{pot:.4f}"))
        self.table.blockSignals(False)
        self.update_calculation()

    def remove_layer_row(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)
            self.update_calculation()

    def load_default_layers(self):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self.add_layer_row("Ni", d_start=40.0, d_end=200.0, pot=0.2410)
        self.add_layer_row("Ti", d_start=40.0, d_end=200.0, pot=-0.0490)
        self.table.blockSignals(False)

    def get_layers_from_table(self):
        layers = []
        for r in range(self.table.rowCount()):
            name_item = self.table.item(r, 0)
            d1_item = self.table.item(r, 1)
            d2_item = self.table.item(r, 2)
            p_item = self.table.item(r, 3)

            if None in (name_item, d1_item, d2_item, p_item):
                continue

            name = name_item.text().strip()
            d1 = safe_float(d1_item.text(), 50.0)
            d2 = safe_float(d2_item.text(), 50.0)
            pot = safe_float(p_item.text(), 0.0)

            layers.append({'name': name, 'd_start': d1, 'd_end': d2, 'pot': pot})
        return layers

    # ----------------- Live Calculation & Plotting -----------------
    def update_calculation(self):
        layers = self.get_layers_from_table()
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
# 4. Entry Point & Light UI Styling
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
QTableWidget {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    gridline-color: #eaeef2;
    border-radius: 4px;
}
QHeaderView::section {
    background-color: #f6f8fa;
    color: #57606a;
    font-weight: bold;
    border: 1px solid #d0d7de;
    padding: 4px;
}
QSpinBox, QDoubleSpinBox {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 4px;
    padding: 3px 6px;
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