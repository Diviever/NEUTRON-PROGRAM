import os
import sys
import numpy as np
import pyqtgraph as pg

# Use pyqtgraph's internal Qt wrapper to ensure 100% version compatibility
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui


# ==============================================================================
# 1. Helper: Locale-safe float conversion
# ==============================================================================
def safe_float(val, default=0.0):
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        cleaned = str(val).strip().replace(',', '.')
        return float(cleaned)
    except (ValueError, TypeError):
        return default


# ==============================================================================
# 2. Physics & Computation Engine (Vectorized Parratt with Sublayers)
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
        pot_values.append(lay['pot'] * 1e-3)  # Convert ueV -> meV

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
# 3. Main PyQt GUI Window (Light Theme)
# ==============================================================================
class MultilayerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neutron Multilayer Reflectivity Designer")
        self.resize(1200, 750)

        # Central layout
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QHBoxLayout(main_widget)

        try:
            orientation = QtCore.Qt.Orientation.Horizontal
        except AttributeError:
            orientation = QtCore.Qt.Horizontal
        splitter = QtWidgets.QSplitter(orientation)
        main_layout.addWidget(splitter)

        # ----------------- Left Panel: Controls & Table -----------------
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)

        # Periodic Repeat Controls
        global_box = QtWidgets.QGroupBox("Multilayer & Interface Settings")
        g_layout = QtWidgets.QVBoxLayout(global_box)

        # Repetitions
        h1 = QtWidgets.QHBoxLayout()
        h1.addWidget(QtWidgets.QLabel("Repetitions (N):"))
        self.spin_repeats = QtWidgets.QSpinBox()
        self.spin_repeats.setRange(1, 500)
        self.spin_repeats.setValue(30)
        self.spin_repeats.valueChanged.connect(self.update_calculation)
        h1.addWidget(self.spin_repeats)
        g_layout.addLayout(h1)

        # Sublayers
        h2 = QtWidgets.QHBoxLayout()
        h2.addWidget(QtWidgets.QLabel("Interlayer Sublayers (n):"))
        self.spin_sublayers = QtWidgets.QSpinBox()
        self.spin_sublayers.setRange(0, 50)
        self.spin_sublayers.setValue(5)
        self.spin_sublayers.valueChanged.connect(self.update_calculation)
        h2.addWidget(self.spin_sublayers)
        g_layout.addLayout(h2)

        # Step thickness
        h3 = QtWidgets.QHBoxLayout()
        h3.addWidget(QtWidgets.QLabel("Sublayer Step d (Å):"))
        self.spin_d_sub = QtWidgets.QDoubleSpinBox()
        self.spin_d_sub.setRange(0.1, 20.0)
        self.spin_d_sub.setSingleStep(0.5)
        self.spin_d_sub.setValue(1.5)
        self.spin_d_sub.valueChanged.connect(self.update_calculation)
        h3.addWidget(self.spin_d_sub)
        g_layout.addLayout(h3)

        # Substrate & Vacuum Potentials
        h4 = QtWidgets.QHBoxLayout()
        h4.addWidget(QtWidgets.QLabel("V_sub (µeV):"))
        self.spin_vsub = QtWidgets.QDoubleSpinBox()
        self.spin_vsub.setRange(-2.0, 5.0)
        self.spin_vsub.setDecimals(3)
        self.spin_vsub.setValue(0.090)
        self.spin_vsub.valueChanged.connect(self.update_calculation)
        h4.addWidget(self.spin_vsub)

        h4.addWidget(QtWidgets.QLabel("V_vac (µeV):"))
        self.spin_vvac = QtWidgets.QDoubleSpinBox()
        self.spin_vvac.setRange(-2.0, 5.0)
        self.spin_vvac.setDecimals(3)
        self.spin_vvac.setValue(0.0)
        self.spin_vvac.valueChanged.connect(self.update_calculation)
        h4.addWidget(self.spin_vvac)
        g_layout.addLayout(h4)

        left_layout.addWidget(global_box)

        # Unit Cell Layer Table
        table_box = QtWidgets.QGroupBox("Periodic Unit Cell Layers")
        t_layout = QtWidgets.QVBoxLayout(table_box)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Material", "d_start (Å)", "d_end (Å)", "V (µeV)"])
        header = self.table.horizontalHeader()
        try:
            header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        except AttributeError:
            header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        self.table.itemChanged.connect(self.update_calculation)
        t_layout.addWidget(self.table)

        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("+ Add Layer")
        self.btn_add.clicked.connect(self.add_layer_row)
        self.btn_del = QtWidgets.QPushButton("- Remove Layer")
        self.btn_del.clicked.connect(self.remove_layer_row)
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_del)
        t_layout.addLayout(btn_layout)

        left_layout.addWidget(table_box)
        splitter.addWidget(left_panel)

        # ----------------- Right Panel: pyqtgraph Visualizer -----------------
        self.plot_widget = pg.GraphicsLayoutWidget()
        splitter.addWidget(self.plot_widget)

        # --- Top Plot: Reflectivity ---
        self.p_refl = self.plot_widget.addPlot(title="<b style='color:#1a1a1a; font-size:11pt;'>Neutron Reflectivity R(q)</b>")
        self.p_refl.setLogMode(x=False, y=True)
        self.p_refl.getAxis('bottom').enableAutoSIPrefix(False)
        self.p_refl.getAxis('left').enableAutoSIPrefix(False)
        self.p_refl.setLabel('bottom', "Momentum transfer q (Å⁻¹)", color='#222222')
        self.p_refl.setLabel('left', "Reflectivity R", color='#222222')
        self.p_refl.showGrid(x=True, y=True, alpha=0.5)

        # Initialize with baseline data to establish a valid bounding box
        q_init = np.linspace(0.001, 0.09, 200)
        R_init = np.ones(200)
        self.curve_refl = self.p_refl.plot(
            q_init, R_init,
            pen=pg.mkPen(color=(0, 82, 204), width=2.5)  # Royal Blue
        )

        # --- Bottom Plot: Potential Profile ---
        self.plot_widget.nextRow()
        self.p_pot = self.plot_widget.addPlot(title="<b style='color:#1a1a1a; font-size:11pt;'>Discretized Potential Profile V(z)</b>")
        self.p_pot.getAxis('bottom').enableAutoSIPrefix(False)
        self.p_pot.getAxis('left').enableAutoSIPrefix(False)
        self.p_pot.setLabel('bottom', "Depth z (Å)", color='#222222')
        self.p_pot.setLabel('left', "Potential V (µeV)", color='#222222')
        self.p_pot.showGrid(x=True, y=True, alpha=0.5)

        z_init = np.linspace(0, 1000, 200)
        V_init = np.zeros(200)
        self.curve_pot = self.p_pot.plot(
            z_init, V_init,
            pen=pg.mkPen(color=(200, 30, 30), width=2.2), # Crimson Red
            fillLevel=0.0,
            brush=(200, 30, 30, 30)
        )

        splitter.setSizes([420, 780])

        # Load Default Materials
        self.load_default_layers()

        # Trigger immediate calculation
        QtCore.QTimer.singleShot(50, self.update_calculation)

    # ----------------- Table Operations -----------------
    def add_layer_row(self, name="Layer", d_start=50.0, d_end=50.0, pot=0.100):
        self.table.blockSignals(True)
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(name)))
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{d_start:.1f}"))
        self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{d_end:.1f}"))
        self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{pot:.4f}"))
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
        
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem("Ni"))
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem("40.0"))
        self.table.setItem(row, 2, QtWidgets.QTableWidgetItem("200.0"))
        self.table.setItem(row, 3, QtWidgets.QTableWidgetItem("0.2410"))

        row2 = self.table.rowCount()
        self.table.insertRow(row2)
        self.table.setItem(row2, 0, QtWidgets.QTableWidgetItem("Ti"))
        self.table.setItem(row2, 1, QtWidgets.QTableWidgetItem("40.0"))
        self.table.setItem(row2, 2, QtWidgets.QTableWidgetItem("200.0"))
        self.table.setItem(row2, 3, QtWidgets.QTableWidgetItem("-0.0490"))
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

    # ----------------- Computation & Live Update -----------------
    def update_calculation(self):
        layers = self.get_layers_from_table()
        if not layers:
            return

        n_rep = self.spin_repeats.value()
        n_sub = self.spin_sublayers.value()
        d_sub = self.spin_d_sub.value()
        v_vac = self.spin_vvac.value()
        v_sub = self.spin_vsub.value()

        # 1. Build Discretized Stack
        thk_stack, pot_stack = build_stack_from_layers(
            layers, n_repeats=n_rep, n_sublayers=n_sub, d_sublayer=d_sub,
            V_vac=v_vac, V_sub=v_sub
        )

        if len(thk_stack) == 0:
            return

        # 2. Update Potential Profile Plot V(z)
        z_edges = np.concatenate(([0.0], np.cumsum(thk_stack)))
        total_z = float(z_edges[-1])
        z_stair = np.repeat(z_edges, 2)[1:-1]
        V_stair = np.repeat(pot_stack.real * 1e3, 2)  # meV -> ueV

        pad = 50.0
        z_plot = np.concatenate(([-pad, 0.0], z_stair, [total_z, total_z + pad]))
        V_plot = np.concatenate(([v_vac, v_vac], V_stair, [v_sub, v_sub]))
        
        self.curve_pot.setData(x=z_plot, y=V_plot)
        self.p_pot.setXRange(-pad, total_z + pad, padding=0.02)
        ymin, ymax = float(np.min(V_plot)), float(np.max(V_plot))
        ypad = (ymax - ymin) * 0.2 if ymax > ymin else 0.1
        self.p_pot.setYRange(ymin - ypad, ymax + ypad)

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
        
        self.curve_refl.setData(x=q_range, y=R_safe)
        self.p_refl.setXRange(0.001, 0.09, padding=0.02)
        self.p_refl.setYRange(-5.0, 0.1, padding=0.02)


# ==============================================================================
# 4. Light Theme Stylesheet & Entry Point
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
    padding: 10px 6px 6px 6px;
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
    color: #24292f;
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
    padding: 2px 4px;
    color: #24292f;
}
QPushButton {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 4px;
    padding: 5px 12px;
    font-weight: bold;
    color: #24292f;
}
QPushButton:hover {
    background-color: #f3f4f6;
    border-color: #8c959f;
}
QPushButton:pressed {
    background-color: #ebecf0;
}
"""

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(LIGHT_STYLE)

    # Set pyqtgraph global light palette
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', '#222222')
    pg.setConfigOptions(antialias=False)

    window = MultilayerApp()
    window.show()
    sys.exit(app.exec_() if hasattr(app, 'exec_') else app.exec())