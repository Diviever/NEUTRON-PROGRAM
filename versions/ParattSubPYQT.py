import os
import sys
import numpy as np
import pyqtgraph as pg

# Use pyqtgraph's internal Qt wrapper to ensure 100% version compatibility
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui


# ==============================================================================
# 1. Physics & Computation Engine (Vectorized Parratt with Sublayers)
# ==============================================================================
def interlayer_potentials(V_left, V_right, n):
    if n <= 0:
        return []
    if n == 1:
        return [0.5 * (V_left + V_right)]
    return [V_left + (V_right - V_left) * (k + 1) / (n + 1) for k in range(n)]


def build_stack_from_layers(layers, n_repeats, n_sublayers, d_sublayer, V_vac, V_sub):
    """
    layers: list of dicts [{'name': 'Ni', 'd_start': 40, 'd_end': 200, 'pot': 0.241}, ...]
    potentials are in ueV.
    """
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
# 2. Main PyQt GUI Window
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

        # QSplitter with version-safe Orientation
        try:
            orientation = QtCore.Qt.Orientation.Horizontal
        except AttributeError:
            orientation = QtCore.Qt.Horizontal
        splitter = QtWidgets.QSplitter(orientation)
        main_layout.addWidget(splitter)

        # ----------------- Left Panel: Controls & Table -----------------
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)

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

        # Top Plot: Reflectivity
        self.p_refl = self.plot_widget.addPlot(title="Neutron Reflectivity R(q)")
        self.p_refl.setLogMode(x=False, y=True)
        self.p_refl.setLabel('bottom', "q", units='Å⁻¹')
        self.p_refl.setLabel('left', "Reflectivity R")
        self.p_refl.showGrid(x=True, y=True, alpha=0.3)
        print(self.p_refl)
        self.curve_refl = self.p_refl.plot(pen=pg.mkPen(color='c', width=2))
        self.p_refl.setYRange(-5, 0.1)

        # Bottom Plot: Potential Profile
        self.plot_widget.nextRow()
        self.p_pot = self.plot_widget.addPlot(title="Discretized Potential Profile V(z)")
        self.p_pot.setLabel('bottom', "Depth z", units='Å')
        self.p_pot.setLabel('left', "Potential V", units='µeV')
        self.p_pot.showGrid(x=True, y=True, alpha=0.3)
        self.curve_pot = self.p_pot.plot(pen=pg.mkPen(color='y', width=1.5), fillLevel=0.0, brush=(255, 255, 0, 40))

        splitter.setSizes([420, 780])

        # Load Default Materials
        self.load_default_layers()

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
        self.add_layer_row("Ni", d_start=40.0, d_end=200.0, pot=0.2410)
        self.add_layer_row("Ti", d_start=40.0, d_end=200.0, pot=-0.0490)

    def get_layers_from_table(self):
        layers = []
        for r in range(self.table.rowCount()):
            try:
                name_item = self.table.item(r, 0)
                d1_item = self.table.item(r, 1)
                d2_item = self.table.item(r, 2)
                p_item = self.table.item(r, 3)

                if None in (name_item, d1_item, d2_item, p_item):
                    continue

                layers.append({
                    'name': name_item.text(),
                    'd_start': float(d1_item.text()),
                    'd_end': float(d2_item.text()),
                    'pot': float(p_item.text())
                })
            except (ValueError, AttributeError):
                continue
        return layers

    # ----------------- Computation & Live Update -----------------
    def update_calculation(self):
        print('updated')
        layers = self.get_layers_from_table()
        if not layers:
            self.curve_refl.clear()
            self.curve_pot.clear()
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
        total_z = z_edges[-1]
        z_stair = np.repeat(z_edges, 2)[1:-1]
        V_stair = np.repeat(pot_stack.real * 1e3, 2)  # meV -> ueV

        pad = 50.0
        z_plot = np.concatenate(([-pad, 0.0], z_stair, [total_z, total_z + pad]))
        V_plot = np.concatenate(([v_vac, v_vac], V_stair, [v_sub, v_sub]))
        self.curve_pot.setData(z_plot, V_plot)

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

        # Clip values to avoid log(0)
        R_clipped = np.clip(R, 1e-7, 1.2)
        print(R_clipped)
        self.curve_refl.setData(q_range, R_clipped)


# ==============================================================================
# 3. Entry Point
# ==============================================================================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOption('background', '#1e1e24')
    pg.setConfigOption('foreground', '#dcdcdc')
    pg.setConfigOptions(antialias=True)

    window = MultilayerApp()
    window.show()
    sys.exit(app.exec_() if hasattr(app, 'exec_') else app.exec())