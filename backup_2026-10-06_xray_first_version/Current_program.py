"""
Neutron Multilayer Reflectivity Designer — start the program from this file.

Files:
  calculation.py  Material and Interface classes + the universal methods:
                    1.   build_stack(materials, N, top, substrate, interface, quantity)  -> values, thicknesses
                         (values = neutron potentials V or X-ray permittivities ε of the same structure)
                    2.1  neutron_reflectivity(potentials, thicknesses, q)                -> R(q), neutrons
                    2.2  xray_reflectivity(epsilons, thicknesses, q, λ, polarization)    -> R(q), X-rays
  optimizer.py    gradient descent on the Material objects + search over N
  gui.py          window: one card per Material, plots, optimizer controls
  gui_style.py    stylesheet and colors
"""
import sys

from gui import QApplication, MultilayerApp
from gui_style import LIGHT_STYLE


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(LIGHT_STYLE)

    window = MultilayerApp()
    window.show()
    sys.exit(app.exec() if hasattr(app, 'exec') else app.exec_())
