"""
Neutron Multilayer Reflectivity Designer — start the program from this file.

Files:
  calculation.py  Material and Interface classes + the two universal methods:
                    build_stack(materials, N, V_vac, V_sub, interface)   -> potentials, thicknesses
                    reflectivity(potentials, thicknesses, q)              -> R(q)
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
