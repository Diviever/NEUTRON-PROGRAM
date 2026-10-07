"""
Neutron Multilayer Reflectivity Designer — start the program from this file.

Project layout:
  calculation.py  thickness functions d(j) + Parratt reflectivity engine
  optimizer.py    supermirror optimizer (gradient descent + N search)
  gui.py          widgets and main window
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
