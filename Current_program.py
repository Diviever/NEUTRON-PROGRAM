"""
Neutron Multilayer Reflectivity Designer — start the program from this file.

Files:
  calculation.py  Material and Interface classes + the universal methods:
                    1.   build_stack(materials, N, top, substrate, interface, quantity)  -> values, thicknesses, roughness
                         (values = neutron potentials V or X-ray permittivities ε of the same structure)
                    2.1  neutron_reflectivity(potentials, thicknesses, q, roughness)                -> R(q), neutrons
                    2.2  xray_reflectivity(epsilons, thicknesses, q, λ, polarization, roughness)    -> R(q), X-rays
                  Roughness: "factor" (fast, as Multifitting) or "slices" (graded profile, exact, slow)
  structure_file.py  material library (materials/<Name>.txt: row 1 V in µeV, row 2 ε real imaginary),
                     .xlsx structure files (column A material, column B thickness in nm)
                     and measured curves (.xy: column 1 θ / 2θ / q, column 2 intensity)
  optimizer.py    gradient descent on the Material objects + search over N
  gui.py          window: one card per Material, parameter tabs, optimizer controls
  plots.py        the plots, drawn with pyqtgraph (install once:  pip install pyqtgraph)
  gui_style.py    light and dark theme: stylesheets and colors
  icon.svg        program icon (icon.png / icon.ico are made from it; the window uses icon.png)
"""
import sys

from gui import QApplication, MultilayerApp, program_icon


if __name__ == "__main__":
    if sys.platform == "win32":
        # Own taskbar entry with the program icon (otherwise Windows groups the window under python.exe)
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("NeutronProgram.MultilayerDesigner")

    app = QApplication(sys.argv)
    app.setWindowIcon(program_icon())

    window = MultilayerApp()                       # applies the theme chosen in Beams & plots (light at start)
    window.show()
    sys.exit(app.exec() if hasattr(app, 'exec') else app.exec_())
