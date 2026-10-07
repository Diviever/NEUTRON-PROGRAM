"""
gui_style.py — visual design of the GUI: Qt stylesheet and plot colors
"""

# Line colors of the materials (cycled in the order the layers are added)
LAYER_COLORS = ['#0052cc', '#c81e1e', '#1a7f37', '#8250df', '#bf8700', '#0a7ea4', '#cf222e']


# Qt stylesheet applied to the whole application
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
QScrollArea {
    border: none;
    background-color: #ffffff;
}
QWidget#CardsContainer {
    background-color: #ffffff;
}
QFrame#LayerCard {
    background-color: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 6px;
}
QFrame#LayerCard QLabel {
    font-weight: normal;
}
QLabel#OptStatus {
    color: #0969da;
    font-weight: bold;
}
QLabel#Formula {
    color: #57606a;
    font-style: italic;
}
QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 4px;
    padding: 3px 6px;
    font-weight: normal;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #0969da;
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
