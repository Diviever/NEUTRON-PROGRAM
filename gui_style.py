"""
gui_style.py — visual design of the GUI: Qt stylesheets and plot colors of the two themes

    THEMES["light"] / THEMES["dark"]   everything one theme needs:
        "style"    Qt stylesheet of the whole application
        "layers"   line colors of the material cards (cycled in the order the materials are added)
        "palette"  Qt palette colors (dark theme only: the parts a stylesheet does not reach,
                   e.g. tooltips, scroll bars, drop-down lists, plot toolbar icons)
        all other keys: colors of the plots (pyqtgraph, plots.py)

The dark theme uses the colors of the program icon: background #2C2C2C, light accent #BFC7FF,
dark accent #5E6598, and shades of gray for everything else.
"""

# ==============================================================================
# Light theme
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
    margin-top: 8px;
    font-weight: bold;
    padding: 9px 5px 5px 5px;
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
QLabel#DropZone {
    background-color: #ffffff;
    border: 2px dashed #8c959f;
    border-radius: 8px;
    padding: 6px;
    color: #57606a;
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
    padding: 1px 4px;
    font-weight: normal;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #0969da;
}
QPushButton {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 4px;
    padding: 3px 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #f3f4f6;
    border-color: #8c959f;
}
"""

# ==============================================================================
# Dark theme — #2C2C2C background, #BFC7FF light accent, #5E6598 dark accent, grays
# ==============================================================================
DARK_STYLE = """
QWidget {
    background-color: #2C2C2C;
    color: #E6E6E6;
    font-size: 9pt;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
QWidget:disabled {
    color: #7A7A7A;
}
QGroupBox {
    background-color: #333333;
    border: 1px solid #474747;
    border-radius: 6px;
    margin-top: 8px;
    font-weight: bold;
    padding: 9px 5px 5px 5px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: #BFC7FF;
}
QScrollArea {
    border: none;
    background-color: #333333;
}
QWidget#CardsContainer {
    background-color: #333333;
}
QFrame#LayerCard {
    background-color: #3A3A3A;
    border: 1px solid #4F4F4F;
    border-radius: 6px;
}
QFrame#LayerCard QLabel {
    font-weight: normal;
    background-color: transparent;
}
QLabel#OptStatus {
    color: #BFC7FF;
    font-weight: bold;
}
QLabel#DropZone {
    background-color: #333333;
    border: 2px dashed #6E6E6E;
    border-radius: 8px;
    padding: 6px;
    color: #A8A8A8;
    font-weight: bold;
}
QLabel#Formula {
    color: #A8A8A8;
    font-style: italic;
}
QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox {
    background-color: #262626;
    border: 1px solid #4F4F4F;
    border-radius: 4px;
    padding: 1px 4px;
    font-weight: normal;
    selection-background-color: #5E6598;
    selection-color: #FFFFFF;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #BFC7FF;
}
QComboBox QAbstractItemView {
    background-color: #333333;
    border: 1px solid #4F4F4F;
    selection-background-color: #5E6598;
    selection-color: #FFFFFF;
}
QPushButton {
    background-color: #3C3C3C;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 3px 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #5E6598;
    border-color: #BFC7FF;
}
QPushButton:pressed {
    background-color: #4C527C;
}
QPushButton:disabled {
    background-color: #333333;
    border-color: #444444;
}
QTabWidget::pane {
    border: 1px solid #474747;
    border-radius: 4px;
    top: -1px;
}
QTabBar::tab {
    background-color: #333333;
    color: #A8A8A8;
    border: 1px solid #474747;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 5px 12px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #2C2C2C;
    color: #BFC7FF;
    border-color: #5E6598;
}
QTabBar::tab:hover:!selected {
    color: #E6E6E6;
}
QTableWidget {
    background-color: #262626;
    alternate-background-color: #303030;
    gridline-color: #474747;
    border: 1px solid #474747;
    selection-background-color: #5E6598;
    selection-color: #FFFFFF;
}
QHeaderView::section {
    background-color: #3A3A3A;
    color: #E6E6E6;
    border: none;
    border-right: 1px solid #474747;
    border-bottom: 1px solid #474747;
    padding: 4px;
}
QTableCornerButton::section {
    background-color: #3A3A3A;
    border: none;
}
QToolTip {
    background-color: #3A3A3A;
    color: #E6E6E6;
    border: 1px solid #5E6598;
}
QSplitter::handle {
    background-color: #2C2C2C;
}
"""

THEMES = {
    "light": {
        "style": LIGHT_STYLE,
        "layers": ['#0052cc', '#c81e1e', '#1a7f37', '#8250df', '#bf8700', '#0a7ea4', '#cf222e'],
        "palette": None,                  # the normal palette of the system style
        # plots
        "figure": '#ffffff', "axes": '#ffffff', "card_plot": '#ffffff',
        "text": '#24292f', "edge": '#57606a', "grid": '#b0b7bf',
        "neutron": '#0052cc', "xray": '#8250df', "data": '#24292f',
        "q_c": '#e02424', "edge_line": '#1a7f37', "target": '#57606a',
        "range_ok": '#1a7f37', "range_low": '#bf8700', "range_alpha": 0.07,
        "potential": '#c81e1e', "epsilon": '#8250df', "warning": '#cf222e',
    },
    "dark": {
        "style": DARK_STYLE,
        "layers": ['#BFC7FF', '#D9D9D9', '#8E96D0', '#A6A6A6', '#5E6598', '#F2F2F2', '#7A7A7A'],
        "palette": {                      # Qt palette role -> color
            "Window": '#2C2C2C', "WindowText": '#E6E6E6', "Base": '#262626', "AlternateBase": '#303030',
            "ToolTipBase": '#3A3A3A', "ToolTipText": '#E6E6E6', "Text": '#E6E6E6', "Button": '#3C3C3C',
            "ButtonText": '#E6E6E6', "BrightText": '#FFFFFF', "Highlight": '#5E6598',
            "HighlightedText": '#FFFFFF', "Link": '#BFC7FF', "PlaceholderText": '#7A7A7A',
        },
        "disabled_text": '#7A7A7A',
        # plots
        "figure": '#2C2C2C', "axes": '#262626', "card_plot": '#3A3A3A',
        "text": '#E6E6E6', "edge": '#8A8A8A', "grid": '#555555',
        "neutron": '#BFC7FF', "xray": '#8E96D0', "data": '#D9D9D9',
        "q_c": '#A6A6A6', "edge_line": '#BFC7FF', "target": '#8A8A8A',
        "range_ok": '#5E6598', "range_low": '#7A7A7A', "range_alpha": 0.22,
        "potential": '#BFC7FF', "epsilon": '#8E96D0', "warning": '#FF8A8A',
    },
}

# The theme in use: gui.apply_theme() refills this dict, the plots (plots.py) read their colors from it
colors = dict(THEMES["light"])
