"""
plots.py — the plots of the program, drawn with pyqtgraph

pyqtgraph draws directly with Qt, so redrawing is fast (also during the optimization). In every plot:
drag = move, mouse wheel = zoom, right-click = menu with "View All" and "Export…" (PNG, SVG, CSV of the
curves), the small "A" button in the corner = back to the full view.
Under every plot a readout line shows the mouse position (x, y; and θ on the R(q) plots); a left click marks
the calculated curve at that x and writes its exact value on the line below. "Save data…" writes the calculated curves of that plot
(full resolution, true values) to a CSV or text file — use it instead of pyqtgraph's own Export → CSV / HDF5,
which exports what is drawn (only the visible, thinned-out points; log10 values on a log axis).

    AngleAxis       top axis of an R(q) plot: ticks at round glancing angles θ (°) of that beam, q = 2k·sin θ
    PlotBox         one plot in an outlined frame (styled like the group boxes): title above, readout below
    PlotPanel       the four plot boxes of the main window: X-ray column R(q) and 1 − Re ε(z),
                    neutron column R(q) and V(z)
                        draw_reflectivity()  R(q) on a log or linear axis + measured points
                        add_vline(), add_segment(), add_range()   marks with a legend line
                        draw_profile()       depth profile as steps
    MiniPlot        small thickness plot of a material card

Colors come from gui_style.colors (the current theme).
"""
import numpy as np

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
except ImportError as error:                               # pyqtgraph is not part of PyQt
    raise ImportError("The plots need the pyqtgraph package:  pip install pyqtgraph") from error

from calculation import glancing_angle, q_from_angle
from gui_style import colors

pg.setConfigOptions(antialias=True)

DASH = [4, 3]                    # pen dash patterns (drawn length, gap) in pen widths
DOT = [1, 3]


# ==============================================================================
# Small helpers
# ==============================================================================
def pen(color, width=1.4, dash=None, alpha=1.0):
    """Line style: color, width in pixels, optional dash pattern, transparency."""
    qcolor = pg.mkColor(color)
    qcolor.setAlphaF(alpha)
    return pg.mkPen(color=qcolor, width=width, dash=dash)


def brush(color, alpha=1.0):
    """Fill style: color with transparency."""
    qcolor = pg.mkColor(color)
    qcolor.setAlphaF(alpha)
    return pg.mkBrush(qcolor)


def font(points):
    f = QtGui.QFont()
    f.setPointSizeF(points)
    return f


def style_plot(plot, background, title, bottom, left, title_size="9pt", label_size="9pt", tick_size=8.0):
    """Theme colors, title and axis labels of one plot (texts may contain HTML: <sub>, <sup>, <i>, <br>)."""
    plot.getViewBox().setBackgroundColor(background)
    box_title = getattr(plot, "box_title", None)            # plot inside a PlotBox: the title is a Qt label above it
    if isinstance(box_title, QtWidgets.QLabel):
        box_title.setText(title)
    else:
        plot.setTitle(title, color=colors["text"], size=title_size, bold=True)
    for name, text in (("bottom", bottom), ("left", left)):
        plot.setLabel(name, text, color=colors["text"], **{"font-size": label_size})
    for name in ("bottom", "left", "top"):
        axis = plot.getAxis(name)
        axis.setPen(colors["edge"])
        axis.setTextPen(colors["text"])
        axis.setStyle(tickFont=font(tick_size))
    plot.showGrid(x=True, y=True, alpha=0.25)


# ==============================================================================
# Top axis: glancing angle
# ==============================================================================
class AngleAxis(pg.AxisItem):
    """
    Top axis of an R(q) plot. The plot itself uses q; this axis puts its ticks at round glancing angles θ
    and writes them in degrees:  q = 2k·sin θ, with k of the beam of that plot (set_k).
    """

    def __init__(self):
        super().__init__(orientation="top")
        self.k = 1.0

    def set_k(self, k):
        self.k = float(k)
        self.picture = None                                 # forget the drawn ticks
        self.update()

    def tickValues(self, minVal, maxVal, size):
        """Round angles between the angles of the two plot edges, returned as q positions."""
        theta_lo = float(glancing_angle(max(minVal, 0.0), self.k))
        theta_hi = float(glancing_angle(max(maxVal, 0.0), self.k))
        levels, used = [], []
        for spacing, offset in self.tickSpacing(theta_lo, theta_hi, size):
            first = np.ceil((theta_lo - offset) / spacing) * spacing + offset
            angles = [a for a in np.arange(first, theta_hi + 1e-9 * spacing, spacing)
                      if all(abs(a - b) > 0.01 * spacing for b in used)]       # no tick twice (major + minor)
            used.extend(angles)
            levels.append((spacing, list(q_from_angle(np.array(angles), self.k))))
        return levels

    def tickStrings(self, values, scale, spacing):
        return [f"{round(float(glancing_angle(q, self.k)), 6):g}" for q in values]


# ==============================================================================
# The four plots of the main window
# ==============================================================================
class PlotBox(QtWidgets.QFrame):
    """
    One plot in an outlined frame, styled like the group boxes (QFrame#PlotFrame in gui_style), with its
    title as a Qt label above the plot: titles of several lines then never overlap the plot's top axis.
    """

    def __init__(self, top_axis=None):
        super().__init__()
        self.setObjectName("PlotFrame")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)
        self.title = QtWidgets.QLabel("")
        self.title.setObjectName("PlotTitle")
        flags = getattr(QtCore.Qt, "AlignmentFlag", QtCore.Qt)            # Qt6: scoped enums, Qt5: flat
        self.title.setAlignment(getattr(flags, "AlignCenter", None) or QtCore.Qt.AlignCenter)
        formats = getattr(QtCore.Qt, "TextFormat", QtCore.Qt)
        self.title.setTextFormat(getattr(formats, "RichText", None) or QtCore.Qt.RichText)   # <sub>, <br>, <i>
        self.title.setWordWrap(True)
        layout.addWidget(self.title)
        self.widget = pg.PlotWidget(axisItems={"top": top_axis} if top_axis else None)
        layout.addWidget(self.widget, 1)
        self.plot = self.widget.getPlotItem()
        self.plot.box_title = self.title                    # style_plot() writes the title here

        # ---- readout of the mouse position and of the clicked curve point ----
        # two lines: the cursor position, and below it the marked point. Their width is "ignored" by the layout:
        # a long text is cut at the box edge instead of widening the plot box.
        policies = getattr(QtWidgets.QSizePolicy, "Policy", QtWidgets.QSizePolicy)
        lines = QtWidgets.QVBoxLayout()
        lines.setContentsMargins(0, 0, 0, 0)
        lines.setSpacing(0)
        self.readout = QtWidgets.QLabel("cursor: move the mouse over the plot")
        self.marked = QtWidgets.QLabel("marked: left-click the plot")
        for label in (self.readout, self.marked):
            label.setObjectName("Formula")
            label.setTextFormat(getattr(formats, "RichText", None) or QtCore.Qt.RichText)
            label.setSizePolicy(policies.Ignored, policies.Preferred)
            lines.addWidget(label)
        bottom = QtWidgets.QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.addLayout(lines, 1)
        self.save_button = QtWidgets.QPushButton("Save data…")
        self.save_button.setToolTip("Write the calculated curves of this plot (all points, true values) to a .csv or .txt file")
        self.save_button.clicked.connect(lambda *_: self.save_data())
        bottom.addWidget(self.save_button)
        layout.addLayout(bottom)
        self.series = []                                    # (name, x[], y[]) of the curves drawn in this plot
        self.top_axis = top_axis                            # AngleAxis of an R(q) plot (for θ), else None
        self.names = ("x", "", "y", "")                     # x name, x unit, y name, y unit (set by the drawing)
        self.curve = None                                   # (x[], y[]) of the main curve, true values
        self.steps = False                                  # True: a depth profile (y constant within a slice)
        self.log_y = False                                  # True: the view's y is log10 of the value
        self.clicked = ""                                   # text of the last clicked point
        self.mark_items = []
        scene = self.widget.scene()
        scene.sigMouseMoved.connect(self.on_mouse_moved)
        scene.sigMouseClicked.connect(self.on_mouse_clicked)

    def apply_theme(self):
        self.widget.setBackground(colors["panel"])

    def set_curve(self, x, y, names, steps=False, log_y=False):
        """The curve the readout and the click refer to (called by the drawing methods)."""
        self.curve = (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        self.names, self.steps, self.log_y = names, steps, log_y
        self.series = []                                    # a new drawing: new curves to save
        self.mark_items = []                                # the redraw (plot.clear) removed the old mark
        self.clicked = ""
        self.marked.setText("marked: left-click the plot")

    def add_series(self, name, x, y):
        """Remember a drawn curve with its true values (for Save data…)."""
        self.series.append((name, np.asarray(x, dtype=float), np.asarray(y, dtype=float)))

    def save_data(self, path=None):
        """
        Write the curves of this plot to a file, one column pair per curve (x, y), all calculated points.
        .csv: ';' between columns and a decimal comma (opens directly in Excel with a Russian locale);
        .txt / .dat: tab between columns and a decimal point.
        Depth profiles are written as one row per slice: z top, z bottom, value.
        """
        if not self.series:
            self.readout.setText("Nothing to save yet.")
            return
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save the curves of this plot", "",
                "CSV for Excel, ';' and decimal comma (*.csv);;Text, tab and decimal point (*.txt *.dat)")
            if not path:
                return
        excel = path.lower().endswith(".csv")
        sep = ";" if excel else "\t"

        def number(v):
            text = f"{v:.8g}"
            return text.replace(".", ",") if excel else text

        if self.steps:                                      # profile: slices
            name, edges, values = self.series[0]
            header = [f"{self.names[0]} top ({self.names[1]})", f"{self.names[0]} bottom ({self.names[1]})",
                      f"{self.names[2]} {self.names[3]}".strip()]
            rows = [[number(a), number(b), number(v)] for a, b, v in zip(edges[:-1], edges[1:], values)]
        else:                                               # curves: x, y column pair per curve
            x_head = f"{self.names[0]} ({self.names[1]})"
            header = []
            for name, _, _ in self.series:
                header += [f"{x_head} [{name}]", f"{self.names[2]} [{name}]"]
            length = max(x.size for _, x, _ in self.series)
            rows = []
            for i in range(length):
                row = []
                for _, x, y in self.series:
                    row += [number(x[i]), number(y[i])] if i < x.size else ["", ""]
                rows.append(row)
        with open(path, "w", encoding="utf-8-sig", newline="") as file:
            file.write(sep.join(header) + "\n")
            for row in rows:
                file.write(sep.join(row) + "\n")
        self.readout.setText(f"Saved {len(self.series)} curve(s), {len(rows)} rows → {path}")

    def describe(self, x, y):
        """'q = 0.0345 Å⁻¹ (θ = 0.789°), R = 0.912' for the point (x, y) in true values."""
        x_name, x_unit, y_name, y_unit = self.names
        text = f"{x_name} = {x:.5g} {x_unit}"
        if self.top_axis is not None:
            text += f" (θ = {float(glancing_angle(max(x, 0.0), self.top_axis.k)):.4g}°)"
        return text + f",  {y_name} = {y:.4g} {y_unit}".rstrip()

    def value_at(self, x):
        """Calculated curve at x: the nearest calculated point (R(q)), or the slice containing x (profiles)."""
        xs, ys = self.curve
        if self.steps:                                      # xs = slice edges, ys = one value per slice
            i = int(np.clip(np.searchsorted(xs, x) - 1, 0, ys.size - 1))
            return x, ys[i]
        i = int(np.argmin(np.abs(xs - x)))
        return xs[i], ys[i]

    def view_point(self, scene_pos):
        """(x, y in view coordinates) of a mouse position inside the plot area, else None."""
        box = self.plot.vb
        if not box.sceneBoundingRect().contains(scene_pos):
            return None
        point = box.mapSceneToView(scene_pos)
        return point.x(), point.y()

    def on_mouse_moved(self, scene_pos):
        point = self.view_point(scene_pos)
        if point is None or self.curve is None:
            return
        x, y = point
        self.show_readout(x, 10.0 ** y if self.log_y else y)

    def show_readout(self, x, y):
        self.readout.setText("cursor: " + self.describe(x, y))

    def on_mouse_clicked(self, event):
        left = getattr(getattr(QtCore.Qt, "MouseButton", QtCore.Qt), "LeftButton", None) or QtCore.Qt.LeftButton
        if event.button() != left or self.curve is None:
            return
        point = self.view_point(event.scenePos())
        if point is not None:
            self.mark(point[0])

    def mark(self, x):
        """Mark the calculated curve at x with a dot and its value, also in the readout line."""
        x, y = self.value_at(x)
        for item in self.mark_items:                        # one mark at a time
            self.plot.removeItem(item)
        dot = pg.PlotDataItem([x], [y], pen=None, symbol="o", symbolSize=8, symbolPen=pen(colors["text"], 1.2),
                              symbolBrush=None)
        self.plot.addItem(dot)                              # PlotDataItem: the plot applies its log axis itself
        self.clicked = self.describe(x, y)
        x_lo, x_hi = self.plot.vb.viewRange()[0]
        right_half = x > 0.5 * (x_lo + x_hi)                # the label goes to the side with room
        label = pg.TextItem(self.clicked, color=colors["text"], anchor=(1 if right_half else 0, 1),
                            fill=brush(colors["axes"], 0.85))
        label.setPos(x, np.log10(y) if self.log_y and y > 0 else y)   # TextItem: view coordinates
        self.plot.addItem(label)
        self.mark_items = [dot, label]
        self.marked.setText("<b>marked:</b> " + self.clicked)


class PlotPanel(QtWidgets.QWidget):
    """Two columns of outlined plot boxes: X-ray R(q) over 1 − Re ε(z), neutron R(q) over V(z)."""

    def __init__(self):
        super().__init__()
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
        self.boxes = [PlotBox(AngleAxis()), PlotBox(), PlotBox(AngleAxis()), PlotBox()]
        for box, (row, col) in zip(self.boxes, ((0, 0), (1, 0), (0, 1), (1, 1))):
            grid.addWidget(box, row, col)
        self.xray, self.epsilon, self.neutron, self.potential = (box.plot for box in self.boxes)
        self.box_of = {box.plot: box for box in self.boxes}
        for plot in (self.xray, self.epsilon, self.neutron, self.potential):
            for name in ("bottom", "left", "top"):
                plot.getAxis(name).enableAutoSIPrefix(False)    # no "(×0.001)": the numbers are shown as they are
        for plot in (self.epsilon, self.potential):
            plot.getAxis("bottom").setHeight(62)                # room for the two-line depth label
        self.legends = {}
        for plot in (self.neutron, self.xray):
            plot.showAxis("top")
            plot.setDownsampling(auto=True, mode="peak")    # long curves: draw only what the pixels can show
            plot.setClipToView(True)
            self.legends[plot] = plot.addLegend(labelTextSize="7pt")
        self.apply_theme()

    def apply_theme(self):
        for box in self.boxes:
            box.apply_theme()

    def clear_all(self):
        for plot in (self.neutron, self.xray, self.potential, self.epsilon):
            plot.clear()
        for legend in self.legends.values():
            legend.clear()

    # ---- R(q) ----
    def draw_reflectivity(self, plot, q, R, k, color, log_scale, log_decades, background=0.0,
                          measured=None, measured_name="", band=None, label="Parratt"):
        """
        R(q) curve, with the glancing angle of this beam (k) on the top axis.
        log_scale: the axis follows the curves, from the decade above max R down to the decade of min R,
                   but at most log_decades decades.
        background is added to the drawn curve only (Multifitting's 'Background').
        measured = (q[], R[]) of a measured curve, drawn as points (None = no file).
        band = (low[], high[]): spread of several random structures, drawn as a shaded band (None = no band).
        """
        R = R + background
        self.box_of[plot].set_curve(q, R, ("q", "Å⁻¹", "R", ""), log_y=log_scale)   # readout / click: true values
        if background != 0:
            label += f" + background {background:.3g}"
        shown = R if measured is None else np.concatenate((R, measured[1]))    # what the y axis must hold
        if band is not None:
            band = (band[0] + background, band[1] + background)
            shown = np.concatenate((shown, band[1]))
        plot.setLogMode(x=False, y=log_scale)
        if log_scale:
            R_top = max(float(shown.max()), 1e-300)
            R_low = float(shown[shown > 0].min()) if np.any(shown > 0) else R_top
            top_decade = np.ceil(np.log10(R_top))
            bottom_decade = max(np.floor(np.log10(R_low)), top_decade - log_decades)
            R = np.clip(R, 10.0 ** (bottom_decade - 1), None)                   # no zeros on a log axis
            y_range = (bottom_decade, np.log10(R_top * 1.5))                     # log axis: range in decades
        else:
            y_range = (0.0, max(1.05, 1.05 * float(shown.max())))
        box = self.box_of[plot]
        box.add_series(label, box.curve[0], box.curve[1])                     # true values, before the log clip
        curve = plot.plot(q, R, pen=pen(color, 1.4), name=label)
        curve.setZValue(1)                                                       # the curve over the points
        if band is not None:
            floor = 10.0 ** (y_range[0] - 1) if log_scale else -np.inf
            low = plot.plot(q, np.maximum(band[0], floor), pen=None)
            high = plot.plot(q, np.maximum(band[1], floor), pen=None)
            plot.addItem(pg.FillBetweenItem(low, high, brush=brush(color, 0.25)))
            box.add_series("16 % of the random structures", q, band[0])
            box.add_series("84 % of the random structures", q, band[1])
            self.legends[plot].addItem(pg.PlotDataItem(pen=None, fillLevel=0, fillBrush=brush(color, 0.4)),
                                       "16–84 % of the random structures")
        if measured is not None:
            q_m, R_m = measured
            keep = R_m > 0 if log_scale else np.isfinite(R_m)                   # log axis: no points ≤ 0
            box.add_series(f"measured: {measured_name}", q_m, R_m)
            plot.plot(q_m[keep], R_m[keep], pen=None, symbol="o", symbolSize=3, symbolPen=None,
                      symbolBrush=brush(colors["data"], 0.75), name=f"measured: {measured_name}")
        plot.setXRange(0.0, q[-1], padding=0)
        plot.setYRange(*y_range, padding=0)
        plot.getAxis("top").set_k(k)
        plot.setLabel("top", "Glancing angle θ (°)", color=colors["text"], **{"font-size": "8pt"})
        self.legends[plot].setLabelTextColor(colors["text"])
        self.legends[plot].setBrush(brush(colors["axes"], 0.9))
        self.legends[plot].setPen(pen(colors["edge"], 1.0))

    def place_legend(self, plot, corner):
        """Legend in the 'lower left' or 'upper right' corner of the plot."""
        anchor = (0, 1) if corner == "lower left" else (1, 0)
        offset = (10, -10) if corner == "lower left" else (-10, 10)
        self.legends[plot].anchor(itemPos=anchor, parentPos=anchor, offset=offset)

    def add_vline(self, plot, x, color, dash, label, width=1.2, alpha=0.8):
        """Vertical line at x across the whole plot, with a legend line."""
        style = pen(color, width, dash, alpha)
        plot.addItem(pg.InfiniteLine(pos=x, angle=90, pen=style, movable=False))
        self.legends[plot].addItem(pg.PlotDataItem(pen=style), label)

    def add_segment(self, plot, x0, x1, y, color, dash, label, width=1.0):
        """Horizontal line at height y from x0 to x1 (works on log axes too: it is a curve)."""
        plot.plot([x0, x1], [y, y], pen=pen(color, width, dash), name=label)

    def add_range(self, plot, x0, x1, color, alpha, label):
        """Shaded vertical band from x0 to x1 behind the curves, with a legend patch."""
        band = pg.LinearRegionItem(values=(x0, x1), brush=brush(color, alpha), pen=pg.mkPen(None), movable=False)
        band.setZValue(-10)
        plot.addItem(band)
        self.legends[plot].addItem(pg.PlotDataItem(pen=None, fillLevel=0, fillBrush=brush(color, 0.35)), label)

    # ---- depth profile ----
    def draw_profile(self, plot, values, thicknesses, z_pad, color, value_name=("value", "")):
        """Depth profile of one quantity: each slice drawn as a step from its top edge to its bottom edge."""
        edges = np.concatenate(([-z_pad], np.cumsum(thicknesses[:-1]), [thicknesses.sum() + z_pad]))
        self.box_of[plot].set_curve(edges, values, ("z", "Å") + tuple(value_name), steps=True)
        self.box_of[plot].add_series(value_name[0], edges, values)
        plot.setLogMode(x=False, y=False)
        plot.plot(edges, values, stepMode="center", pen=pen(color, 1.2), fillLevel=0.0,
                  fillBrush=brush(color, 0.15))
        span = values.max() - values.min()
        plot.setXRange(edges[0], edges[-1], padding=0)
        plot.setYRange(values.min() - 0.2 * span - 0.01, values.max() + 0.2 * span + 0.01, padding=0)


def framed(widget, margin=3):
    """The widget inside an outlined frame like the plot boxes (QFrame#PlotFrame)."""
    frame = QtWidgets.QFrame()
    frame.setObjectName("PlotFrame")
    layout = QtWidgets.QVBoxLayout(frame)
    layout.setContentsMargins(margin, margin, margin, margin)
    layout.addWidget(widget)
    return frame


# ==============================================================================
# Small plot of a material card
# ==============================================================================
class MiniPlot(pg.PlotWidget):
    """Thickness of a material in every period. Fixed view; the mouse wheel scrolls the list of cards."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(210, 135)
        self.setMouseEnabled(x=False, y=False)
        for name in ("bottom", "left"):
            self.getAxis(name).enableAutoSIPrefix(False)
        self.hideButtons()
        self.setMenuEnabled(False)

    def wheelEvent(self, event):
        event.ignore()                                       # let the scroll area of the cards scroll

    def draw(self, period, d, color, title):
        plot = self.getPlotItem()
        plot.clear()
        self.setBackground(colors["panel"])
        plot.plot(period, d, pen=pen(color, 1.4), symbol="o" if d.size <= 150 else None, symbolSize=4,
                  symbolPen=None, symbolBrush=brush(color))
        if np.any(d < 0):
            plot.addItem(pg.InfiniteLine(pos=0.0, angle=0, pen=pen(colors["warning"], 1.0, DASH), movable=False))
        style_plot(plot, colors["axes"], title, "period  (1 = top)", "d (Å)",
                   title_size="8pt", label_size="7pt", tick_size=6.5)
        low, high = min(float(d.min()), 0.0 if np.any(d < 0) else float(d.min())), float(d.max())
        span = high - low if high > low else max(abs(high), 1.0)
        plot.setXRange(0.5, d.size + 0.5, padding=0)
        plot.setYRange(low - 0.1 * span, high + 0.1 * span, padding=0)
