"""
plots.py — the plots of the program, drawn with pyqtgraph

pyqtgraph draws directly with Qt, so redrawing is fast (also during the optimization). In every plot:
drag = move, mouse wheel = zoom, right-click = menu with "View All" and "Export…" (PNG, SVG, CSV of the
curves), the small "A" button in the corner = back to the full view.

    AngleAxis       top axis of an R(q) plot: ticks at round glancing angles θ (°) of that beam, q = 2k·sin θ
    PlotPanel       the four plots of the main window: neutron R(q), X-ray R(q), V(z), 1 − Re ε(z)
                        draw_reflectivity()  R(q) on a log or linear axis + measured points
                        add_vline(), add_segment(), add_range()   marks with a legend line
                        draw_profile()       depth profile as steps
    MiniPlot        small thickness plot of a material card

Colors come from gui_style.colors (the current theme).
"""
import numpy as np

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui
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
class PlotPanel(pg.GraphicsLayoutWidget):
    """Neutron R(q) and X-ray R(q) on top, the depth profiles V(z) and 1 − Re ε(z) below."""

    def __init__(self):
        super().__init__()
        self.neutron = self.addPlot(row=0, col=0, axisItems={"top": AngleAxis()})
        self.xray = self.addPlot(row=0, col=1, axisItems={"top": AngleAxis()})
        self.potential = self.addPlot(row=1, col=0)
        self.epsilon = self.addPlot(row=1, col=1)
        self.legends = {}
        for plot in (self.neutron, self.xray):
            plot.showAxis("top")
            plot.setDownsampling(auto=True, mode="peak")    # long curves: draw only what the pixels can show
            plot.setClipToView(True)
            self.legends[plot] = plot.addLegend(labelTextSize="7pt")
        self.apply_theme()

    def apply_theme(self):
        self.setBackground(colors["figure"])

    def clear_all(self):
        for plot in (self.neutron, self.xray, self.potential, self.epsilon):
            plot.clear()
        for legend in self.legends.values():
            legend.clear()

    # ---- R(q) ----
    def draw_reflectivity(self, plot, q, R, k, color, log_scale, log_decades, background=0.0,
                          measured=None, measured_name=""):
        """
        R(q) curve, with the glancing angle of this beam (k) on the top axis.
        log_scale: the axis follows the curves, from the decade above max R down to the decade of min R,
                   but at most log_decades decades.
        background is added to the drawn curve only (Multifitting's 'Background').
        measured = (q[], R[]) of a measured curve, drawn as points (None = no file).
        """
        R = R + background
        label = "Parratt" if background == 0 else f"Parratt + background {background:.3g}"
        shown = R if measured is None else np.concatenate((R, measured[1]))    # what the y axis must hold
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
        curve = plot.plot(q, R, pen=pen(color, 1.4), name=label)
        curve.setZValue(1)                                                       # the curve over the points
        if measured is not None:
            q_m, R_m = measured
            keep = R_m > 0 if log_scale else np.isfinite(R_m)                   # log axis: no points ≤ 0
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
    def draw_profile(self, plot, values, thicknesses, z_pad, color):
        """Depth profile of one quantity: each slice drawn as a step from its top edge to its bottom edge."""
        edges = np.concatenate(([-z_pad], np.cumsum(thicknesses[:-1]), [thicknesses.sum() + z_pad]))
        plot.setLogMode(x=False, y=False)
        plot.plot(edges, values, stepMode="center", pen=pen(color, 1.2), fillLevel=0.0,
                  fillBrush=brush(color, 0.15))
        span = values.max() - values.min()
        plot.setXRange(edges[0], edges[-1], padding=0)
        plot.setYRange(values.min() - 0.2 * span - 0.01, values.max() + 0.2 * span + 0.01, padding=0)


# ==============================================================================
# Small plot of a material card
# ==============================================================================
class MiniPlot(pg.PlotWidget):
    """Thickness of a material in every period. Fixed view; the mouse wheel scrolls the list of cards."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(240, 150)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.setMenuEnabled(False)

    def wheelEvent(self, event):
        event.ignore()                                       # let the scroll area of the cards scroll

    def draw(self, period, d, color, title):
        plot = self.getPlotItem()
        plot.clear()
        self.setBackground(colors["card_plot"])
        plot.plot(period, d, pen=pen(color, 1.4), symbol="o" if d.size <= 150 else None, symbolSize=4,
                  symbolPen=None, symbolBrush=brush(color))
        if np.any(d < 0):
            plot.addItem(pg.InfiniteLine(pos=0.0, angle=0, pen=pen(colors["warning"], 1.0, DASH), movable=False))
        style_plot(plot, colors["card_plot"], title, "period  (1 = top)", "d (Å)",
                   title_size="8pt", label_size="7pt", tick_size=6.5)
        low, high = min(float(d.min()), 0.0 if np.any(d < 0) else float(d.min())), float(d.max())
        span = high - low if high > low else max(abs(high), 1.0)
        plot.setXRange(0.5, d.size + 0.5, padding=0)
        plot.setYRange(low - 0.1 * span, high + 0.1 * span, padding=0)
