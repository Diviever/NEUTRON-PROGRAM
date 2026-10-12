"""
optimizer.py — supermirror optimizer (no GUI code)

    reference_for_m(m)        typical layer count and reflectivity of real supermirrors
    hayter_mook_start(...)    optional start profile for the materials
    GradientDescent           changes material.poly IN PLACE to raise ⟨R⟩ on 0 … m·q_c
    optimize(...)             the full search: gradient descent at N, then larger N, until ⟨R⟩ ≥ target

Every structure is evaluated with calculation.build_stack() and calculation.neutron_reflectivity().
"""
import math

import numpy as np

from calculation import build_stack, neutron_reflectivity, mean_reflectivity, critical_q


# ==============================================================================
# Reference values of real supermirrors (start values of the editable table in the app)
# ==============================================================================
#            m   total layers  R_low  R_high
M_REFERENCE = [
    [1.0,      1,  0.98, 1.00],     # plain Ni mirror (assumed anchor)
    [2.0,    150,  0.92, 0.95],
    [3.0,    400,  0.88, 0.91],
    [4.0,   1000,  0.80, 0.85],
    [6.0,   4000,  0.65, 0.72],
    [7.0,   8000,  0.55, 0.60],
]


def reference_for_m(m, table):
    """(total layers, R_low, R_high) for any m from `table` (rows m, layers, R_low, R_high, sorted by m):
    interpolated inside the table, the last segment continued above it."""
    ms, layers, r_low, r_high = (np.array(col, dtype=float) for col in zip(*table))
    if m <= ms[-1]:
        n_layers = np.exp(np.interp(np.log(m), np.log(ms), np.log(layers)))
        return float(n_layers), float(np.interp(m, ms, r_low)), float(np.interp(m, ms, r_high))
    # beyond the table: continue the last segment
    slope = np.log(layers[-1] / layers[-2]) / np.log(ms[-1] / ms[-2])
    n_layers = layers[-1] * (m / ms[-1]) ** slope
    lo = r_low[-1] + (r_low[-1] - r_low[-2]) / (ms[-1] - ms[-2]) * (m - ms[-1])
    hi = r_high[-1] + (r_high[-1] - r_high[-2]) / (ms[-1] - ms[-2]) * (m - ms[-1])
    return float(n_layers), max(float(lo), 0.0), max(float(hi), 0.0)


# ==============================================================================
# Optional start profile
# ==============================================================================
def hayter_mook_start(materials, m, q_c):
    """
    Puts every material into poly mode with a Hayter–Mook-type thickness law (degree is kept).

    The bilayer at position t reflects at q(t), with q⁴ spread evenly between
    (m·q_c)⁴ at the substrate (t = 0) and q_c⁴ at the vacuum side (t = 1).
    Bragg condition with refraction:  Λ(t) = 2π / sqrt(q(t)² − q_c,avg²),
    q_c,avg from the mean potential. Λ is shared equally between the materials,
    and the curve is fitted by each material's polynomial (least squares).
    """
    t = np.linspace(0.0, 1.0, 200)
    q_t = (q_c**4 + (1.0 - t) * ((m * q_c)**4 - q_c**4)) ** 0.25
    q_c_avg = critical_q(np.mean([mat.potential for mat in materials]))
    bilayer = 2.0 * np.pi / np.sqrt(np.maximum(q_t**2 - q_c_avg**2, 1e-10))
    for mat in materials:
        mat.poly = np.polynomial.polynomial.polyfit(t, bilayer / len(materials), len(mat.poly) - 1)
        mat.mode = "poly"


# ==============================================================================
# Gradient descent (Adam) on the polynomial coefficients
# ==============================================================================
class GradientDescent:
    """
    Minimizes   loss = 1 − ⟨R⟩ + penalty_weight · (penalty for layers thinner than d_min)
    over all coefficients material.poly of all materials.

    The materials are changed IN PLACE — they are the same objects the GUI shows.
    Adam: every coefficient moves about `step` Å per iteration in the direction of its
    smoothed gradient, whatever the size of that gradient.
    """

    def __init__(self, materials, N, ambient, substrate, interface, q, q_top,
                 step, step_decay, d_min, penalty_weight, h, beta1, beta2):
        self.materials = materials
        self.N = N
        self.ambient, self.substrate = ambient, substrate   # Material objects above and below the stack
        self.interface = interface            # calculation.Interface: how the layers are joined
        self.q, self.q_top = q, q_top
        self.step = step                      # Å, Adam step size ...
        self.step_decay = step_decay          # ... shrinking as step / (1 + iteration / step_decay)
        self.d_min = d_min                    # Å, thinner layers are penalized
        self.penalty_weight = penalty_weight  # weight of that penalty in the loss
        self.h = h                            # Å, coefficient change used to measure the gradient
        self.beta1, self.beta2 = beta1, beta2 # Adam memory of the gradient and of its square
        self.mean_R = 0.0                     # ⟨R⟩ of the last evaluated structure
        self.set_N(N)

    def set_N(self, N):
        """New number of repeats; the Adam memory starts again from zero."""
        self.N = N
        self.iteration = 0
        self.avg_grad = [np.zeros_like(mat.poly) for mat in self.materials]
        self.avg_grad_sq = [np.zeros_like(mat.poly) for mat in self.materials]

    def loss(self):
        """Evaluate the current materials (METHOD 1 + METHOD 2) and return the loss. Also sets self.mean_R."""
        potentials, thicknesses, roughness = build_stack(self.materials, self.N, self.ambient, self.substrate,
                                                         self.interface)
        R = neutron_reflectivity(potentials, thicknesses, self.q, roughness)
        self.mean_R = mean_reflectivity(R, self.q, self.q_top)

        too_thin = 0.0
        if self.d_min > 0:
            t = np.linspace(0.0, 1.0, self.N)
            for mat in self.materials:
                too_thin += np.mean(np.clip(self.d_min - mat.thickness(t), 0.0, None) ** 2) / self.d_min**2
        return 1.0 - self.mean_R + self.penalty_weight * too_thin

    def gradient(self, material, loss_now):
        """d(loss)/d(coefficient) for every coefficient of one material (forward difference)."""
        grad = np.zeros_like(material.poly)
        for k in range(len(material.poly)):
            old = material.poly[k]
            material.poly[k] = old + self.h
            grad[k] = (self.loss() - loss_now) / self.h
            material.poly[k] = old
        return grad

    def make_step(self, loss_now):
        """One Adam step. loss_now = self.loss() of the current coefficients."""
        mean_R = self.mean_R                                   # gradient() overwrites it
        grads = [self.gradient(mat, loss_now) for mat in self.materials]
        self.mean_R = mean_R

        self.iteration += 1
        step = self.step / (1.0 + self.iteration / self.step_decay)
        b1, b2 = self.beta1, self.beta2
        for mat, grad, avg, avg_sq in zip(self.materials, grads, self.avg_grad, self.avg_grad_sq):
            avg[:] = b1 * avg + (1 - b1) * grad                # smoothed gradient
            avg_sq[:] = b2 * avg_sq + (1 - b2) * grad**2       # smoothed squared gradient
            avg_hat = avg / (1 - b1**self.iteration)           # bias correction of the first steps
            avg_sq_hat = avg_sq / (1 - b2**self.iteration)
            mat.poly -= step * avg_hat / (np.sqrt(avg_sq_hat) + 1e-12)


# ==============================================================================
# The full search
# ==============================================================================
def optimize(gd, target, N_start, N_max, growth, max_iter, patience, min_gain, on_step):
    """
    0. Materials in linear or user mode are switched to poly mode (straight line / least-squares fit).
    1. Gradient descent at the current N until ⟨R⟩ ≥ target (done), or until ⟨R⟩ has not
       improved by more than min_gain for `patience` iterations, or max_iter iterations are used.
    2. Otherwise keep the best coefficients of this N, increase N (× growth) and go to 1.

    on_step(N, iteration, mean_R, best_R) is called every iteration (GUI refresh);
    returning False stops the search.
    Returns "success", "n_max" or "stopped". The result is left in the materials and in gd.N.
    """
    for mat in gd.materials:
        if mat.mode != "poly":                  # linear and user thicknesses become a polynomial to optimize
            mat.switch_to_poly()
    gd.set_N(N_start)

    while True:
        best_R, best_poly, no_gain = -1.0, None, 0
        status = "next N"
        for iteration in range(1, max_iter + 1):
            loss_now = gd.loss()
            if gd.mean_R > best_R + min_gain:
                best_R, best_poly, no_gain = gd.mean_R, [mat.poly.copy() for mat in gd.materials], 0
            else:
                no_gain += 1

            if not on_step(gd.N, iteration, gd.mean_R, best_R):
                status = "stopped"
                break
            if gd.mean_R >= target:
                return "success"
            if no_gain >= patience:
                break
            gd.make_step(loss_now)

        for mat, poly in zip(gd.materials, best_poly):           # keep the best coefficients of this N
            mat.poly = poly
        if status == "stopped":
            return "stopped"

        next_N = max(gd.N + 1, math.ceil(gd.N * growth))
        if next_N > N_max:
            return "n_max"
        gd.set_N(next_N)
