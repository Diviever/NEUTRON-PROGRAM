"""
calculation.py — physics of the multilayer mirror (no GUI code)

    Material                 one material of the repeating unit: name, potential, thickness law
    Interface                roughness of every boundary: smooth erf or linear transition of width σ, optional chaos
    build_stack(...)         METHOD 1: materials + N + V_vac + V_sub + interface ->  potentials[], thicknesses[]
    reflectivity(...)        METHOD 2: potentials[], thicknesses[], q[] ->  R(q)
    energy_from_k(), k_from_energy(), glancing_angle(), q_from_angle()   neutron beam conversions

Both the plots and the optimizer call only these two methods.
Units: potential in µeV, thickness in Å, q and k in Å⁻¹, energy in meV, angle in degrees.

Note: with real potentials (no absorption) R depends only on q = 2·k·sin θ, not on the
beam energy itself. The energy decides at which glancing angle θ a given q is reached.
"""
import math

import numpy as np

erf = np.vectorize(math.erf, otypes=[float])              # error function for numpy arrays (no scipy needed)

# Physical constants (not parameters)
HBAR2_2M = 81.8042 / (4.0 * np.pi**2)      # ħ²/2mₙ = 2.0721 meV·Å²:   E = HBAR2_2M · k²
H_OVER_M = 3956.03                          # h/mₙ in m/s·Å:              v = H_OVER_M / λ
KZ2_PER_UEV = 1e-3 / HBAR2_2M               # kz² [Å⁻²] produced by 1 µeV:  kz² = (q/2)² − KZ2_PER_UEV · V


# ==============================================================================
# Material
# ==============================================================================
class Material:
    """
    One material of the repeating unit (e.g. Ni or Ti).

    name       str
    potential  float, µeV
    linear     [d_start, d_end]   thickness (Å) at t = 0 (substrate side) and t = 1 (vacuum side)
    poly       [c0, c1, c2, ...]  d(t) = c0 + c1·t + c2·t² + ...   (Å)
    mode       "linear" or "poly" — which of the two arrays gives the thickness

    t = j / (N − 1) is the relative position of repeat j in the stack:
    j = 0 (t = 0) lies on the substrate, j = N − 1 (t = 1) next to the vacuum.
    """

    def __init__(self, name, potential, linear=(50.0, 50.0), degree=4):
        self.name = name
        self.potential = float(potential)
        self.linear = np.array(linear, dtype=float)
        self.poly = np.zeros(degree + 1)
        self.mode = "linear"

    def thickness(self, t):
        """Thickness at relative position t (a number or an array of numbers)."""
        if self.mode == "linear":
            return self.linear[0] + (self.linear[1] - self.linear[0]) * t
        return np.polynomial.polynomial.polyval(t, self.poly)

    def switch_to_poly(self):
        """Poly mode that starts from the current straight line: c0 = d_start, c1 = d_end − d_start."""
        self.poly[:] = 0.0
        self.poly[0] = self.linear[0]
        self.poly[1] = self.linear[1] - self.linear[0]
        self.mode = "poly"

    def set_degree(self, degree):
        """Change the polynomial degree, keeping the existing coefficients."""
        new_poly = np.zeros(degree + 1)
        n = min(len(self.poly), len(new_poly))
        new_poly[:n] = self.poly[:n]
        self.poly = new_poly


# ==============================================================================
# Interface roughness / interdiffusion
# ==============================================================================
class Interface:
    """
    Roughness of every boundary of the stack (also vacuum/first layer and last layer/substrate).

    The nominal layers keep their thicknesses. Only the potential profile is blurred:
    the sharp step ΔV at boundary z_b becomes the smooth step  ΔV · F((z − z_b) / σ_b).
    Steps of neighbouring boundaries simply add up, i.e. the sharp profile is convolved with
    the roughness distribution. So when σ reaches the layer thickness the contrast between the
    materials fades, and for σ ≫ period the stack becomes the average (fully intermixed) medium
    without any Bragg peaks.

    profile   "erf"     F = error-function step (Gaussian roughness of rms width σ)
              "linear"  F = straight ramp of width √12·σ (same rms width as the erf)
    sigma     rms width σ (Å) of every transition; 0 = sharp boundaries
    n_slices  the smooth profile is cut into this many slices across each transition
              (erf: ±4σ, linear: the ramp) to be used by the Parratt calculation
    chaos     0 … 1: every boundary gets its own width σ_b = σ · (1 + chaos·u), u random in [−1, 1]
    seed      start of the random numbers — same seed, same structure (reproducible curves and gradients)
    """

    def __init__(self, profile="erf", sigma=3.0, n_slices=10, chaos=0.0, seed=1):
        self.profile = profile
        self.sigma = float(sigma)
        self.n_slices = int(n_slices)
        self.chaos = float(chaos)
        self.seed = int(seed)

    def reach(self, sigma_b):
        """Distance from the boundary beyond which the transition is complete."""
        return 4.0 * sigma_b if self.profile == "erf" else math.sqrt(3.0) * sigma_b

    def step_fraction(self, x, sigma_b):
        """How far (0 … 1) the transition has gone at distance x (array, Å) from the boundary."""
        if self.profile == "erf":
            return 0.5 * (1.0 + erf(x / (math.sqrt(2.0) * sigma_b)))
        return np.clip(0.5 + x / (math.sqrt(12.0) * sigma_b), 0.0, 1.0)


# ==============================================================================
# METHOD 1: the layer stack
# ==============================================================================
def build_stack(materials, N, V_vac, V_sub, interface):
    """
    Returns two arrays of equal length, ordered from the vacuum (first element)
    to the substrate (last element):

        potentials   (µeV)
        thicknesses  (Å)    — vacuum and substrate are semi-infinite and get 0

    1. Sharp stack: the unit cell (all materials in list order) repeated N times;
       negative thicknesses from a polynomial are built as 0 Å.
    2. Rough stack (σ > 0): the profile is blurred at every boundary (see Interface)
       and cut into slices: fine slices across the transitions, one slice for the
       flat middle of a layer.
    """
    # ---- 1. sharp stack: potential of every medium and depth of every boundary ----
    potentials = [V_vac]
    boundaries = []                                         # depth of the top of each layer, then the substrate
    z = 0.0
    for j in range(N - 1, -1, -1):                          # from the vacuum side down to the substrate
        t = j / (N - 1) if N > 1 else 0.0
        for material in materials:
            boundaries.append(z)
            potentials.append(material.potential)
            z += max(material.thickness(t), 0.0)
    boundaries.append(z)
    potentials.append(V_sub)

    if interface.sigma <= 0:
        thicknesses = [0.0] + list(np.diff(boundaries)) + [0.0]
        return np.array(potentials), np.array(thicknesses)

    # ---- 2. rough stack ----
    random = np.random.default_rng(interface.seed)
    widths = interface.sigma * (1.0 + interface.chaos * random.uniform(-1.0, 1.0, len(boundaries)))
    widths = np.maximum(widths, 1e-3)                       # chaos = 1 may give 0: keep a tiny width
    jumps = np.diff(potentials)                             # ΔV at every boundary

    # slice edges: all boundaries + n_slices steps across every transition
    edges = list(boundaries)
    for z_b, w in zip(boundaries, widths):
        r = interface.reach(w)
        edges.extend(np.linspace(z_b - r, z_b + r, interface.n_slices + 1))
    edges = np.unique(np.round(edges, 6))
    centres = 0.5 * (edges[:-1] + edges[1:])

    # smooth potential at every slice centre: V_vac + Σ ΔV_b · F((z − z_b)/σ_b)
    V = np.full(centres.size, float(V_vac))
    passed = np.zeros(centres.size + 1)                     # steps that are complete from slice i on
    for z_b, w, dV in zip(boundaries, widths, jumps):
        r = interface.reach(w)
        lo, hi = np.searchsorted(centres, [z_b - r, z_b + r])
        V[lo:hi] += dV * interface.step_fraction(centres[lo:hi] - z_b, w)
        passed[hi] += dV
    V += np.cumsum(passed)[:-1]

    potentials = np.concatenate(([V_vac], V, [V_sub]))
    thicknesses = np.concatenate(([0.0], np.diff(edges), [0.0]))
    return potentials, thicknesses


# ==============================================================================
# METHOD 2: the reflectivity
# ==============================================================================
def reflectivity(potentials, thicknesses, q):
    """
    Specular reflectivity R(q) of the stack by the Parratt recursion.
    Starts inside the substrate (no reflected wave) and works up to the vacuum.
    """
    q = np.asarray(q, dtype=float)

    def kz(V):
        return np.sqrt((q / 2.0) ** 2 - KZ2_PER_UEV * V + 0j)

    X = np.zeros(q.size, dtype=complex)                     # reflection amplitude below the current interface
    kz_lower = kz(potentials[-1])
    for i in range(len(potentials) - 2, -1, -1):
        kz_upper = kz(potentials[i])
        r = (kz_upper - kz_lower) / (kz_upper + kz_lower)   # Fresnel coefficient of interface i / i+1
        phase = np.exp(2j * kz_lower * thicknesses[i + 1])  # round trip through layer i+1
        X = (r + X * phase) / (1.0 + r * X * phase)
        kz_lower = kz_upper
    return np.abs(X) ** 2


# ==============================================================================
# Small helpers used by the plots and the optimizer
# ==============================================================================
def critical_q(potential):
    """Critical momentum transfer q_c (Å⁻¹) of a material with the given potential (µeV)."""
    return 2.0 * np.sqrt(KZ2_PER_UEV * max(potential, 0.0))


def q_grid(q_top, q_min, q_max_factor, points):
    """q values used everywhere (plots and optimizer): q_min … q_max_factor · q_top, q_top = m·q_c."""
    return np.linspace(q_min, q_max_factor * q_top, int(points))


def mean_reflectivity(R, q, q_top):
    """⟨R⟩: mean reflectivity of the points with q ≤ q_top (nan if there are none)."""
    inside = R[q <= q_top]
    return float(np.mean(inside)) if inside.size else float("nan")


# ==============================================================================
# Neutron beam: energy E, wave vector k = p/ħ, wavelength, speed, glancing angle
# ==============================================================================
def energy_from_k(k):
    """E (meV) of a neutron with wave vector k (Å⁻¹)."""
    return HBAR2_2M * k**2


def k_from_energy(E):
    """k (Å⁻¹) of a neutron with energy E (meV)."""
    return np.sqrt(E / HBAR2_2M)


def glancing_angle(q, k):
    """Glancing angle θ (degrees) at which momentum transfer q is reached:  q = 2·k·sin θ."""
    return np.degrees(np.arcsin(np.clip(np.asarray(q) / (2.0 * k), -1.0, 1.0)))


def q_from_angle(theta_deg, k):
    """Momentum transfer q (Å⁻¹) at glancing angle θ (degrees)."""
    return 2.0 * k * np.sin(np.radians(np.asarray(theta_deg)))
