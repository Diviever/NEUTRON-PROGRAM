"""
calculation.py — physics of the multilayer mirror (no GUI code)

    Material                 one material of the repeating unit: name, potential, ε, thickness law
    Interface                how boundaries are made rough: erf or linear transition, slicing, optional chaos
    build_stack(...)         METHOD 1:   materials + N + ambient + substrate + interface -> values[], thicknesses[]
                                         (values = neutron potentials V or X-ray permittivities ε of the same structure)
    neutron_reflectivity()   METHOD 2.1: potentials[], thicknesses[], q[]          ->  R(q) for neutrons
    xray_reflectivity()      METHOD 2.2: permittivities[], thicknesses[], q[], λ   ->  R(q) for X-rays
    energy_from_k(), k_from_energy(), glancing_angle(), q_from_angle()   neutron beam conversions
    xray_wavelength(), xray_energy()                                    X-ray beam conversions

Both the plots and the optimizer call only these methods.
Units: potential in µeV, thickness in Å, q and k in Å⁻¹, neutron energy in meV, X-ray energy in keV,
angle in degrees. ε is the complex dielectric permittivity (dimensionless, ε = n² ≈ 1 − 2δ + 2iβ).

Note: with real potentials (no absorption) the neutron R depends only on q = 2·k·sin θ, not on the
beam energy itself. The energy decides at which glancing angle θ a given q is reached.
"""
import math

import numpy as np

erf = np.vectorize(math.erf, otypes=[float])              # error function for numpy arrays (no scipy needed)

# Physical constants (not parameters)
HBAR2_2M = 81.8042 / (4.0 * np.pi**2)      # ħ²/2mₙ = 2.0721 meV·Å²:   E = HBAR2_2M · k²
H_OVER_M = 3956.03                          # h/mₙ in m/s·Å:              v = H_OVER_M / λ
KZ2_PER_UEV = 1e-3 / HBAR2_2M               # kz² [Å⁻²] produced by 1 µeV:  kz² = (q/2)² − KZ2_PER_UEV · V
HC_KEV_A = 12.398419843                     # h·c in keV·Å:                λ = HC_KEV_A / E   (X-rays)


# ==============================================================================
# Material
# ==============================================================================
class Material:
    """
    One material of the repeating unit (e.g. Ni or Ti).

    name       str
    potential  float, µeV — neutron effective potential
    epsilon    complex   — X-ray dielectric permittivity ε = ε' + iε'' at the X-ray energy in use
    sigma      float, Å  — rms roughness of the UPPER boundary of every layer of this material
    linear     [d_start, d_end]   thickness (Å) at t = 0 (substrate side) and t = 1 (vacuum side)
    poly       [c0, c1, c2, ...]  d(t) = c0 + c1·t + c2·t² + ...   (Å)
    mode       "linear" or "poly" — which of the two arrays gives the thickness

    t = j / (N − 1) is the relative position of repeat j in the stack:
    j = 0 (t = 0) lies on the substrate, j = N − 1 (t = 1) next to the vacuum.
    """

    def __init__(self, name, potential, linear=(50.0, 50.0), degree=4, epsilon=1.0 + 0.0j, sigma=0.0):
        self.name = name
        self.potential = float(potential)
        self.epsilon = complex(epsilon)
        self.sigma = float(sigma)
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
    How the boundaries of the stack are made rough. The roughness σ itself belongs to the media
    (as in Multifitting): material.sigma is the rms width of the UPPER boundary of every layer of that
    material, substrate.sigma the width of the boundary between the last layer and the substrate.

    The nominal layers keep their thicknesses. Only the profile (V or ε) is blurred:
    the sharp jump at boundary z_b becomes the smooth step  jump · F((z − z_b) / σ_b).
    Steps of neighbouring boundaries simply add up, i.e. the sharp profile is convolved with
    the roughness distribution. So when σ reaches the layer thickness the contrast between the
    materials fades, and for σ ≫ period the stack becomes the average (fully intermixed) medium.

    profile   "erf"     F = error-function step (Gaussian roughness of rms width σ), complete at ±4σ
              "linear"  F = straight ramp of width √12·σ (same rms width as the erf), complete at ±√3σ
    slice_thickness   each transition (z_b − reach … z_b + reach) is cut into n = ⌈2·reach / d⌉ equal
              slices of thickness ≤ d (Å). Equal slices act as a grating with a false Bragg peak at
              q = 2π/d, so keep d ≤ π/(4·q_max). A boundary with σ = 0 stays a sharp step (no slices).
    chaos     0 … 1: every boundary gets its own width σ_b · (1 + chaos·u), u random in [−1, 1]
    seed      start of the random numbers — same seed, same structure (reproducible curves and gradients)
    """

    def __init__(self, profile="erf", slice_thickness=2.0, chaos=0.0, seed=1):
        self.profile = profile
        self.slice_thickness = float(slice_thickness)
        self.chaos = float(chaos)
        self.seed = int(seed)

    def reach(self, sigma_b):
        """Distance from the boundary beyond which the transition is complete."""
        return 4.0 * sigma_b if self.profile == "erf" else math.sqrt(3.0) * sigma_b

    def n_slices(self, sigma_b):
        """Number of slices across one transition of width σ_b (0 for a sharp boundary)."""
        return math.ceil(2.0 * self.reach(sigma_b) / self.slice_thickness) if sigma_b > 0 else 0

    def step_fraction(self, x, sigma_b):
        """How far (0 … 1) the transition has gone at distance x (array, Å) from the boundary."""
        if self.profile == "erf":
            return 0.5 * (1.0 + erf(x / (math.sqrt(2.0) * sigma_b)))
        return np.clip(0.5 + x / (math.sqrt(12.0) * sigma_b), 0.0, 1.0)


# ==============================================================================
# METHOD 1: the layer stack
# ==============================================================================
def build_stack(materials, N, ambient, substrate, interface, quantity="potential"):
    """
    Returns two arrays of equal length, ordered from the ambient medium (first element)
    to the substrate (last element):

        values       quantity = "potential": neutron potentials V (µeV)
                     quantity = "epsilon":   X-ray permittivities ε (complex)
        thicknesses  (Å)    — ambient and substrate are semi-infinite and get 0

    ambient, substrate: Material objects for the media above and below the stack
    (only their potential / epsilon and, for the substrate, sigma are used).
    Both quantities go through the same steps, so neutrons and X-rays see the same structure.

    1. Sharp stack: the unit cell (all materials in list order) repeated N times; every boundary
       gets the σ of the medium below it. Negative thicknesses from a polynomial are built as 0 Å.
    2. Rough stack: every boundary with σ > 0 is blurred (see Interface) and cut into
       Interface.n_slices(σ) slices; the flat middle of a layer stays one slice.
       Total slices ≈ Σ over boundaries of n_slices + one per layer middle.
    """
    # ---- 1. sharp stack: value of every medium, depth and roughness of every boundary ----
    values = [getattr(ambient, quantity)]
    boundaries, sigmas = [], []                             # top of each layer (then the substrate) and its σ
    z = 0.0
    for j in range(N - 1, -1, -1):                          # from the ambient side down to the substrate
        t = j / (N - 1) if N > 1 else 0.0
        for material in materials:
            boundaries.append(z)
            sigmas.append(material.sigma)                   # upper boundary of this layer
            values.append(getattr(material, quantity))      # material.potential or material.epsilon
            z += max(material.thickness(t), 0.0)
    boundaries.append(z)
    sigmas.append(substrate.sigma)
    values.append(getattr(substrate, quantity))

    # ---- 2. rough stack ----
    random = np.random.default_rng(interface.seed)
    widths = np.array(sigmas) * (1.0 + interface.chaos * random.uniform(-1.0, 1.0, len(boundaries)))
    widths = np.maximum(widths, 0.0)
    jumps = np.diff(values)                                 # jump of V (or ε) at every boundary

    # slice edges: all boundaries + n_slices(σ_b) equal slices across every rough transition
    edges = list(boundaries)
    for z_b, w in zip(boundaries, widths):
        if w > 0:
            r = interface.reach(w)
            edges.extend(np.linspace(z_b - r, z_b + r, interface.n_slices(w) + 1))
    edges = np.unique(np.round(edges, 6))
    centres = 0.5 * (edges[:-1] + edges[1:])

    # value at every slice centre: value_top + Σ jump_b · F((z − z_b)/σ_b)   (sharp boundary: plain step)
    smooth = np.full(centres.size, values[0], dtype=np.result_type(jumps, float))   # float for V, complex for ε
    passed = np.zeros(centres.size + 1, dtype=smooth.dtype) # steps that are complete from slice i on
    for z_b, w, jump in zip(boundaries, widths, jumps):
        r = interface.reach(w)
        lo, hi = np.searchsorted(centres, [z_b - r, z_b + r])
        if hi > lo:
            smooth[lo:hi] += jump * interface.step_fraction(centres[lo:hi] - z_b, w)
        passed[hi] += jump
    smooth += np.cumsum(passed)[:-1]

    values = np.concatenate(([values[0]], smooth, [values[-1]]))
    thicknesses = np.concatenate(([0.0], np.diff(edges), [0.0]))
    return values, thicknesses


# ==============================================================================
# METHOD 2.1: the neutron reflectivity
# ==============================================================================
def neutron_reflectivity(potentials, thicknesses, q):
    """
    Specular neutron reflectivity R(q) of the stack by the Parratt recursion.
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
# METHOD 2.2: the X-ray reflectivity
# ==============================================================================
def xray_reflectivity(epsilons, thicknesses, q, wavelength, polarization="s"):
    """
    Specular X-ray reflectivity R(q) of the stack — METHOD 2.1 with the potentials replaced by the
    dielectric permittivities, following Svechnikov, J. Appl. Cryst. 53 (2020), Eq. (1):

        r(z_j) = [r_j + r(z_j+1)·exp(2iχ_j+1·l_j+1)] / [1 + r_j·r(z_j+1)·exp(2iχ_j+1·l_j+1)],   r(z_N+1) = 0
        χ_j    = k·(ε_j − cos²φ)^½      z component of the wave number in medium j
        k = 2π/λ,  φ = glancing angle,  q = 2k·sin φ  (so cos²φ = 1 − (q/2k)²),  l_j = thickness of layer j

    r_j is the Fresnel coefficient of boundary j / j+1:
        s-polarization:  r_j = (χ_j − χ_j+1) / (χ_j + χ_j+1)
        p-polarization:  r_j = (ε_j+1·χ_j − ε_j·χ_j+1) / (ε_j+1·χ_j + ε_j·χ_j+1)
    (Since (q/2)² = k² sin²φ, χ_j² = (q/2)² − k²(1 − ε_j): the neutron formula with K·V replaced by k²(1 − ε).)
    """
    q = np.asarray(q, dtype=float)
    k = 2.0 * np.pi / wavelength
    cos2_phi = 1.0 - (q / (2.0 * k)) ** 2

    def chi(eps):
        return k * np.sqrt(eps - cos2_phi + 0j)

    X = np.zeros(q.size, dtype=complex)                     # r(z_N+1) = 0: no reflection from inside the substrate
    chi_lower, eps_lower = chi(epsilons[-1]), epsilons[-1]
    for j in range(len(epsilons) - 2, -1, -1):
        chi_upper, eps_upper = chi(epsilons[j]), epsilons[j]
        if polarization == "p":
            r = (eps_lower * chi_upper - eps_upper * chi_lower) / (eps_lower * chi_upper + eps_upper * chi_lower)
        else:
            r = (chi_upper - chi_lower) / (chi_upper + chi_lower)
        phase = np.exp(2j * chi_lower * thicknesses[j + 1])  # round trip through layer j+1
        X = (r + X * phase) / (1.0 + r * X * phase)
        chi_lower, eps_lower = chi_upper, eps_upper
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


# ==============================================================================
# X-ray beam: photon energy <-> wavelength, critical angle of a material
# ==============================================================================
def xray_wavelength(E_keV):
    """λ (Å) of a photon with energy E (keV)."""
    return HC_KEV_A / E_keV


def xray_energy(wavelength):
    """E (keV) of a photon with wavelength λ (Å)."""
    return HC_KEV_A / wavelength


def xray_critical_q(epsilon, wavelength):
    """q_c (Å⁻¹) of total external reflection for X-rays: sin²φ_c = 1 − Re ε."""
    return 4.0 * np.pi / wavelength * np.sqrt(max(1.0 - complex(epsilon).real, 0.0))
