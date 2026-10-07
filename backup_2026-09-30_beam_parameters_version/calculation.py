"""
calculation.py — physics of the multilayer mirror (no GUI code)

    Material                 one material of the repeating unit: name, potential, thickness law
    build_stack(...)         METHOD 1: materials + N + V_vac + V_sub  ->  potentials[], thicknesses[]
    reflectivity(...)        METHOD 2: potentials[], thicknesses[], q[] ->  R(q)
    energy_from_k(), k_from_energy(), glancing_angle(), q_from_angle()   neutron beam conversions

Both the plots and the optimizer call only these two methods.
Units: potential in µeV, thickness in Å, q and k in Å⁻¹, energy in meV, angle in mrad.

Note: with real potentials (no absorption) R depends only on q = 2·k·sin θ, not on the
beam energy itself. The energy decides at which glancing angle θ a given q is reached.
"""
import numpy as np

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
# METHOD 1: the layer stack
# ==============================================================================
def build_stack(materials, N, V_vac, V_sub, n_sub=5, d_sub=1.5):
    """
    Returns two arrays of equal length, ordered from the vacuum (first element)
    to the substrate (last element):

        potentials   (µeV)
        thicknesses  (Å)    — vacuum and substrate are semi-infinite and get 0

    The unit cell (all materials in list order) is repeated N times.
    Between any two neighbouring media, n_sub sublayers of thickness d_sub
    change the potential in linear steps (interface roughness / interdiffusion).
    Negative thicknesses from a polynomial are built as 0 Å.
    """
    potentials = [V_vac]
    thicknesses = [0.0]

    def add_layer(V, d):
        V_above = potentials[-1]
        for s in range(1, n_sub + 1):                       # graded interface
            potentials.append(V_above + (V - V_above) * s / (n_sub + 1))
            thicknesses.append(d_sub)
        potentials.append(V)
        thicknesses.append(d)

    for j in range(N - 1, -1, -1):                          # from the vacuum side down to the substrate
        t = j / (N - 1) if N > 1 else 0.0
        for material in materials:
            add_layer(material.potential, max(material.thickness(t), 0.0))
    add_layer(V_sub, 0.0)                                   # substrate

    return np.array(potentials), np.array(thicknesses)


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
    """Glancing angle θ (mrad) at which momentum transfer q is reached:  q = 2·k·sin θ."""
    return 1e3 * np.arcsin(np.clip(np.asarray(q) / (2.0 * k), -1.0, 1.0))


def q_from_angle(theta_mrad, k):
    """Momentum transfer q (Å⁻¹) at glancing angle θ (mrad)."""
    return 2.0 * k * np.sin(np.asarray(theta_mrad) * 1e-3)
