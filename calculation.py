"""
calculation.py — physics of the multilayer mirror (no GUI code)

    Material                 one material of the repeating unit: name, potential, ε, thickness law,
                             roughness σ and profile (erf / lin / exp mix) of its upper boundary
    step_fraction(), reach() the shape of one rough boundary: weighted mix of erf, linear and exponential steps
    profile_transform(), roughness_factors()   the same shape in q space: Fourier transform of the step's
                             derivative, and the factors it gives a boundary's reflection and transmission
    Interface                how roughness is calculated ("factor" or "slices"), slicing, optional chaos
    build_stack(...)         METHOD 1:   materials + N + ambient + substrate + interface
                                         -> values[], thicknesses[], roughness
                                         (values = neutron potentials V or X-ray permittivities ε of the same structure)
    neutron_reflectivity()   METHOD 2.1: potentials[], thicknesses[], q[], roughness          ->  R(q) for neutrons
    xray_reflectivity()      METHOD 2.2: permittivities[], thicknesses[], q[], λ, roughness   ->  R(q) for X-rays

Two ways to include the roughness (Interface.method):
    "factor"  sharp layers; the reflection and transmission of every boundary are multiplied by factors made
              from the Fourier transform of its roughness profile (convolution in z = multiplication in q).
              Fast: one step per layer. The idea of Multifitting (Svechnikov 2020, Eq. 2), here with the
              Névot–Croce / Vidal–Vincent form. Good while σ is well below the layer thicknesses.
    "slices"  the rough profile itself (convolution of the sharp profile with the roughness), cut into thin
              slices. Exact also when neighbouring transitions overlap or the layers intermix, but slow:
              thousands of slices instead of one step per layer.
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
    One material of the repeating unit (e.g. Ni or Ti). Every structure is N periods of the materials
    in table order; each material has its own thickness in every period (its thickness array).

    name       str
    potential  float, µeV — neutron effective potential
    epsilon    complex   — X-ray dielectric permittivity ε = ε' + iε'' at the X-ray energy in use
    sigma      float, Å  — rms roughness of the UPPER boundary of every layer of this material
    profile    [w_erf, w_lin, w_exp], each 0 … 1 — shape of that boundary: mix of the three steps
               (see step_fraction; [1, 0, 0] = pure erf, Gaussian roughness)
    mode       "linear", "poly" or "user" — which of the three descriptions gives the thicknesses:
    linear     [d_start, d_end]   thickness (Å) at t = 0 (substrate side) and t = 1 (top side)
    poly       [c0, c1, c2, ...]  d(t) = c0 + c1·t + c2·t² + ...   (Å)
    user       [d_1, d_2, ...]    thickness (Å) in every period, TOP period first (e.g. from a structure file)

    thicknesses(N) gives the array actually used: the thickness in each of the N periods, top period first.
    t is the relative position of a period: t = 1 for the top period, t = 0 for the one on the substrate.
    """

    def __init__(self, name, potential, linear=(50.0, 50.0), degree=4, epsilon=1.0 + 0.0j, sigma=0.0,
                 profile=(1.0, 0.0, 0.0)):
        self.name = name
        self.potential = float(potential)
        self.epsilon = complex(epsilon)
        self.sigma = float(sigma)
        self.profile = np.array(profile, dtype=float)       # weights of erf, lin, exp
        self.linear = np.array(linear, dtype=float)
        self.poly = np.zeros(degree + 1)
        self.user = np.zeros(0)
        self.mode = "linear"

    def thickness(self, t):
        """Thickness at relative position t (a number or an array of numbers)."""
        if self.mode == "linear":
            return self.linear[0] + (self.linear[1] - self.linear[0]) * t
        if self.mode == "poly":
            return np.polynomial.polynomial.polyval(t, self.poly)
        values = self.user[::-1]                            # user values from the substrate (t = 0) up
        if values.size < 2:
            return (values[0] if values.size else 0.0) + 0.0 * np.asarray(t)
        return np.interp(t, np.linspace(0.0, 1.0, values.size), values)

    def thicknesses(self, N):
        """Thickness array: one value per period, top period first. A user array of length N is used as it is;
        for any other N (and for the linear / polynomial laws) d(t) is sampled at t = 1 … 0."""
        if self.mode == "user" and self.user.size == N:
            return self.user.copy()
        t = np.linspace(1.0, 0.0, N) if N > 1 else np.zeros(1)
        return np.asarray(self.thickness(t), dtype=float) * np.ones(N)

    def switch_to_poly(self):
        """Poly mode that starts from the current thicknesses: the straight line (c0 = d_start,
        c1 = d_end − d_start), or a least-squares fit of the user array."""
        degree = len(self.poly) - 1
        self.poly[:] = 0.0
        if self.mode == "user" and self.user.size > 1:
            t = np.linspace(1.0, 0.0, self.user.size)
            fit = np.polynomial.polynomial.polyfit(t, self.user, min(degree, self.user.size - 1))
            self.poly[:fit.size] = fit
        elif self.mode == "user":
            self.poly[0] = self.user[0] if self.user.size else 0.0
        else:
            self.poly[0] = self.linear[0]
            self.poly[1] = self.linear[1] - self.linear[0]
        self.mode = "poly"

    def switch_to_user(self, N):
        """User mode. A user array that is already there (e.g. from a file) is kept;
        otherwise it starts from the current thicknesses of the N periods."""
        if self.user.size == 0:
            self.user = np.maximum(self.thicknesses(N), 0.0)
        self.mode = "user"

    def set_degree(self, degree):
        """Change the polynomial degree, keeping the existing coefficients."""
        new_poly = np.zeros(degree + 1)
        n = min(len(self.poly), len(new_poly))
        new_poly[:n] = self.poly[:n]
        self.poly = new_poly


# ==============================================================================
# Shape of one rough boundary (as in Multifitting, Svechnikov 2020, Eq. 3 and Table 2)
# ==============================================================================
PROFILE_NAMES = ("erf", "lin", "exp")
REACH = (4.0, math.sqrt(3.0), 7.0)          # each step is complete (to ~3·10⁻⁵) at ±REACH·σ


def profile_weights(weights):
    """The weights [erf, lin, exp] made ≥ 0 and normalized to a sum of 1 (all zero -> pure erf)."""
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    return w / w.sum() if w.sum() > 0 else np.array([1.0, 0.0, 0.0])


def step_fraction(x, sigma, weights):
    """
    How far (0 … 1) a boundary of rms width σ has gone at distance x (array, Å) from its centre.
    All three steps have the same rms width σ; the boundary is their weighted mix:

        erf   ½·(1 + erf(x / √2σ))                       Gaussian roughness, smooth
        lin   straight ramp from −√3σ to +√3σ            uniform mixing zone
        exp   ½·exp(√2·x/σ) for x ≤ 0,                   steep in the middle with long tails
              1 − ½·exp(−√2·x/σ) for x > 0

        F(x) = (w_erf·erf + w_lin·lin + w_exp·exp) / (w_erf + w_lin + w_exp)
    """
    w = profile_weights(weights)
    x = np.asarray(x, dtype=float)
    F = np.zeros(x.shape)
    if w[0] > 0:
        F += w[0] * 0.5 * (1.0 + erf(x / (math.sqrt(2.0) * sigma)))
    if w[1] > 0:
        F += w[1] * np.clip(0.5 + x / (math.sqrt(12.0) * sigma), 0.0, 1.0)
    if w[2] > 0:
        tail = 0.5 * np.exp(-math.sqrt(2.0) * np.abs(x) / sigma)
        F += w[2] * np.where(x <= 0, tail, 1.0 - tail)
    return F


def profile_transform(Q2, sigma, weights):
    """
    Fourier transform of the derivative of the mixed step (step_fraction) at wave vector Q, given Q² (may be
    complex): how much of a boundary's sharp reflection survives the roughness.
        erf   exp(−Q²σ²/2)
        lin   sin(√3·σ·Q) / (√3·σ·Q)
        exp   1 / (1 + Q²σ²/2)
    (Svechnikov 2020, Table 2, with Q = 2·kz.) The mix uses the same normalized weights as step_fraction:
    the transform of a sum is the sum of the transforms.
    """
    w = profile_weights(weights)
    Q2 = np.asarray(Q2, dtype=complex)
    result = np.zeros(Q2.shape, dtype=complex)
    if w[0] > 0:
        result += w[0] * np.exp(-Q2 * sigma**2 / 2.0)
    if w[1] > 0:
        u = math.sqrt(3.0) * sigma * np.sqrt(Q2)
        small = np.abs(u) < 1e-9
        result += w[1] * np.where(small, 1.0, np.sin(u) / np.where(small, 1.0, u))
    if w[2] > 0:
        result += w[2] / (1.0 + Q2 * sigma**2 / 2.0)
    return result


def roughness_factors(kz_upper, kz_lower, sigma, weights):
    """
    Factors of a rough boundary for the Parratt recursion (Névot–Croce for r, Vidal–Vincent for t,
    written with the transform of the chosen profile, so erf / lin / exp mixes work the same way):
        r  -> r⁰ · W,          W = transform(Q² = 4·kz_upper·kz_lower)
        t·t' -> (1 − r⁰²) · T,   T = 1 / transform(Q² = (kz_upper − kz_lower)²)²
    For erf: W = exp(−2·kz_u·kz_l·σ²), T = exp((kz_u − kz_l)²·σ²).
    The transmission part matters below the critical edges (evanescent layers); without it (r only) R can
    even exceed 1 there.
    """
    W = profile_transform(4.0 * kz_upper * kz_lower, sigma, weights)
    T = 1.0 / profile_transform((kz_upper - kz_lower) ** 2, sigma, weights) ** 2
    return W, T


def reach(sigma, weights):
    """Distance from the boundary centre beyond which the mixed step is complete: the widest step used."""
    w = profile_weights(weights)
    return sigma * max(r for r, wk in zip(REACH, w) if wk > 0)


# ==============================================================================
# Slicing of the rough boundaries
# ==============================================================================
class Interface:
    """
    How the roughness of the boundaries is calculated. The roughness itself belongs to the media
    (as in Multifitting): material.sigma and material.profile describe the UPPER boundary of every layer of
    that material, substrate.sigma / substrate.profile the boundary between the last layer and the substrate.

    The nominal layers keep their thicknesses. Only the profile (V or ε) is blurred:
    the sharp jump at boundary z_b becomes the smooth step  jump · F((z − z_b) / σ_b)  (F = step_fraction).
    Steps of neighbouring boundaries simply add up, i.e. the sharp profile is convolved with
    the roughness distribution. So when σ reaches the layer thickness the contrast between the
    materials fades, and for σ ≫ period the stack becomes the average (fully intermixed) medium.

    method    "factor": sharp layers, reflection and transmission × roughness_factors (fast, like Multifitting)
              "slices": the blurred profile below, cut into slices (exact for overlapping transitions, slow)
    slice_thickness   ("slices" only) each transition (z_b − reach … z_b + reach) is cut into n = ⌈2·reach / d⌉ equal
              slices of thickness ≤ d (Å). Equal slices act as a grating with a false Bragg peak at
              q = 2π/d, so keep d ≤ π/(4·q_max). A boundary with σ = 0 stays a sharp step (no slices).
    chaos     0 … 1: every boundary gets its own width σ_b · (1 + chaos·u), u random in [−1, 1]
    seed      start of the random numbers — same seed, same structure (reproducible curves and gradients)
    """

    def __init__(self, method="factor", slice_thickness=2.0, chaos=0.0, seed=1):
        self.method = method
        self.slice_thickness = float(slice_thickness)
        self.chaos = float(chaos)
        self.seed = int(seed)

    def n_slices(self, sigma_b, weights):
        """Number of slices across one transition of width σ_b and this profile (0 for a sharp boundary)."""
        return math.ceil(2.0 * reach(sigma_b, weights) / self.slice_thickness) if sigma_b > 0 else 0


# ==============================================================================
# METHOD 1: the layer stack
# ==============================================================================
def build_stack(materials, N, ambient, substrate, interface, quantity="potential", graded=None):
    """
    Returns three things, ordered from the ambient medium (first element) to the substrate (last element):

        values       quantity = "potential": neutron potentials V (µeV)
                     quantity = "epsilon":   X-ray permittivities ε (complex)
        thicknesses  (Å)    — ambient and substrate are semi-infinite and get 0
        roughness    (widths σ_b, profiles w_b) of every boundary b, for the reflectivity methods;
                     None when the roughness is already in the values (graded stack)

    ambient, substrate: Material objects for the media above and below the stack
    (only their potential / epsilon and, for the substrate, sigma and profile are used).
    Both quantities go through the same steps, so neutrons and X-rays see the same structure.

    1. Sharp stack: N periods of all materials in list order; period k uses material.thicknesses(N)[k].
       Every boundary gets the σ and profile of the medium below it. Layers with thickness ≤ 0 (a negative
       polynomial, or a material missing from a period of a file structure) are left out.
       Chaos: every σ_b is changed by its own random factor (1 + chaos·u).
       interface.method == "factor" (and graded not True): this sharp stack is returned with its roughness.
    2. Graded stack (method "slices", or graded=True e.g. for the depth plots): every boundary with σ > 0 is
       blurred (step_fraction) and cut into Interface.n_slices(σ, profile) slices; the flat middle of a layer
       stays one slice. Total slices ≈ Σ over boundaries of n_slices + one per layer middle.
    """
    # ---- 1. sharp stack: value of every medium, depth, roughness and profile of every boundary ----
    values = [getattr(ambient, quantity)]
    boundaries, sigmas, shapes = [], [], []                 # top of each layer (then the substrate), its σ and profile
    z = 0.0
    layer_thickness = []
    thickness_arrays = [material.thicknesses(N) for material in materials]   # each: N values, top period first
    for period in range(N):                                 # from the ambient side down to the substrate
        for material, d in zip(materials, thickness_arrays):
            if d[period] <= 0:
                continue
            boundaries.append(z)
            sigmas.append(material.sigma)                   # upper boundary of this layer
            shapes.append(material.profile)
            values.append(getattr(material, quantity))      # material.potential or material.epsilon
            layer_thickness.append(d[period])
            z += d[period]
    boundaries.append(z)
    sigmas.append(substrate.sigma)
    shapes.append(substrate.profile)
    values.append(getattr(substrate, quantity))

    random = np.random.default_rng(interface.seed)
    widths = np.array(sigmas) * (1.0 + interface.chaos * random.uniform(-1.0, 1.0, len(boundaries)))
    widths = np.maximum(widths, 0.0)

    if not (graded or (graded is None and interface.method == "slices")):
        thicknesses = np.array([0.0] + layer_thickness + [0.0])
        return np.array(values), thicknesses, (widths, shapes)

    # ---- 2. graded stack ----
    jumps = np.diff(values)                                 # jump of V (or ε) at every boundary

    # slice edges: all boundaries + n_slices(σ_b) equal slices across every rough transition
    edges = list(boundaries)
    for z_b, w, shape in zip(boundaries, widths, shapes):
        if w > 0:
            r = reach(w, shape)
            edges.extend(np.linspace(z_b - r, z_b + r, interface.n_slices(w, shape) + 1))
    edges = np.unique(np.round(edges, 6))
    centres = 0.5 * (edges[:-1] + edges[1:])

    # value at every slice centre: value_top + Σ jump_b · F((z − z_b)/σ_b)   (sharp boundary: plain step)
    smooth = np.full(centres.size, values[0], dtype=np.result_type(jumps, float))   # float for V, complex for ε
    passed = np.zeros(centres.size + 1, dtype=smooth.dtype) # steps that are complete from slice i on
    for z_b, w, shape, jump in zip(boundaries, widths, shapes, jumps):
        r = reach(w, shape)
        lo, hi = np.searchsorted(centres, [z_b - r, z_b + r])
        if hi > lo:
            smooth[lo:hi] += jump * step_fraction(centres[lo:hi] - z_b, w, shape)
        passed[hi] += jump
    smooth += np.cumsum(passed)[:-1]

    values = np.concatenate(([values[0]], smooth, [values[-1]]))
    thicknesses = np.concatenate(([0.0], np.diff(edges), [0.0]))
    return values, thicknesses, None


# ==============================================================================
# The Parratt recursion (used by METHOD 2.1 and 2.2)
# ==============================================================================
def parratt(values, thicknesses, kz_of, fresnel, roughness=None):
    """
    Specular reflectivity |r|² of the stack, starting inside the substrate (no reflected wave there) and
    working up to the ambient medium. At boundary j (medium j above, j+1 below):

        X_j = [r_j + (t_j·t'_j + r_j²)·X_j+1·p] / [1 + r_j·X_j+1·p],   p = exp(2i·kz_j+1·d_j+1)

    p is the round trip through layer j+1. For a sharp boundary t·t' = 1 − r², which gives the usual
    X_j = (r_j + X_j+1·p) / (1 + r_j·X_j+1·p).

    kz_of(value)                   z component of the wave number in a medium with this V or ε (array over q)
    fresnel(kz_u, kz_l, v_u, v_l)  Fresnel reflection coefficient of the boundary upper / lower
    roughness = (σ_b, profile_b) of every boundary (sharp stack): r and t·t' get the roughness_factors.

    For a sharp stack kz is computed once per distinct medium (there are only a few) and the roughness
    factors once per distinct (medium above, medium below, σ, profile).
    """
    kz_cache, factor_cache = {}, {}

    def kz(value):
        if roughness is None:                               # graded stack: (almost) every slice is different
            return kz_of(value)
        if value not in kz_cache:
            kz_cache[value] = kz_of(value)
        return kz_cache[value]

    X = 0.0                                                 # reflection amplitude below the current boundary
    kz_lower, v_lower = kz(values[-1]), values[-1]
    for j in range(len(values) - 2, -1, -1):
        v_upper = values[j]
        kz_upper = kz(v_upper)
        r0 = fresnel(kz_upper, kz_lower, v_upper, v_lower)  # sharp boundary j / j+1
        r, tt = r0, 1.0 - r0 * r0
        if roughness is not None and roughness[0][j] > 0:
            key = (v_upper, v_lower, float(roughness[0][j]), tuple(roughness[1][j]))
            if key not in factor_cache:
                factor_cache[key] = roughness_factors(kz_upper, kz_lower, roughness[0][j], roughness[1][j])
            W, T = factor_cache[key]
            r, tt = r0 * W, (1.0 - r0 * r0) * T
        if j + 1 < len(values) - 1:                         # a layer below this boundary (not the substrate)
            Xp = X * np.exp(2j * kz_lower * thicknesses[j + 1])
            X = (r + (tt + r * r) * Xp) / (1.0 + r * Xp)
        else:
            X = r
        kz_lower, v_lower = kz_upper, v_upper
    R = np.abs(X) ** 2
    if roughness is not None:
        # the factors are an approximation: at narrow resonances below a critical edge they can overshoot
        # the physical limit R = 1 by about 1–2 % (σ ≤ 0.1 of the layer thickness); R never exceeds 1
        R = np.minimum(R, 1.0)
    return R


# ==============================================================================
# METHOD 2.1: the neutron reflectivity
# ==============================================================================
def neutron_reflectivity(potentials, thicknesses, q, roughness=None):
    """
    Specular neutron reflectivity R(q) of the stack (Parratt recursion).
    kz² = (q/2)² − KZ2_PER_UEV·V,   r = (kz_upper − kz_lower) / (kz_upper + kz_lower).
    roughness: the third result of build_stack (None for a graded stack).
    """
    q = np.asarray(q, dtype=float)
    return parratt(potentials, thicknesses,
                   kz_of=lambda V: np.sqrt((q / 2.0) ** 2 - KZ2_PER_UEV * V + 0j),
                   fresnel=lambda ku, kl, vu, vl: (ku - kl) / (ku + kl),
                   roughness=roughness)


# ==============================================================================
# METHOD 2.2: the X-ray reflectivity
# ==============================================================================
def xray_reflectivity(epsilons, thicknesses, q, wavelength, polarization="s", roughness=None):
    """
    Specular X-ray reflectivity R(q) of the stack — METHOD 2.1 with the potentials replaced by the
    dielectric permittivities, following Svechnikov, J. Appl. Cryst. 53 (2020), Eq. (1):

        χ_j = k·(ε_j − cos²φ)^½      z component of the wave number in medium j
        k = 2π/λ,  φ = glancing angle,  q = 2k·sin φ  (so cos²φ = 1 − (q/2k)²)

    r_j is the Fresnel coefficient of boundary j / j+1:
        s-polarization:  r_j = (χ_j − χ_j+1) / (χ_j + χ_j+1)
        p-polarization:  r_j = (ε_j+1·χ_j − ε_j·χ_j+1) / (ε_j+1·χ_j + ε_j·χ_j+1)
    (Since (q/2)² = k² sin²φ, χ_j² = (q/2)² − k²(1 − ε_j): the neutron formula with K·V replaced by k²(1 − ε).)
    roughness: the third result of build_stack (None for a graded stack).
    """
    q = np.asarray(q, dtype=float)
    k = 2.0 * np.pi / wavelength
    cos2_phi = 1.0 - (q / (2.0 * k)) ** 2
    if polarization == "p":
        fresnel = lambda cu, cl, eu, el: (el * cu - eu * cl) / (el * cu + eu * cl)
    else:
        fresnel = lambda cu, cl, eu, el: (cu - cl) / (cu + cl)
    return parratt(epsilons, thicknesses, kz_of=lambda eps: k * np.sqrt(eps - cos2_phi + 0j),
                   fresnel=fresnel, roughness=roughness)


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
