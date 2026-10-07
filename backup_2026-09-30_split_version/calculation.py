"""
calculation.py — physics of the multilayer (no GUI code)

  * thickness_profile()        d(j) of one material over the N repeats (linear / polynomial)
  * build_stack_from_layers()  full (sub)layer stack: thicknesses + potentials
  * calculate_parratt()        specular reflectivity R(q) by the Parratt recursion
"""
import numpy as np


# ==============================================================================
# 1. Thickness profile d(j) for one material across the N repeats
# ==============================================================================
POLY_MAX_DEGREE = 6
THICKNESS_MODES = ("linear", "polynomial")


def thickness_profile(spec, n_repeats):
    """
    Thickness of one material for every repeat j = 0 … N-1.
    j = 0 is the substrate side (deepest bilayer), j = N-1 the vacuum side (top).

    spec['mode'] == 'linear'      : d(t) = d_start + (d_end - d_start) * t,   t = j / (N-1)
    spec['mode'] == 'polynomial'  : d(x) = c0 + c1*x + c2*x^2 + ... ,
                                    x = t = j/(N-1)  (spec['arg'] == 't', default)
                                    x = j            (spec['arg'] == 'j')
                                    coefficients in spec['coeffs'] = [c0, c1, c2, ...]
    Returns the raw values (may be negative for a badly chosen polynomial).
    """
    n = int(n_repeats)
    if n <= 0:
        return np.array([], dtype=float)
    j = np.arange(n, dtype=float)
    t = j / (n - 1) if n > 1 else np.zeros(1)

    mode = spec.get('mode', 'linear')
    if mode == 'polynomial':
        coeffs = np.asarray(spec.get('coeffs', [0.0]), dtype=float)
        if coeffs.size == 0:
            return np.zeros(n)
        x = j if spec.get('arg', 't') == 'j' else t
        return np.polynomial.polynomial.polyval(x, coeffs)

    d0 = float(spec.get('d_start', 50.0))
    d1 = float(spec.get('d_end', 50.0))
    return d0 + (d1 - d0) * t


# ==============================================================================
# 2. Vectorized Parratt Calculation Engine
# ==============================================================================
def interlayer_potentials(V_left, V_right, n):
    if n <= 0:
        return []
    if n == 1:
        return [0.5 * (V_left + V_right)]
    return [V_left + (V_right - V_left) * (k + 1) / (n + 1) for k in range(n)]


def build_stack_from_layers(layers, n_repeats, n_sublayers, d_sublayer, V_vac, V_sub):
    if not layers or n_repeats <= 0:
        return np.array([]), np.array([])

    n_mat = len(layers)
    thk_arrays = []
    pot_values = []

    for lay in layers:
        # d(j) from the layer's function; negative values are clipped to 0 Å.
        # Reversed so that the stack is written from the vacuum side (j = N-1) downwards.
        thk_seq = np.clip(thickness_profile(lay, n_repeats), 0.0, None)[::-1]
        thk_arrays.append(thk_seq)
        pot_values.append(lay['pot'] * 1e-3)  # Convert µeV -> meV

    V_vac_meV = V_vac * 1e-3
    V_sub_meV = V_sub * 1e-3

    main_thks = [thk_arrays[m][k] for k in range(n_repeats) for m in range(n_mat)]
    print(main_thks)
    main_pots = [pot_values[m] for _ in range(n_repeats) for m in range(n_mat)]

    potentials, thicknesses = [], []

    # Vacuum -> First Layer
    potentials += interlayer_potentials(V_vac_meV, main_pots[0], n_sublayers)
    thicknesses += [d_sublayer] * n_sublayers

    # Bulk layers + interfaces
    n_total_bulk = len(main_pots)
    for i, (V, d) in enumerate(zip(main_pots, main_thks)):
        potentials.append(V)
        thicknesses.append(d)

        V_next = main_pots[i + 1] if i + 1 < n_total_bulk else V_sub_meV
        potentials += interlayer_potentials(V, V_next, n_sublayers)
        thicknesses += [d_sublayer] * n_sublayers

    return np.array(thicknesses, dtype=float), np.array(potentials, dtype=complex)


def calculate_parratt(thicknesses, potentials, q_vals, V_vac, V_sub, lam_A=5.0):
    """
    Parratt recursion. kz is computed only for the distinct potential values
    (materials + interface grading steps), so memory stays O(unique·Q) even for
    stacks with tens of thousands of (sub)layers.
    """
    if len(thicknesses) == 0:
        return np.zeros_like(q_vals)

    q = np.asarray(q_vals, dtype=float)
    k0 = 2.0 * np.pi / lam_A
    E_meV = 81.8042 / (lam_A**2)
    pot_factor = (k0**2) / E_meV

    V_vac_meV = V_vac * 1e-3
    V_sub_meV = V_sub * 1e-3

    all_pots = np.concatenate(([V_vac_meV], potentials, [V_sub_meV])).astype(complex)
    all_thks = np.concatenate(([0.0], thicknesses, [0.0]))
    total_layers = len(all_pots)

    u_pots, u_idx = np.unique(np.round(all_pots, 12), return_inverse=True)
    kz_inc_sq = (q / 2.0)**2
    kz_u = np.sqrt(kz_inc_sq[None, :] - (pot_factor * u_pots[:, None]) + 0.0j)

    X = np.zeros(len(q), dtype=complex)

    for j in range(total_layers - 2, -1, -1):
        kz_j = kz_u[u_idx[j]]
        kz_next = kz_u[u_idx[j + 1]]
        d_next = all_thks[j + 1]

        r = (kz_j - kz_next) / (kz_j + kz_next)
        phi = np.exp(2.0j * kz_next * d_next)
        X = (r + X * phi) / (1.0 + r * X * phi)

    return np.abs(X)**2
