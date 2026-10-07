import numpy as np
import matplotlib.pyplot as plt


def interlayer_potentials(V_left, V_right, n):
    """
    Potentials of the n sublayers that grade the interface between
    materials with potentials V_left and V_right.

    n = 0 : sharp interface (returns [])
    n = 1 : single sublayer at the mean potential
    n > 1 : linear staircase from V_left to V_right
    """
    if n <= 0:
        return []
    if n == 1:
        return [0.5 * (V_left + V_right)]
    return [V_left + (V_right - V_left) * (k + 1) / (n + 1)
            for k in range(n)]


# ==============================================================================
# Builder: full stack (vacuum -> N-material multilayer -> substrate)
# ==============================================================================
def build_layer_stack(array_of_thk,
                       array_of_pot,
                       V_Sub,
                       V_vac,
                       n_interlayers,
                       d_interlayer):

    n_mat = len(array_of_thk)
    if n_mat == 0:
        raise ValueError("array_of_thk must contain at least one material.")
    if len(array_of_pot) != n_mat:
        raise ValueError(
            f"array_of_thk has {n_mat} materials but array_of_pot has "
            f"{len(array_of_pot)} potentials; they must match 1:1 and be "
            f"in the same order."
        )

    # Beam hits the thickest repeat first -> reverse each material's array
    thk_arrays = [np.asarray(t)[::-1] for t in array_of_thk]

    lengths = {t.size for t in thk_arrays}
    if len(lengths) != 1:
        raise ValueError(
            f"All thickness arrays in array_of_thk must have the same "
            f"length, got sizes {[t.size for t in thk_arrays]}."
        )
    n_repeats = lengths.pop()

    # Interleave materials repeat-by-repeat: mat0, mat1, ..., matM-1, mat0, ...
    main_thicknesses = [thk_arrays[m][k]
                         for k in range(n_repeats)
                         for m in range(n_mat)]
    main_potentials = [array_of_pot[m]
                        for _ in range(n_repeats)
                        for m in range(n_mat)]

    potentials, thicknesses = [], []

    # --- vacuum -> first layer ---
    potentials += interlayer_potentials(V_vac, main_potentials[0], n_interlayers)
    thicknesses += [d_interlayer] * n_interlayers

    # --- bulk layers and internal interfaces ---
    n_layers = len(main_potentials)
    for i, (V, d) in enumerate(zip(main_potentials, main_thicknesses)):
        potentials.append(V)
        thicknesses.append(d)

        V_next = main_potentials[i + 1] if i + 1 < n_layers else V_Sub
        potentials += interlayer_potentials(V, V_next, n_interlayers)
        thicknesses += [d_interlayer] * n_interlayers

    return potentials, thicknesses, V_vac, V_Sub


# ==============================================================================
# Potential profile V(z): turn the layer stack into a depth-resolved staircase
# ==============================================================================
def potential_profile(potentials, thicknesses, V_vac, V_Sub, pad_frac=0.05):

    total = float(np.sum(thicknesses)) if len(thicknesses) else 0.0
    pad = pad_frac * total if total > 0 else 1.0

    z_edges = np.concatenate(([0.0], np.cumsum(thicknesses)))

    if len(potentials) > 0:
        # Duplicate each edge/level so consecutive (z, V) pairs draw flat
        # segments with vertical jumps at every interface.
        z_stair = np.repeat(z_edges, 2)[1:-1]
        V_stair = np.repeat(potentials, 2)
    else:
        z_stair, V_stair = np.array([]), np.array([])

    z = np.concatenate(([-pad, 0.0], z_stair, [total, total + pad]))
    V = np.concatenate(([V_vac, V_vac], V_stair, [V_Sub, V_Sub]))

    return z, V


def plot_potential_profile(potentials, thicknesses, V_vac, V_Sub,
                            ax=None, pad_frac=0.05,
                            title="Structure potential profile V(z)"):
    """
    Plot V(z): one flat step per structural layer, at that layer's
    potential, so every level in the graph corresponds directly to a
    level (bulk layer or interlayer) in the built stack.
    """
    z, V = potential_profile(potentials, thicknesses, V_vac, V_Sub, pad_frac)
    total = float(np.sum(thicknesses)) if len(thicknesses) else 0.0
    pad = pad_frac * total if total > 0 else 1.0

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.plot(z, V, color='navy', lw=1.5)
    ax.fill_between(z, V, 0.0, alpha=0.15, color='navy')

    # Dotted reference line at every distinct potential level present
    # (vacuum, substrate, and each material), with its value labeled.
    levels = sorted(set([V_vac, V_Sub] + list(potentials)))
    for lvl in levels:
        ax.axhline(lvl, color='gray', ls=':', lw=0.6, alpha=0.6)

    ymin, ymax = ax.get_ylim()
    pady = 0.08 * (ymax - ymin) if ymax > ymin else 1.0
    ax.set_ylim(ymin - pady, ymax + pady)
    for lvl in levels:
        ax.annotate(f"{lvl:.3g}", xy=(total + pad, lvl),
                    xytext=(4, 0), textcoords='offset points',
                    va='center', fontsize=7, color='dimgray', clip_on=False)

    ax.axvspan(-pad, 0.0, color='0.9', alpha=0.5, lw=0)
    ax.axvspan(total, total + pad, color='0.85', alpha=0.5, lw=0)
    y_text = ax.get_ylim()[1] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.text(-pad / 2, y_text, "Vacuum", ha='center', va='top',
            fontsize=8, color='gray')
    ax.text(total + pad / 2, y_text, "Substrate", ha='center', va='top',
            fontsize=8, color='gray')

    ax.set_xlim(-pad, total + pad)
    ax.set_xlabel("Depth z (Å)")
    ax.set_ylabel("Potential V")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    return ax


# ==============================================================================
# Reflectivity: only needs the stack and the boundary potentials
# ==============================================================================
def reflectivity(potentials, thicknesses, q_vals,
                  V_in, V_out,
                  E_meV=3.27, lam_A=5.0):

    k_tot = 2.0 * np.pi / lam_A

    R = np.empty_like(q_vals, dtype=float)

    for idx, q in enumerate(q_vals):
        kz_inc = q / 2.0

        kz_prev = np.sqrt(kz_inc**2 - (k_tot**2 * V_in / E_meV) + 0.0j)

        M = np.eye(2, dtype=complex)

        for V, d in zip(potentials, thicknesses):
            kz = np.sqrt(kz_inc**2 - (k_tot**2 * V / E_meV) + 0.0j)

            M_int = (1.0 / (2.0 * kz_prev)) * np.array([
                [kz_prev + kz, kz_prev - kz],
                [kz_prev - kz, kz_prev + kz]
            ], dtype=complex)

            P = np.array([
                [np.exp(-1.0j * kz * d), 0.0j],
                [0.0j,                   np.exp( 1.0j * kz * d)]
            ], dtype=complex)

            M = M @ M_int @ P
            kz_prev = kz

        kz_out = np.sqrt(kz_inc**2 - (k_tot**2 * V_out / E_meV) + 0.0j)

        M_out = (1.0 / (2.0 * kz_prev)) * np.array([
            [kz_prev + kz_out, kz_prev - kz_out],
            [kz_prev - kz_out, kz_prev + kz_out]
        ], dtype=complex)

        M = M @ M_out

        r = M[1, 0] / M[0, 0]
        R[idx] = np.abs(r)**2

    return R


# ==============================================================================
# EXECUTION AND PLOTTING
# ==============================================================================
if __name__ == "__main__":

    # ---- RSD thickness arrays (thin -> thick) ---- 
    N = 2
    d_Ni_A = np.linspace(30, 120, N)
    d_Ti_A = np.linspace(30, 120, N)

    q_Ni_edge = 0.0215
    q_range = np.linspace(0.001, 0.08, 1000)

    m_val = 3.0

    # Same 2-material call as before, now going through the generalized API:
    pot_, thk_, V_in, V_out = build_layer_stack(
        array_of_thk=[d_Ni_A, d_Ti_A],
        array_of_pot=[241e-6, -49e-6],
        V_Sub=90e-6,
        V_vac=0.0,
        n_interlayers=5,
        d_interlayer=2,
    )

    # Example of the new N-material capability (3+ materials), e.g. Ni/Ti/Cu:
    # d_Cu_A = np.linspace(15, 60, N)
    # pot_, thk_, V_in, V_out = build_layer_stack(
    #     array_of_thk=[d_Ni_A, d_Ti_A, d_Cu_A],
    #     array_of_pot=[241e-6, -49e-6, 61e-6],
    #     V_Sub=90e-6,
    #     V_vac=0.0,
    #     n_interlayers=0,
    #     d_interlayer=0.1,
    # )

    # --- Potential profile V(z): one graph level per structure level ---
    plot_potential_profile(pot_, thk_, V_in, V_out)

    R_step = reflectivity(pot_, thk_, q_range, V_in, V_out)

    plt.figure(figsize=(10, 6))
    plt.plot(q_range / q_Ni_edge, R_step, 'b-', label=f'RSD TMM (N={N})')
    plt.axvline(1.0, color='k', linestyle='--', alpha=0.5, label='Ni Critical Edge')
    plt.axvline(m_val, color='r', linestyle=':', label=f'm = {m_val}')

    plt.title("Reflectivity of RSD Supermirror")
    plt.xlabel("$q^* = q / q_{Ni}$")
    plt.ylabel("Reflectivity (log scale)")
    plt.ylim(1e-4, 1.2)
    plt.xlim(0, 4)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.show()
