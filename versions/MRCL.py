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
    """
    Build a full multilayer stack (vacuum -> N-material multilayer -> substrate).

    Generalized to an arbitrary number of materials (the original version
    was hard-coded to exactly two, e.g. Ni/Ti).

    Parameters
    ----------
    array_of_thk : sequence of array-like
        One 1-D thickness array per material, e.g. [d_mat1, d_mat2, ..., d_matM].
        All arrays must have the same length N (number of bilayer/multilayer
        repeats). Arrays are given thin -> thick; the beam encounters the
        thickest repeat first, so they are reversed internally.
    array_of_pot : sequence of float
        Scattering potential of each material, in the SAME order as
        array_of_thk (array_of_pot[m] goes with array_of_thk[m]).
    V_Sub : float
        Substrate potential.
    V_vac : float
        Vacuum / incident-medium potential.
    n_interlayers : int
        Number of graded sublayers inserted at every interface (0 = sharp).
    d_interlayer : float
        Thickness of each graded sublayer.

    Returns
    -------
    potentials, thicknesses : list[float]
        Full layer-by-layer stack, ordered as the beam encounters it.
    V_vac, V_Sub : float
        Echoed back for convenience (used as the incident/exit media in
        `reflectivity`).
    """
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
    N = 200
    d_Ni_A = np.linspace(30, 120, N)
    d_Ti_A = np.linspace(20, 90, N)

    q_Ni_edge = 0.0215
    q_range = np.linspace(0.001, 0.08, 1000)

    m_val = 3.0

    # Same 2-material call as before, now going through the generalized API:
    pot_, thk_, V_in, V_out = build_layer_stack(
        array_of_thk=[d_Ni_A, d_Ti_A],
        array_of_pot=[241e-6, -49e-6],
        V_Sub=90e-6,
        V_vac=0.0,
        n_interlayers=0,
        d_interlayer=0.1,
    )

#    Ni - 241e-6
#    Ti - -49e-6
#    Ni80Mo20 - 220e-6

    R_step = reflectivity(pot_, thk_, q_range, V_in, V_out)

    plt.figure(figsize=(10, 6))
    plt.semilogy(q_range / q_Ni_edge, R_step, 'b-', label=f'RSD TMM (N={N})')
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
