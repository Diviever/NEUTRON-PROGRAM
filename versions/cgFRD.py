import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter


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
# Builder: full stack (vacuum -> Ni/Ti multilayer -> substrate)
# ==============================================================================
def build_layer_stack(array_of_thk,
                      array_of_pot, 
                      V_Sub,
                      V_vac,
                      n_interlayers, 
                      d_interlayer):

    # Beam hits thickest layers first
    thick_1 = array_of_thk[0][::-1]
    thick_2 = array_of_thk[1][::-1]

    # Build main bulk layers (Ni, Ti, Ni, Ti, ...)
    main_thicknesses = []
    main_potentials  = []
    
    for thick_1, thick_2 in zip(thick_1, thick_2):
        main_thicknesses.append(thick_1); main_potentials.append(array_of_pot[0])
        main_thicknesses.append(thick_2); main_potentials.append(array_of_pot[1])

    potentials  = []
    thicknesses = []

    # --- vacuum -> first layer ---
    potentials  += interlayer_potentials(V_vac, main_potentials[0], n_interlayers)
    thicknesses += [d_interlayer] * n_interlayers

    # --- bulk layers and internal interfaces ---
    for i, (V, d) in enumerate(zip(main_potentials, main_thicknesses)):
        # Bulk layer
        potentials.append(V)
        thicknesses.append(d)

        # Interface to the next material (or to the substrate after the last)
        V_next = main_potentials[i + 1] if i + 1 < len(main_potentials) else V_Sub
        potentials  += interlayer_potentials(V, V_next, n_interlayers)
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
        # Perpendicular wavevector in vacuum (same convention as your code)
        kz_inc = q / 2.0

        # Wavevector in the incident medium
        kz_prev = np.sqrt(kz_inc**2 - (k_tot**2 * V_in / E_meV) + 0.0j)

        M = np.eye(2, dtype=complex)

        for V, d in zip(potentials, thicknesses):
            kz = np.sqrt(kz_inc**2 - (k_tot**2 * V / E_meV) + 0.0j)

            # Interface matrix
            M_int = (1.0 / (2.0 * kz_prev)) * np.array([
                [kz_prev + kz, kz_prev - kz],
                [kz_prev - kz, kz_prev + kz]
            ], dtype=complex)

            # Propagation matrix
            P = np.array([
                [np.exp(-1.0j * kz * d), 0.0j],
                [0.0j,                   np.exp( 1.0j * kz * d)]
            ], dtype=complex)

            M = M @ M_int @ P
            kz_prev = kz

        # Final interface to the exit medium
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

# ---- RSD thickness arrays (thin -> thick) ----
N = 200
d_Ni_A = np.linspace(30, 120, N)
d_Ti_A = np.linspace(20, 90, N)


# 3. Define q range (1/Angstrom)
q_Ni_edge = 0.0215  
q_range = np.linspace(0.001, 0.08, 1000)

# Parameters
m_val = 3.0
alpha_val = 1.625
sigma_val = 1.5  # nm
h_val = 0      # nm

pot_, thk_, V_in, V_out = build_layer_stack(array_of_thk = [d_Ni_A, d_Ti_A], 
                                            array_of_pot = [241e-6, -49e-6],
                                            V_Sub = 90e-6,
                                            V_vac = 0.0,
                                            n_interlayers = 0,
                                            d_interlayer = 0.1)

R_step = reflectivity(pot_, thk_, q_range, V_in, V_out)

plt.figure(figsize=(10, 6))
plt.semilogy(q_range / q_Ni_edge, R_step, 'b-', label=f'RSD TMM (N={len(d_Ni_A)})')
plt.axvline(1.0, color='k', linestyle='--', alpha=0.5, label='Ni Critical Edge')
plt.axvline(m_val, color='r', linestyle=':', label=f'm = {m_val}')

plt.title(f"Reflectivity of RSD Supermirror (Substrate & $k_z$ Fixes applied)")
plt.xlabel("$q^* = q / q_{Ni}$")
plt.ylabel("Reflectivity (log scale)")
plt.ylim(1e-4, 1.2)
plt.xlim(0, 4)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()
plt.show()