import numpy as np
import matplotlib.pyplot as plt
import time


# ==============================================================================
# 1. Interface Grader: Discretize Interface into N Sublayers
# ==============================================================================
def interlayer_potentials(V_left, V_right, n, profile="linear"):
    """
    Computes the potentials of n sublayers that grade the interface 
    between materials with potentials V_left and V_right.
    
    profile: "linear", "erf" (Gaussian error function), or "sine"
    """
    if n <= 0:
        return []
    if n == 1:
        return [0.5 * (V_left + V_right)]
    
    if profile == "linear":
        return [V_left + (V_right - V_left) * (k + 1) / (n + 1) for k in range(n)]
    
    elif profile == "erf":
        # Error function (Gaussian integral) profile
        from scipy.special import erf
        x = np.linspace(-2.0, 2.0, n + 2)[1:-1]
        scale = 0.5 * (1.0 + erf(x))
        return list(V_left + (V_right - V_left) * scale)
    
    elif profile == "sine":
        # Smooth sinusoidal transition
        x = np.linspace(0.0, np.pi, n + 2)[1:-1]
        scale = 0.5 * (1.0 - np.cos(x))
        return list(V_left + (V_right - V_left) * scale)
    
    else:
        raise ValueError(f"Unknown profile type: {profile}")


# ==============================================================================
# 2. Multilayer Stack Builder (Vacuum -> Sublayers -> Bulk -> Substrate)
# ==============================================================================
def build_layer_stack(array_of_thk,
                      array_of_pot,
                      V_Sub,
                      V_vac,
                      n_interlayers,
                      d_interlayer,
                      profile="linear"):
    """
    Constructs the discretized multilayer stack with graded sublayers.
    Beam hits the thickest layer first (reversed internally).
    """
    n_mat = len(array_of_thk)
    if n_mat == 0:
        raise ValueError("array_of_thk must contain at least one material.")
    if len(array_of_pot) != n_mat:
        raise ValueError("array_of_thk and array_of_pot must match in length.")

    # Beam hits the thickest repeat first -> reverse each material's array
    thk_arrays = [np.asarray(t)[::-1] for t in array_of_thk]
    n_repeats = len(thk_arrays[0])

    # Interleave materials: mat0, mat1, ..., matM-1
    main_thicknesses = [thk_arrays[m][k] for k in range(n_repeats) for m in range(n_mat)]
    main_potentials = [array_of_pot[m] for _ in range(n_repeats) for m in range(n_mat)]

    potentials, thicknesses = [], []

    # --- vacuum -> first layer ---
    potentials += interlayer_potentials(V_vac, main_potentials[0], n_interlayers, profile=profile)
    thicknesses += [d_interlayer] * n_interlayers

    # --- bulk layers and internal interfaces ---
    n_layers = len(main_potentials)
    for i, (V, d) in enumerate(zip(main_potentials, main_thicknesses)):
        potentials.append(V)
        thicknesses.append(d)

        V_next = main_potentials[i + 1] if i + 1 < n_layers else V_Sub
        potentials += interlayer_potentials(V, V_next, n_interlayers, profile=profile)
        thicknesses += [d_interlayer] * n_interlayers

    return np.asarray(thicknesses, dtype=float), np.asarray(potentials, dtype=complex)


# ==============================================================================
# 3. Vectorized Parratt Reflectivity Engine for Discretized Sublayers
# ==============================================================================
def reflectivity_parratt_sublayers(thicknesses,
                                   potentials,
                                   q_vals,
                                   V_in=0.0,
                                   V_out=90e-6,
                                   E_meV=3.27,
                                   lam_A=5.0):
    """
    Calculates specular neutron reflectivity using Parratt's recursion 
    across all discretized sublayers simultaneously for all q-values.
    
    Parameters:
    -----------
    thicknesses : array (N_sublayers,)
        Thickness of every bulk layer and intermediate sublayer slice (Å).
    potentials  : array (N_sublayers,)
        Optical potential of every sublayer (meV). Supports complex values for absorption.
    q_vals      : array (M_q,)
        q-vector range in 1/Å.
    V_in, V_out : float
        Potentials of the incoming ambient medium (vacuum) and exit substrate (meV).
    """
    q = np.asarray(q_vals, dtype=float)
    k0 = 2.0 * np.pi / lam_A
    pot_factor = (k0**2) / E_meV

    # Full system stack: [Vacuum, sublayer_1, sublayer_2, ..., sublayer_N, Substrate]
    all_pots = np.concatenate(([V_in], potentials, [V_out]))
    all_thks = np.concatenate(([0.0], thicknesses, [0.0]))
    total_layers = len(all_pots)

    # Calculate kz for each layer across all q-points simultaneously: shape (total_layers, len(q))
    kz_inc_sq = (q / 2.0)**2
    # kz = sqrt((q/2)^2 - k0^2 * (V/E))
    kz = np.sqrt(kz_inc_sq[None, :] - (pot_factor * all_pots[:, None]) + 0.0j)

    # Amplitude reflection coefficient array (starts at substrate = 0)
    X = np.zeros(len(q), dtype=complex)

    # Backward Parratt recursion from substrate to vacuum
    for j in range(total_layers - 2, -1, -1):
        kz_j = kz[j]
        kz_next = kz[j + 1]
        d_next = all_thks[j + 1]

        # Sharp Fresnel reflection at this sublayer interface step
        r = (kz_j - kz_next) / (kz_j + kz_next)

        # Phase propagation across sublayer (j+1)
        # Numerically stable: exponential decay when kz is imaginary
        phi = np.exp(2.0j * kz_next * d_next)

        # Parratt recursive step
        X = (r + X * phi) / (1.0 + r * X * phi)

    # Reflectivity R = |X_vacuum|^2
    return np.abs(X)**2


# ==============================================================================
# 4. Visualization: Potential Profile Staircase V(z)
# ==============================================================================
def plot_potential_profile(thicknesses, potentials, V_vac, V_Sub, pad_A=50.0):
    total_z = np.sum(thicknesses)
    z_edges = np.concatenate(([0.0], np.cumsum(thicknesses)))

    z_stair = np.repeat(z_edges, 2)[1:-1]
    V_stair = np.repeat(potentials.real, 2)

    z = np.concatenate(([-pad_A, 0.0], z_stair, [total_z, total_z + pad_A]))
    V = np.concatenate(([V_vac, V_vac], V_stair, [V_Sub, V_Sub]))

    plt.figure(figsize=(10, 4))
    plt.plot(z, V * 1e6, color='navy', lw=1.2)
    plt.fill_between(z, V * 1e6, 0.0, alpha=0.15, color='navy')

    plt.xlabel("Depth $z$ (Å)")
    plt.ylabel("Potential $V$ ($\mu$eV)")
    plt.title(f"Discretized Potential Profile $V(z)$ (Total Sublayers: {len(thicknesses)})")
    plt.xlim(-pad_A, total_z + pad_A)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# ==============================================================================
# 5. EXECUTION AND TEST
# ==============================================================================
if __name__ == "__main__":

    # Supermirror Parameters
    N_repeats = 200
    d_Ni_A = np.linspace(40, 200, N_repeats)
    d_Ti_A = np.linspace(40, 200, N_repeats)

    V_Ni = 241e-6    # 241 neV (in meV)
    V_Ti = -49e-6    # -49 neV (in meV)
    V_Sub = 90e-6    # Si substrate (90 neV in meV)
    V_vac = 0.0

    # Sublayer Interface Parameters
    n_sublayers = 4         # 6 graded steps per interface
    d_sublayer_step = 5.5   # 1.5 Å per step (Total transition width = 9 Å)
    profile_type = "linear" # "linear", "erf", or "sine"

    # Build the full sublayer stack
    thk_stack, pot_stack = build_layer_stack(
        array_of_thk=[d_Ni_A, d_Ti_A],
        array_of_pot=[V_Ni, V_Ti],
        V_Sub=V_Sub,
        V_vac=V_vac,
        n_interlayers=n_sublayers,
        d_interlayer=d_sublayer_step,
        profile=profile_type
    )

    print(f"Built multilayer stack with {len(thk_stack)} discrete sublayers.")

    # Calculate Reflectivity using the Parratt Sublayer Engine
    q_range = np.linspace(0.001, 0.09, 200)
    q_Ni_edge = 0.0215

    t0 = time.perf_counter()
    R = reflectivity_parratt_sublayers(
        thicknesses=thk_stack,
        potentials=pot_stack,
        q_vals=q_range,
        V_in=V_vac,
        V_out=V_Sub,
        E_meV=3.27,
        lam_A=5.0
    )
    t1 = time.perf_counter()
    print(f"Calculated {len(q_range)} q-points across {len(thk_stack)} sublayers in {(t1 - t0)*1000:.2f} ms")

    # Plotting
    plot_potential_profile(thk_stack, pot_stack, V_vac, V_Sub)

    plt.figure(figsize=(10, 6))
    plt.plot(q_range / q_Ni_edge, R, 'b-', lw=1.5, 
                 label=f'Parratt Sublayers ($N={N_repeats}$, {n_sublayers} steps/interface)')
    plt.axvline(1.0, color='k', linestyle='--', alpha=0.5, label='Ni Critical Edge')
    plt.axvline(3.0, color='r', linestyle=':', label='m = 3.0')

    plt.title("Neutron Reflectivity with Discretized Sublayer Interfaces")
    plt.xlabel("$q^* = q / q_{\\mathrm{Ni}}$")
    plt.ylabel("Reflectivity $R$ (log scale)")
    plt.ylim(1e-5, 1.3)
    plt.xlim(0, 4)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()