import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter

def generate_rsd_structure(m, alpha, sigma0, h, max_bilayers=300):
    """
    Generate Ni/Ti layer thicknesses. 
    Order: Thinnest (bilayer 1) to Thickest (final bilayer).
    """
    lambda_Ni = 58.4          
    lambda_c_star = 53.2      
    lambda_cTi_abs = 129.5    

    D = lambda_Ni / (2.0 * m)   
    z_phys = 0.0                

    d_Ni_list = []
    d_Ti_list = []

    while D < lambda_Ni / 2.0 and len(d_Ni_list) < max_bilayers:
        sigma_n = np.sqrt(sigma0**2 + h * z_phys)

        # Eq. 5: Physical thicknesses
        d_Ni = D / (2.0 * np.sqrt(1.0 - (2.0 * D / lambda_Ni)**2))
        d_Ti = D / (2.0 * np.sqrt(1.0 + (2.0 * D / lambda_cTi_abs)**2))

        d_Ni_list.append(d_Ni)
        d_Ti_list.append(d_Ti)

        z_phys += d_Ni + d_Ti
        step = alpha * D * (D / lambda_c_star)**4 * np.exp(-4.0 * np.pi**2 * sigma_n**2 / D**2)
        D += step

    return np.array(d_Ni_list), np.array(d_Ti_list)


# ==============================================================================
# 2. FIXED TRANSFER MATRIX METHOD (TMM)
# ==============================================================================

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


def reflectivity_from_thicknesses(d_Ni_A, d_Ti_A, q_vals,
                                  V_NiMo=220e-6, V_Ti=-49e-6, V_Sub=90e-6,
                                  E_meV=3.27, lam_A=5.0,
                                  n_interlayers=4, d_interlayer=10.0):
    """
    Reflectivity with rough interfaces modelled by inserting
    `n_interlayers` sublayers of thickness `d_interlayer` (Å) at each
    interface. The sublayer potentials interpolate between the two
    adjacent bulk potentials.

    Parameters
    ----------
    d_Ni_A, d_Ti_A : 1D arrays
        Layer thicknesses (Å), ordered thin -> thick as produced by RSD.
    q_vals : 1D array
        Values of q (1/Å) at which to compute R.
    n_interlayers : int
        Number of graded sublayers per interface (0 = sharp).
    d_interlayer : float
        Thickness (Å) of each sublayer.
    """
    k_tot = 2.0 * np.pi / lam_A

    # Beam hits thickest layers first
    thick_Ni = d_Ni_A[::-1]
    thick_Ti = d_Ti_A[::-1]

    # ---------- Build main layer list (Ni, Ti, Ni, Ti, ...) ----------
    main_thicknesses = []
    main_potentials = []
    for d_ni, d_ti in zip(thick_Ni, thick_Ti):
        main_thicknesses.append(d_ni)
        main_potentials.append(V_NiMo)
        main_thicknesses.append(d_ti)
        main_potentials.append(V_Ti)

    # ---------- Insert interlayers at every interface ----------
    V_vac = 0.0
    potentials  = []
    thicknesses = []

    # Vacuum -> main[0]
    potentials  += interlayer_potentials(V_vac, main_potentials[0], n_interlayers)
    thicknesses += [d_interlayer] * n_interlayers

    for i, (V, d) in enumerate(zip(main_potentials, main_thicknesses)):
        # Bulk layer
        potentials.append(V)
        thicknesses.append(d)

        # Interface to next material (or substrate after the last layer)
        V_next = main_potentials[i + 1] if i + 1 < len(main_potentials) else V_Sub
        potentials  += interlayer_potentials(V, V_next, n_interlayers)
        thicknesses += [d_interlayer] * n_interlayers

    # ---------- Transfer-matrix loop ----------
    R = np.empty_like(q_vals, dtype=float)

    for idx, q in enumerate(q_vals):
        kz_inc = q / 2.0                       # as in your original code
        M = np.eye(2, dtype=complex)
        kz_prev = kz_inc                       # incident medium = vacuum

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

        # Final interface: last sublayer (or last bulk layer) -> substrate
        kz_sub = np.sqrt(kz_inc**2 - (k_tot**2 * V_Sub / E_meV) + 0.0j)

        M_sub = (1.0 / (2.0 * kz_prev)) * np.array([
            [kz_prev + kz_sub, kz_prev - kz_sub],
            [kz_prev - kz_sub, kz_prev + kz_sub]
        ], dtype=complex)

        M = M @ M_sub

        r = M[1, 0] / M[0, 0]
        R[idx] = np.abs(r)**2

    return R


# ==============================================================================
# 3. EXECUTION AND PLOTTING
# ==============================================================================

# Parameters
m_val = 3.0
alpha_val = 1.625
sigma_val = 1.5  # nm
h_val = 0      # nm

# 1. Generate layers (Physical units: nm)
d_Ni_nm, d_Ti_nm = generate_rsd_structure(m_val, alpha_val, sigma_val, h_val)

# 2. Convert to Angstrom for TMM
d_Ni_A = d_Ni_nm * 10.0
d_Ti_A = d_Ti_nm * 10.0

# 3. Define q range (1/Angstrom)
q_Ni_edge = 0.0215  
q_range = np.linspace(0.001, 0.08, 1000)

# 4. Calculate Reflectivity
R_res = reflectivity_from_thicknesses(d_Ni_A, d_Ti_A, q_range)

# Plotting


plt.plot(d_Ni_A)

workbook = xlsxwriter.Workbook('Layers.xlsx')
worksheet = workbook.add_worksheet()

# Add headers
worksheet.write('A1', 'NiMo')
worksheet.write('B1', 'Ti')

# Write array data
worksheet.write_column('A1', d_Ni_nm)
worksheet.write_column('B1', d_Ti_nm)

# Close the workbook
workbook.close()


plt.figure(figsize=(10, 6))
plt.semilogy(q_range / q_Ni_edge, R_res, 'b-', label=f'RSD TMM (N={len(d_Ni_nm)})')
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