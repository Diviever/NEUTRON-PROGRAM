import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. RSD ALGORITHM (Analytical version, Eq. 12 in the paper)
#    Generates the supermirror layer sequence.
#    All thicknesses are returned in nanometres.
# ==============================================================================

def generate_rsd_structure(m, alpha, sigma0, h, max_bilayers=600):
    """
    Generate Ni/Ti supermirror layer thicknesses using the analytical RSD recursion.

    Parameters:
        m (float): Angular acceptance in units of Ni critical edge.
        alpha (float): Reflectance parameter (e.g., 1.625 for ~88% reflectivity).
        sigma0 (float): Initial interface roughness [nm].
        h (float): Roughness growth rate [nm] (statistical growth law).
        max_bilayers (int): Safety limit.

    Returns:
        d_Ni_list (list): Physical thickness of Ni in each bilayer [nm].
        d_Ti_list (list): Physical thickness of Ti in each bilayer [nm].
    """
    # RSD constants for Ni/Ti (from the paper)
    lambda_Ni = 58.4          # critical wavelength for Ni [nm]
    lambda_c_star = 53.2      # reduced characteristic wavelength [nm]
    lambda_cTi_abs = 129.5    # |lambda_c| for Ti (repulsive) [nm]

    # Start with the thinnest bilayer
    D = lambda_Ni / (2.0 * m)   # optical thickness of bilayer 1 [nm]
    z_phys = 0.0                # cumulative physical thickness deposited [nm]

    d_Ni_list = []
    d_Ti_list = []

    while D < lambda_Ni / 2.0 and len(d_Ni_list) < max_bilayers:
        # Current roughness at this bilayer (statistical growth)
        sigma_n = np.sqrt(sigma0**2 + h * z_phys)

        # Convert optical thickness D into physical Ni and Ti thicknesses
        # (refraction‑corrected quarter‑wave condition, Eq. 5)
        d_Ni = D / (2.0 * np.sqrt(1.0 - (2.0 * D / lambda_Ni)**2))
        d_Ti = D / (2.0 * np.sqrt(1.0 + (2.0 * D / lambda_cTi_abs)**2))

        d_Ni_list.append(d_Ni)
        d_Ti_list.append(d_Ti)

        # Update cumulative physical thickness (for roughness growth)
        z_phys += d_Ni + d_Ti

        # Analytical RSD recursion (Eq. 12)
        step = alpha * D * (D / lambda_c_star)**4 * np.exp(-4.0 * np.pi**2 * sigma_n**2 / D**2)
        D += step

    return np.array(d_Ni_list), np.array(d_Ti_list)


# ==============================================================================
# 2. TRANSFER MATRIX METHOD (from neutron.py, adapted for arbitrary layer lists)
#    Returns reflectivity R(q) for a given set of layer thicknesses.
# ==============================================================================

def reflectivity_from_thicknesses(d_Ni_A, d_Ti_A, q_A,
                                 V_Ni_meV=241e-6, V_Ti_meV=-49e-6,
                                 E_meV=3.27, lam_A=5.0):
    """
    Calculate the specular neutron reflectivity of a Ni/Ti multilayer.

    Parameters:
        d_Ni_A (array): Ni layer thicknesses [Angstrom].
        d_Ti_A (array): Ti layer thicknesses [Angstrom].
        q_A (array): Perpendicular wavevector transfer [1/Angstrom].
        V_Ni_meV, V_Ti_meV: Optical potentials [meV].
        E_meV: Neutron energy [meV].
        lam_A: Neutron wavelength [Angstrom] (used for k).

    Returns:
        R (array): Reflectivity |r|^2.
    """
    # Neutron wavevector magnitude
    k = 2.0 * np.pi / lam_A   # 1/Angstrom

    # Build the layer sequence: Ni, Ti, Ni, Ti, ...
    N = len(d_Ni_A)
    thicknesses = []
    potentials = []
    for i in range(N):
        thicknesses.append(d_Ni_A[i])
        potentials.append(V_Ni_meV)
        thicknesses.append(d_Ti_A[i])
        potentials.append(V_Ti_meV)

    # Pre‑compute p for each layer? We do it on the fly in the loop.
    R = np.empty_like(q_A, dtype=float)

    for idx, qi in enumerate(q_A):
        # Initialise transfer matrix (maps output side to input side)
        M = np.eye(2, dtype=complex)
        p_prev = qi   # incident medium is vacuum (potential=0)

        # Propagate through all layers
        for V, d in zip(potentials, thicknesses):
            # Perpendicular wavevector inside the layer
            p = np.sqrt(qi**2 - (k**2 * V / E_meV) + 0.0j)

            # Interface matrix from p_prev to p (standard convention)
            M_int = (1.0 / (2.0 * p_prev)) * np.array([
                [p_prev + p, p_prev - p],
                [p_prev - p, p_prev + p]
            ], dtype=complex)

            # Propagation inside the layer
            P = np.array([
                [np.exp(-1.0j * p * d), 0.0j],
                [0.0j,                 np.exp( 1.0j * p * d)]
            ], dtype=complex)

            M_layer = M_int @ P
            M = M @ M_layer
            p_prev = p

        # Final interface back to vacuum (p = qi)
        p = qi
        M_int = (1.0 / (2.0 * p_prev)) * np.array([
            [p_prev + p, p_prev - p],
            [p_prev - p, p_prev + p]
        ], dtype=complex)
        M = M @ M_int

        # Extract reflection coefficient (M maps output → input)
        r = M[1, 0] / M[0, 0]
        R[idx] = np.abs(r)**2

    return R


# ==============================================================================
# 3. MAIN: Generate a supermirror, compute its reflectivity, and plot.
# ==============================================================================

# ---- RSD parameters (match Fig. 4a and 7a in the paper) ----
m = 3.0
alpha = 1.625
sigma0 = 1.5   # nm
h = 0.0        # constant roughness (no growth)

print("Generating RSD structure...")
d_Ni_nm, d_Ti_nm = generate_rsd_structure(m, alpha, sigma0, h)
N_bilayers = len(d_Ni_nm)
print(f"Number of bilayers generated: {N_bilayers}")

# Convert to Angstrom for the TMM
d_Ni_A = d_Ni_nm * 10.0
d_Ti_A = d_Ti_nm * 10.0

# ---- TMM calculation ----
q_Ni = 0.0215   # Ni critical edge [1/Angstrom]  (4π / 58.4 nm)
q_max = 0.07    # 1/Angstrom  (covers m ~ 3.25)
q = np.linspace(0.001, q_max, 800)

print("Computing reflectivity (TMM)...")
R = reflectivity_from_thicknesses(d_Ni_A, d_Ti_A, q)

# Normalise q by the Ni critical edge to get q* = q / q_Ni
q_star = q / q_Ni

# ---- Plot ----
plt.figure(figsize=(9, 6))
plt.plot(q_star, R, 'b-', linewidth=1.5, label=f'RSD, {N_bilayers} bilayers')

# Vertical lines at Ni edge and supermirror edge
plt.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.0, label=r'$q^* = 1$ (Ni edge)')
plt.axvline(x=m,   color='red',  linestyle='--', linewidth=1.0, label=f'$q^* = {m}$ (SM edge)')

plt.xlabel(r'$q^* = q / q_{\mathrm{Ni}}$')
plt.ylabel('Reflectivity $R$')
plt.title(f'Ni/Ti supermirror, m = {m}, $\\alpha$ = {alpha}, ' +
          f'$\\sigma_0$ = {sigma0} nm, h = {h} nm')
plt.xlim(0, q_max / q_Ni)
plt.ylim(0, 1.05)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()