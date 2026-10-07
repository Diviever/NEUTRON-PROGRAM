import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. PHYSICAL CONSTANTS FOR Ni/Ti (from the paper)
# ----------------------------------------------------------------------
lambda_Ni = 58.4          # Critical wavelength of Ni [nm]
lambda_c_star = 53.2      # Reduced characteristic wavelength for Ni-Ti pair [nm]
lambda_cTi_abs = 129.5    # |lambda_c| for Ti (repulsive potential) [nm]

# ----------------------------------------------------------------------
# 2. RSD PARAMETERS (choose to match your desired supermirror)
# ----------------------------------------------------------------------
m = 3.0                   # Angular acceptance (e.g., m=3)
alpha = 1.625             # Reflectance parameter (from Fig. 7a for m=3, sigma=1.5nm)
sigma0 = 1.5              # Substrate / initial interface roughness [nm]
h = 0.0                   # Roughness growth rate [nm] (set 0 for constant roughness)

# ----------------------------------------------------------------------
# 3. INITIALIZATION
# ----------------------------------------------------------------------
D = lambda_Ni / m / 2.0   # Optical thickness of the thinnest bilayer (n=1) [nm]
z_phys = 0.0              # Cumulative physical thickness deposited so far [nm]

# Storage lists
D_list = []               # Optical thickness of each bilayer
sigma_list = []           # Roughness at each bilayer
d_Ni_list = []            # Physical thickness of Ni layer [nm]
d_Ti_list = []            # Physical thickness of Ti layer [nm]

max_bilayers = 500        # Safety limit to avoid infinite loops

# ----------------------------------------------------------------------
# 4. RSD RECURSION (Equation 12 in the paper)
#    D_{n+1} = D_n + alpha * D_n * (D_n / lambda_c_star)^4 
#              * exp( -4 pi^2 sigma_n^2 / D_n^2 )
# ----------------------------------------------------------------------
while D < lambda_Ni / 2.0 and len(D_list) < max_bilayers:
    
    # 4a. Current roughness at this bilayer (statistical growth law)
    sigma_n = np.sqrt(sigma0**2 + h * z_phys)
    
    # 4b. Convert optical thickness D into physical Ni and Ti thicknesses
    #     (Equation 5 in the paper, with refraction correction)
    d_Ni = D / (2 * np.sqrt(1 - (2 * D / lambda_Ni)**2))
    d_Ti = D / (2 * np.sqrt(1 + (2 * D / lambda_cTi_abs)**2))
    
    # 4c. Store the results
    D_list.append(D)
    sigma_list.append(sigma_n)
    d_Ni_list.append(d_Ni)
    d_Ti_list.append(d_Ti)
    
    # 4d. Update cumulative physical thickness (for roughness growth)
    z_phys += d_Ni + d_Ti
    
    # 4e. Compute the next optical thickness using the analytical RSD recursion
    step = alpha * D * (D / lambda_c_star)**4 * np.exp(-4 * np.pi**2 * sigma_n**2 / D**2)
    D += step

# ----------------------------------------------------------------------
# 5. DISPLAY THE RESULTS
# ----------------------------------------------------------------------
n_bilayers = len(D_list)
print(f"Total number of bilayers generated: {n_bilayers}")
print(f"Final optical thickness D: {D_list[-1]:.3f} nm")
print(f"Final physical thickness: Ni = {d_Ni_list[-1]:.3f} nm, Ti = {d_Ti_list[-1]:.3f} nm")
print("\nFirst 10 bilayers (index, D, sigma, d_Ni, d_Ti):")
print(" n    D[nm]   sigma[nm]   d_Ni[nm]   d_Ti[nm]")
for i in range(min(10, n_bilayers)):
    print(f"{i+1:3d}  {D_list[i]:7.3f}  {sigma_list[i]:8.3f}  {d_Ni_list[i]:9.3f}  {d_Ti_list[i]:9.3f}")

# ----------------------------------------------------------------------
# 6. PLOT THE LAYER THICKNESS DISTRIBUTION
# ----------------------------------------------------------------------
bilayer_index = np.arange(1, n_bilayers + 1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot optical thickness D vs bilayer number
ax1.plot(bilayer_index, D_list, 'b-', linewidth=1.5)
ax1.set_ylabel('Optical thickness $D_n$ [nm]')
ax1.set_title(f'RSD Sequence for Ni/Ti, m = {m}, alpha = {alpha}')
ax1.grid(True, linestyle=':', alpha=0.6)

# Plot physical Ni and Ti thicknesses vs bilayer number
ax2.plot(bilayer_index, d_Ni_list, 'r-', label='Ni layer', linewidth=1.5)
ax2.plot(bilayer_index, d_Ti_list, 'g-', label='Ti layer', linewidth=1.5)
ax2.set_xlabel('Bilayer index n (deposition order: thin → thick)')
ax2.set_ylabel('Physical layer thickness [nm]')
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()