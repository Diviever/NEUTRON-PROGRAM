import numpy as np
import matplotlib.pyplot as plt

# Cold neutron parameters
lam = 5.0              # wavelength in Angstrom
k = 2*np.pi/lam        # wavevector in 1/Angstrom
E = 3.27               # neutron energy in meV

# Effective optical potentials for Ni and Ti (meV)
V_Ni = 245e-6          # ~245 neV
V_Ti = -50e-6          # ~-50 neV

# Layer thicknesses in Angstrom
d_Ni = 100.0
d_Ti = 150.0

# Number of Ni/Ti bilayers
N_bilayers = 10

# Build the layer sequence: Ni, Ti, Ni, Ti, ...
potentials = []
thicknesses = []
for _ in range(N_bilayers):
    potentials.extend([V_Ni, V_Ni])
    thicknesses.extend([d_Ni, d_Ni])

# Perpendicular wavevector range
q = np.linspace(1e-4, 0.05, 1000)

# Critical wavevector for Ni
q_c_Ni = k * np.sqrt(V_Ni/E)

def layer_matrix(q_val, V, d, p_prev):

    p = np.sqrt(q_val**2 - (k**2 * V / E) + 0j)

    M_int = (1/(2*p_prev)) * np.array([
        [p_prev + p, p_prev - p],
        [p_prev - p, p_prev + p]
    ])

    P = np.array([
        [np.exp(-1j*p*d), 0],
        [0, np.exp( 1j*p*d)]
    ])

    return M_int @ P, p

r = np.empty_like(q, dtype=complex)
t = np.empty_like(q, dtype=complex)

for idx, qi in enumerate(q):
    M = np.eye(2, dtype=complex)
    p_prev = qi   

    for V, d in zip(potentials, thicknesses):
        M_layer, p_prev = layer_matrix(qi, V, d, p_prev)
        M = M @ M_layer

    p = qi
    M_int = (1/(2*p_prev)) * np.array([
        [p_prev + p, p_prev - p],
        [p_prev - p, p_prev + p]
    ])
    M = M @ M_int

    t[idx] = 1.0 / M[0, 0]
    r[idx] = M[1, 0] / M[0, 0]

# Plot
plt.figure(figsize=(7, 5))
plt.plot(q, np.abs(r)**2, label=r"$|r|^2$ reflectance")
plt.axvline(q_c_Ni, color="gray", ls="--", label=r"$q_c^{\mathrm{Ni}}$")

plt.xlabel(r"$q$ (Å$^{-1}$)")
plt.ylabel("Reflectance")
plt.title(f"Ni/Ti multilayer: {N_bilayers} bilayers")
plt.legend()
plt.grid()
plt.show()