import numpy as np
import matplotlib.pyplot as plt

# Realistic cold-neutron parameters
lam = 5.0              # wavelength in Angstrom
k = 2*np.pi/lam        # wavevector in 1/Angstrom
E = 3.27               # meV
V = 252e-6              # meV Nikel const

b = 250.0              # barrier thickness in Angstrom
D = 250.0              # gap thickness in Angstrom

Number_of_additional_barriers = 6

# Critical perpendicular wavevector
q_c = k * np.sqrt(V/E)

# Perpendicular wavevector range (avoid q = 0)
q_max = 0.02           # 1/Angstrom, corresponds to ~2.3 deg grazing angle
q = np.linspace(1e-4, q_max, 500)

# Wavevector inside the barrier (complex for q < q_c)
q_b = np.sqrt(q**2 - q_c**2 + 0j)

def barrier_matrix(qi, qbi, width):
    M1 = (1/(2*qi)) * np.array([
        [qi + qbi, qi - qbi],
        [qi - qbi, qi + qbi]
    ])
    P = np.array([
        [np.exp( 2j*qbi*width), 0],
        [0, np.exp(-2j*qbi*width)]
    ])
    M3 = (1/(2*qbi)) * np.array([
        [qi + qbi, qbi - qi],
        [qbi - qi, qi + qbi]
    ])
    return M1 @ P @ M3

r = np.empty_like(q, dtype=complex)
t = np.empty_like(q, dtype=complex)

for i, (qi, qbi) in enumerate(zip(q, q_b)):
    M_bar = barrier_matrix(qi, qbi, b)

    P_gap = np.array([
        [np.exp( 1j*qi*D), 0],
        [0, np.exp(-1j*qi*D)]
    ])

    M_tot = M_bar

    for j in range(Number_of_additional_barriers):

        M_tot = M_bar @ P_gap @ M_tot

    r[i] = -M_tot[0, 1] / M_tot[0, 0]
    t[i] = M_tot[1, 1] + M_tot[1, 0] * r[i]

plt.figure(figsize=(7, 5))
plt.plot(q, np.abs(r)**2, label=r"$|r|^2$ (reflectance)")
# plt.plot(q, np.abs(t)**2, label=r"$|t|^2$ (transmittance)")

plt.xlabel(r"$q$ (Å$^{-1}$)")
plt.ylabel("Reflectance")
plt.title(f"Cold neutron: {Number_of_additional_barriers + 1} Nickel barrier, D = 400 Å")
plt.legend()
plt.grid()
plt.show()