import numpy as np
import matplotlib.pyplot as plt

def phi(z):
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z)
    mask = z > 0
    out[mask] = np.exp(-1.0 / z[mask])
    return out

def smooth_step(z):
    pz = phi(z)
    p1z = phi(1 - z)
    denom = pz + p1z
    out = np.zeros_like(z)
    mask = denom > 0
    out[mask] = pz[mask] / denom[mask]
    return out

def cutoff(r, r0, r1):
    return 1 - smooth_step((r - r0) / (r1 - r0))

r = np.linspace(0, 20, 2000)
g1 = cutoff(r, 10, 14)
g4 = cutoff(r, 10, 16)
g5 = cutoff(r, 10, 12)

# Plot 1: full view
plt.figure(figsize=(9, 5))
plt.plot(r, g1, label="g1 = cutoff(r, 10, 14)")
plt.plot(r, g4, label="g4 = cutoff(r, 10, 16)")
plt.plot(r, g5, label="g5 = cutoff(r, 10, 12)")
plt.xlabel("r")
plt.ylabel("gate value")
plt.title("Smooth cutoff gates")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 2: zoom near transition
plt.figure(figsize=(9, 5))
plt.plot(r, g1, label="g1 = cutoff(r, 10, 14)")
plt.plot(r, g4, label="g4 = cutoff(r, 10, 16)")
plt.plot(r, g5, label="g5 = cutoff(r, 10, 12)")
plt.xlim(9, 17)
plt.ylim(-0.02, 1.02)
plt.xlabel("r")
plt.ylabel("gate value")
plt.title("Zoom near the transition")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

