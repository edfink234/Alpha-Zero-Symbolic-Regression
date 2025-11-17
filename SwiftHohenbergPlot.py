import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from numpy import sqrt, exp, cos, sin
from mpl_toolkits.mplot3d import Axes3D
show = False
sech = lambda x: 1.0/np.cosh(x)

PERIODIC_IN_THETA = True
PrintFormula = False
# Define the polar coordinates
SH, r, theta, mu, nu = symbols('\\text{SwiftHohenberg} r theta \\mu \\nu')

# Define the function f as a function of r and theta
f = Function('f')(r, theta)

# Calculate the first Laplacian
laplacian_f = diff(f, r, 2) + (1/r) * diff(f, r) + (1/(r**2)) * diff(f, theta, 2)

# Calculate the double Laplacian
double_laplacian_f = diff(laplacian_f, r, 2) + (1/r) * diff(laplacian_f, r) + (1/(r**2)) * diff(laplacian_f, theta, 2)

swift_hohenberg = mu*f + nu*f*f - f*f*f - (f + 2*laplacian_f + double_laplacian_f)

if PrintFormula:
    print(sp.multiline_latex(SH, swift_hohenberg, 2).replace(r"\frac", r"\dfrac"))

# Create edges instead of centers
N = 331
r_edges = np.linspace(0.01, 10, N)
theta_edges = np.linspace(0, 2*np.pi, N)

# Compute centers
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
R, Theta = np.meshgrid(r_centers, theta_centers)

# Evaluate function on cell centers
Z = sin(R)*sin(Theta) + 0.604 \
    if PERIODIC_IN_THETA else (((0.148475282221305 * Theta) - (np.sin(Theta) * (1.0000132758892615 * np.sin(R)))) - 0.0922858190550785)

# Convert to Cartesian
X = R * np.cos(Theta)
Y = R * np.sin(Theta)

# 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel(r"$f(r,\theta)$")
ax.set_title("Swift-Hohenberg 2D Pattern")
formula_label = None
if PERIODIC_IN_THETA:
    formula_label = r"$f(r,\theta) = \sin(r)\sin(\theta) + 0.604$"
else:
    formula_label = r"$f(r,\theta) = 0.148\cdot\theta - \sin\theta\,\sin r - 0.092$"
fig.colorbar(surf, shrink=0.5, aspect=10, label=formula_label)

# Improve viewing angle
ax.view_init(elev=35, azim=235)

plt.tight_layout()
if show:
    plt.show()
else:
    plt.savefig(f"SwiftHohenberg2D{'Periodic' if PERIODIC_IN_THETA else 'NonPeriodic'}.pdf")
    from os import system
    system(f"open SwiftHohenberg2D{'Periodic' if PERIODIC_IN_THETA else 'NonPeriodic'}.pdf")
