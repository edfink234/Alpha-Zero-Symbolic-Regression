import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
show = False
np.sech = lambda x: 1.0/np.cosh(x)

PERIODIC_IN_THETA = True
PlotType = "2D"
# Create edges instead of centers
N = 331
r_edges = np.linspace(0.01, 10, N)
theta_edges = np.linspace(0, 2*np.pi, N, endpoint=False)

round_floats = lambda expr, ndigits: expr.xreplace({f: Float(round(float(f), ndigits)) for f in expr.atoms(Float)})
f_per_idx = 4

r, theta = symbols('r theta')
f =  [sin(r)*sin(theta), sin(r)*sin(theta)+0.604, 0.998846776839887*0.999950000416665**(r**4)*sin(r)*sin(theta) + 0.604, 0.88898139159952*0.999884875453817**(r**4.03)*sqrt(1 - cos(r)**2)*sin(theta) - (r/(r + 1.01))**((sin(theta) + 10.0/r)*(r + sin(r) + 0.01)) + 0.760176150613572][f_per_idx] \
        if PERIODIC_IN_THETA else \
        (((0.148475282221305 * theta) - (sin(theta) * (1.0000132758892615 * sin(r)))) - 0.0922858190550785)

# Compute centers
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
R, Theta = np.meshgrid(r_centers, theta_centers)

# Evaluate function on cell centers
Z = [np.sin(R)*np.sin(Theta), np.sin(R)*np.sin(Theta) + 0.604, 0.998846776839887*0.999950000416665**(R**4)*np.sin(R)*np.sin(Theta) + 0.604, 0.88898139159952*0.999884875453817**(R**4.03)*np.sqrt(1 - np.cos(R)**2)*np.sin(Theta) - (R/(R + 1.01))**((np.sin(Theta) + 10.0/R)*(R + np.sin(R) + 0.01)) + 0.760176150613572][f_per_idx] \
    if PERIODIC_IN_THETA else (((0.148475282221305 * Theta) - (np.sin(Theta) * (1.0000132758892615 * np.sin(R)))) - 0.0922858190550785)

# Convert to Cartesian
X = R * np.cos(Theta)
Y = R * np.sin(Theta)

print(f'sp.latex(f) = {latex(f)}')
formula_label = latex(round_floats(f, 4), mul_symbol='dot')
formula_label = r"$f(r,\theta) = " + formula_label + "$"

# 3D Plot
if PlotType == "3D":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(r"$f(r,\theta)$")
    ax.set_title("Swift-Hohenberg 2D Pattern")
    fig.colorbar(surf, shrink=0.5, aspect=10, label=formula_label)

    # Improve viewing angle
    ax.view_init(elev=35, azim=235)
#2D Plot
else:
    fig, ax = plt.subplots()

    # Filled contour plot
    vmin, vmax = -2.0, 2.0
    levels = np.linspace(vmin, vmax, 100)

    # Close the periodic seam in theta for plotting
    Xc = np.vstack([X, X[0:1, :]])
    Yc = np.vstack([Y, Y[0:1, :]])
    Zc = np.vstack([Z, Z[0:1, :]])

    contour = ax.contourf(
        Xc, Yc, Zc,
        cmap="viridis",
        vmin=vmin, vmax=vmax,
        levels=levels,
        extend="both"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Swift-Hohenberg 2D Pattern")

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(formula_label, fontsize=7)
    cbar.set_ticks([-2, -1, 0, 1, 2])

    ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
if show:
    plt.show()
else:
    plt.savefig(f"SwiftHohenberg2D{'Periodic'+str(f_per_idx) if PERIODIC_IN_THETA else 'NonPeriodic'}.pdf")
    print(f"Saved SwiftHohenberg2D{'Periodic'+str(f_per_idx) if PERIODIC_IN_THETA else 'NonPeriodic'}.pdf")
    from os import system
    system(f"open SwiftHohenberg2D{'Periodic'+str(f_per_idx) if PERIODIC_IN_THETA else 'NonPeriodic'}.pdf")
