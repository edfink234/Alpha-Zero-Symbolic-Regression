import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
show = False
sech = lambda x: 1.0/np.cosh(x)

PERIODIC_IN_THETA = True
PlotType = "2D"
# Create edges instead of centers
N = 331
r_edges = np.linspace(0.01, 10, N)
theta_edges = np.linspace(0, 2*np.pi, N)

round_floats = lambda expr, ndigits: expr.xreplace({f: Float(round(float(f), ndigits)) for f in expr.atoms(Float)})
f_per_idx = 2

r, theta = symbols('r theta')
f =  [-0.998846776839887*0.999950000416665**(r**4)*sin(r)*sin(theta) + 2.71782596428238e-13*10.36319**(r + 0.01) + 0.00164034101997398*theta - (r/(r + 2))**(r + 6.35319) + 0.6448561035289, sin(r)*sin(theta)+0.604, sin(r)*sin(theta)][f_per_idx] \
        if PERIODIC_IN_THETA else \
        (((0.148475282221305 * theta) - (sin(theta) * (1.0000132758892615 * sin(r)))) - 0.0922858190550785)

# Compute centers
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
R, Theta = np.meshgrid(r_centers, theta_centers)

# Evaluate function on cell centers
Z = [-0.998846776839887*0.999950000416665**(R**4)*np.sin(R)*np.sin(Theta) + 2.71782596428238e-13*10.36319**(R + 0.01) + 0.00164034101997398*Theta - (R/(R + 2))**(R + 6.35319) + 0.6448561035289, np.sin(R)*np.sin(Theta) + 0.604, np.sin(R)*np.sin(Theta)][f_per_idx] \
    if PERIODIC_IN_THETA else (((0.148475282221305 * Theta) - (np.sin(Theta) * (1.0000132758892615 * np.sin(R)))) - 0.0922858190550785)

# Convert to Cartesian
X = R * np.cos(Theta)
Y = R * np.sin(Theta)

print(f'sp.latex(f) = {latex(f)}')
formula_label = latex(round_floats(f, 5), mul_symbol='dot')
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

    contour = ax.contourf(
        X, Y, Z,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        levels = levels,
        extend = "both" # shows out-of-range values correctly
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
