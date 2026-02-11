import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from math import pi
from numpy import linalg as LA
show = False
np.sech = lambda x: 1.0/np.cosh(x)

PERIODIC_IN_THETA = True
PRINT_LATEX_ONLY = False
PlotType = "2D"
# Create edges instead of centers
N = 1000
r_max = 10
r_edges = np.linspace(0.01, r_max, N)
theta_edges = np.linspace(0, 2*np.pi, N, endpoint=False)

round_floats = lambda expr, ndigits: expr.xreplace({f: Float(round(float(f), ndigits)) for f in expr.atoms(Float)})
f_per_idx = 5
mu, nu = 1, 1
r, theta = symbols('r theta')
f =  [sin(r)*sin(theta), \
      sin(r)*sin(theta)+0.604, \
      0.998846776839887*0.999950000416665**(r**4)*sin(r)*sin(theta) + 0.604, \
          0.88898139159952*0.999884875453817**(r**4.03)*sqrt(1 - cos(r)**2)*sin(theta) + 0.760176150613572, \
      -0.28580222883408**(r + 10)*(1.01 - sin(theta))*(167.620651926117*r**7.38905609893065 + 0.000105912014609458) + 0.833098208613807*0.999884875453817**(r**(17/4))*sqrt(1 - cos(r)**2)*sin(theta)  + 0.797073913381706, \
      -182.159206127457*0.28580222883408**(r + 10)*(1.02 - sin(theta))*(r + 0.00999991666708333)**7.50905609893065 + 0.745258709383936*0.999884875453817**(r**4.25)*sqrt(1 - cos(r)**2)*sin(theta + 6.28319)  + 0.815307524508096][f_per_idx] \
        if PERIODIC_IN_THETA else \
        (((0.148475282221305 * theta) - (sin(theta) * (1.0000132758892615 * sin(r)))) - 0.0922858190550785)

formula_label = latex(round_floats(f, 5), mul_symbol='dot')
print(f'f = {formula_label}')

if PRINT_LATEX_ONLY:
    exit()
# Calculate the first Laplacian (Laplacian of f)
laplacian_f = diff(f, r, 2) + (1/r) * diff(f, r) + (1/(r**2)) * diff(f, theta, 2)

# Calculate the double Laplacian (Laplacian of the first Laplacian)
double_laplacian_f = diff(laplacian_f, r, 2) + (1/r) * diff(laplacian_f, r) + (1/(r**2)) * diff(laplacian_f, theta, 2)

swift_hohenberg = mu*f + nu*f*f - f*f*f - (f + 2*laplacian_f + double_laplacian_f)
r_vals, theta_vals = np.meshgrid(np.linspace(0.01, r_max, N), np.linspace(0, 2*pi, N))
func = lambdify((r, theta), swift_hohenberg)
func_vals = func(r_vals, theta_vals)
squared_norm_error = LA.norm(func_vals.flatten())**2
mean_squared_error = squared_norm_error / func_vals.size

# Compute centers
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
R, Theta = np.meshgrid(r_centers, theta_centers)

# Evaluate function on cell centers
Z = [np.sin(R)*np.sin(Theta), \
     np.sin(R)*np.sin(Theta) + 0.604, \
     0.998846776839887*0.999950000416665**(R**4)*np.sin(R)*np.sin(Theta) + 0.604, \
     0.88898139159952*0.999884875453817**(R**4.03)*np.sqrt(1 - np.cos(R)**2)*np.sin(Theta) - (R/(R + 1.01))**((np.sin(Theta) + 10.0/R)*(R + np.sin(R) + 0.01)) + 0.760176150613572, \
     -0.28580222883408**(R + 10)*(1.01 - np.sin(Theta))*(167.620651926117*R**7.38905609893065 + 0.000105912014609458) + 0.833098208613807*0.999884875453817**(R**(17/4))*np.sqrt(1 - np.cos(R)**2)*np.sin(Theta)  + 0.797073913381706, \
     -182.159206127457*0.28580222883408**(R + 10)*(1.02 - np.sin(Theta))*(R + 0.00999991666708333)**7.50905609893065 + 0.745258709383936*0.999884875453817**(R**4.25)*np.sqrt(1 - np.cos(R)**2)*np.sin(Theta + 6.28319) + 0.815307524508096][f_per_idx] \
    if PERIODIC_IN_THETA else (((0.148475282221305 * Theta) - (np.sin(Theta) * (1.0000132758892615 * np.sin(R)))) - 0.0922858190550785)

# Convert to Cartesian
X = R * np.cos(Theta)
Y = R * np.sin(Theta)

formula_label = r"$f(r,\theta) = " + formula_label + "$"

# 3D Plot
if PlotType == "3D":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(r"$f(r,\theta)$")
    ax.set_title(f"Swift-Hohenberg 2D Pattern, MSE = {mean_squared_error:.3f}")
    fig.colorbar(surf, shrink=0.5, aspect=10, label=formula_label)

    # Improve viewing angle
    ax.view_init(elev=35, azim=235)
#2D Plot
else:
    fig, ax = plt.subplots()

    # Filled contour plot
    vmin, vmax = -1.0, 1.0
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
    ax.set_title(f"Swift-Hohenberg 2D Pattern, MSE = {mean_squared_error:.3f}")

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(formula_label, fontsize=7)
    cbar.set_ticks([-1, 0, 1])

    ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
domain_str = f"_r_{int(r_edges[0])}_{int(r_edges[-1])}_theta_{int(theta_edges[0])}_{int(theta_edges[-1])}_N_{N}"
if show:
    plt.show()
else:
    filename = f"SwiftHohenbergPlots/SwiftHohenberg2D{'Periodic'+str(f_per_idx)+domain_str if PERIODIC_IN_THETA else 'NonPeriodic'}.pdf"
    plt.savefig(filename)
    print(f"Saved {filename}")
    from os import system
    filename = filename[:-4]
    system(f"open {filename}.pdf")
    system(f"sips -s format png -s dpiWidth 480 -s dpiHeight 480 -z 2400 2400 {filename}.pdf --out {filename}.png")
