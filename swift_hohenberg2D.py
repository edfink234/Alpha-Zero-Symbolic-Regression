from sympy import *
import sympy as sp
import numpy as np
from sympy.utilities.lambdify import lambdify
from scipy.optimize import least_squares
from numpy import linalg as LA
from math import pi
import matplotlib.pyplot as plt

sech=lambda x:1/cosh(x)

# Define the polar coordinates
SH = symbols('\\text{SwiftHohenberg} r theta mu nu')
r, theta = symbols('r theta')
mu, nu = 1, 1
# Define the function f as a function of r and theta
GENERIC = False
PERIODIC_IN_THETA = True
COMPUTE_NUMERIC = False
f = None
if GENERIC:
    f = Function('f')(r, theta)
else:
    f =  sin(r)*sin(theta) + 0.604 \
        if PERIODIC_IN_THETA else \
        (((0.148475282221305 * theta) - (sin(theta) * (1.0000132758892615 * sin(r)))) - 0.0922858190550785)

latex_f = sp.latex(f)
latex_f = latex_f.replace(r"(r", r"(\sqrt{x^2 + y^2}")
latex_f = latex_f.replace(r"\theta", r"\arctan{\dfrac{y}{x}}")
#print(latex_f)

# Calculate the first Laplacian (Laplacian of f)
laplacian_f = diff(f, r, 2) + (1/r) * diff(f, r) + (1/(r**2)) * diff(f, theta, 2)

# Calculate the double Laplacian (Laplacian of the first Laplacian)
double_laplacian_f = diff(laplacian_f, r, 2) + (1/r) * diff(laplacian_f, r) + (1/(r**2)) * diff(laplacian_f, theta, 2)

swift_hohenberg = mu*f + nu*f*f - f*f*f - (f + 2*laplacian_f + double_laplacian_f)

print(f"swift_hohenberg = {str(swift_hohenberg.evalf()).replace('r','r_val').replace('theta', 'theta_val')}")

# print(*swift_hohenberg.args, sep="\n")
r_vals, theta_vals = [None]*2
func_vals = None
N = 330
if not GENERIC:
    r_vals, theta_vals = np.meshgrid(np.linspace(0.01, 10, N), np.linspace(0, 2*pi, N))
    f_SR = lambdify((r, theta), f)
    f_SR_r = lambdify((r, theta), f_r := diff(f, r))
    f_SR_theta = lambdify((r, theta), f_theta := diff(f, theta))

    print(f"Variance of f = {np.var(f_SR(r_vals, theta_vals))}")
    print(f"Max(∂f/∂r) = {np.max(f_SR_r_vals:=f_SR_r(r_vals, theta_vals))}")
    print(f"Max(∂f/∂θ) = {np.max(f_SR_theta_vals:=f_SR_theta(r_vals, theta_vals))}")
    print(f"Median(∂f/∂r) = {np.median(np.sort(f_SR_r_vals))}")
    print(f"Median(∂f/∂θ) = {np.median(np.sort(f_SR_theta_vals))}")

    func = lambdify((r, theta), swift_hohenberg)
    func_vals = func(r_vals, theta_vals)
#    print(f"func_vals.size = {func_vals.size}")
#    print(f"func_vals.shape = {func_vals.shape}")
#    print(f"func_vals = {func_vals}");
#    print(f"diff(func_vals, axis = 0) = {np.diff(func_vals, axis = 0)}") #diff(f, theta)
#    print(f"diff(func_vals, axis = 1) = {np.diff(func_vals, axis = 1)}") #diff(f, r)
    squared_norm_error = LA.norm(func_vals.flatten())**2
    print(f"squared-norm error = {squared_norm_error}")
#    print(sp.multiline_latex(SH, swift_hohenberg, 2).replace(r"\frac", r"\dfrac"))
    mean_squared_error = squared_norm_error / func_vals.size
    print(f"mean-squared_error = {mean_squared_error}")

#ROOT-FINDING#
##############

# Build grids (overwrite any previous r_vals/theta_vals for the solver part)
N = 50
Nr = N
Nth = N
th_vec = np.linspace(0.0, 2.0*np.pi, Nth, endpoint=False)  # periodic, no duplicate endpoint
r_edges = np.linspace(0.0, 10.0, Nr + 1)                   # edges include r=0
r_vec   = 0.5*(r_edges[:-1] + r_edges[1:])                 # midpoints: strictly r>0
func_vals = func(r_vec, th_vec)
#print(f"Mean-squared error = {(LA.norm(func_vals.flatten())**2) / func_vals.size}")
#print(f"r_vec = {r_vec}")
dr  = float(r_edges[1] - r_edges[0])
dth = float(th_vec[1] - th_vec[0])

assert Nr >= 4 and Nth >= 4, "Need at least 4 points each way for the stencils."
assert dr  > 0 and np.isfinite(dr),  f"Bad dr: {dr}"
assert dth > 0 and np.isfinite(dth), f"Bad dth: {dth}"
assert np.all(r_vec > 0),            "Radial midpoints must be > 0."

# Mesh (use 'ij' so r varies along axis 0, theta along axis 1)
r_vals, theta_vals = np.meshgrid(r_vec, th_vec, indexing='ij')

# --- Finite-difference helpers (2nd-order one-sided in r and theta) ---
def dtheta2(F):
    if PERIODIC_IN_THETA:
        Fm1 = np.roll(F, 1, axis=1)
        Fp1 = np.roll(F, -1, axis=1)
        return (Fm1 - 2.0*F + Fp1) / (dth**2)
    else:
        G = np.empty_like(F)
        G[:, 1:-1] = (F[:, 2:] - 2.0*F[:, 1:-1] + F[:, :-2]) / (dth**2)
        G[:, 0]  = ( 2.0*F[:, 0]  - 5.0*F[:, 1]  + 4.0*F[:, 2]  - F[:, 3]  ) / (dth**2)
        G[:, -1] = ( 2.0*F[:, -1] - 5.0*F[:, -2] + 4.0*F[:, -3] - F[:, -4] ) / (dth**2)
        return G

#f'(x) = (f(x+h)-f(x))/h -> f''(x) = (f'(x+h) - f'(x))/h
def dr_first(F):
    G = np.empty_like(F)
    G[1:-1, :] = (F[2:, :] - F[:-2, :]) / (2.0*dr)
    # 2nd-order one-sided at boundaries
    G[0,  :]   = (-3.0*F[0, :] + 4.0*F[1, :] - 1.0*F[2, :]) / (2.0*dr)
    G[-1, :]   = ( 3.0*F[-1, :] - 4.0*F[-2, :] + 1.0*F[-3, :]) / (2.0*dr)
    return G

def dr_second(F):
    H = np.empty_like(F)
    H[1:-1, :] = (F[2:, :] - 2.0*F[1:-1, :] + F[:-2, :]) / (dr**2)
    # 2nd-order one-sided at boundaries
    H[0,  :]   = ( 2.0*F[0, :] - 5.0*F[1, :] + 4.0*F[2, :] - F[3, :] ) / (dr**2)
    H[-1, :]   = ( 2.0*F[-1, :] - 5.0*F[-2, :] + 4.0*F[-3, :] - F[-4, :] ) / (dr**2)
    return H

r_col = r_vec.reshape(-1, 1)                  # shape (Nr,1)
inv_r  = 1.0 / r_col                          # safe: r_vec > 0
inv_r2 = inv_r**2

def laplacian_polar(F):
    Fr  = dr_first(F)
    Frr = dr_second(F)
    Ftt = dtheta2(F)
    return Frr + inv_r*Fr + inv_r2*Ftt

# --- Residual operator R(U) flattened -> cost function ---
def residual_vec(x):
    U = x.reshape(Nr, Nth)
    Lu  = laplacian_polar(U)
    L2u = laplacian_polar(Lu)  # Δ(ΔU)
    R = mu*U + nu*(U*U) - (U*U*U) - (U + 2.0*Lu + L2u)
    return R.ravel()

# Initial seed: use your field f(r,theta) (NOT the residual) on the new grid
f_func = lambdify((r, theta), f)
U0 = f_func(r_vals, theta_vals)
x0 = U0.ravel().copy()

# Sanity checks before solving
R0 = residual_vec(x0)
print(f"Grid checks: dr={dr}, dth={dth}, r_min={r_vec.min()}, r_max={r_vec.max()}")
print("Any nonfinite in U0? ", np.any(~np.isfinite(U0)))
print("Any nonfinite in R0? ", np.any(~np.isfinite(R0)))
print(f"Initial ||R(U0)||^2 = {float(np.dot(R0, R0))}")
print(f"Initial mean-squared residual = {float(np.dot(R0, R0))/R0.size}")
if not COMPUTE_NUMERIC:
    exit()
tolerance = 1e-6

# Solve the nonlinear system on the grid
res = least_squares(
    residual_vec, x0,
    method='trf',
    ftol=tolerance, xtol=tolerance, gtol=tolerance,
    max_nfev=100, verbose=2
)

x_star = res.x
R_star = residual_vec(x_star)
final_sq_norm = float(np.dot(R_star, R_star))

print("\n=== Swift–Hohenberg field solve (grid) ===")
print("Success:", res.success)
print("Message:", res.message)
print("Function evals:", res.nfev)
print(f"Final ||R(U*)||^2 = {final_sq_norm}")
print(f"Final mean-squared residual = {final_sq_norm / R_star.size}")

U_star = x_star.reshape(Nr, Nth)

#PLOTTING
#========

# Build polar grid in the same "style" as your snippet (default 'xy' indexing)
R, Theta = np.meshgrid(r_vec, th_vec)   # shapes: (Nth, Nr)

# Match Z's shape to (Nth, Nr)
Z = U_star.T

# Convert to Cartesian
X = R * np.cos(Theta)
Y = R * np.sin(Theta)

# 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(r,θ)")
ax.set_title("Swift–Hohenberg 2D Solution (U★) — 3D Surface")
fig.colorbar(surf, shrink=0.5, aspect=10, label="f(r, θ)")

ax.view_init(elev=35, azim=235)
plt.tight_layout()
plt.savefig(f"LeastSquaresSeededBySRSolve{'Periodic' if PERIODIC_IN_THETA else 'NonPeriodic'}.pdf")
