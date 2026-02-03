'''
    Questions for Priya:
        1. Say that we have an approximate solution f to a PDE such that when we plug f(x) into 
           the PDE (i.e. symbolically differentiate) and evaluate the resulting expression PDE(f(x))
           on a collocation grid of x-points, we get a resulting vector L whose norm is some small number ε. 
           The question we have is that, if instead of evaluating the derivatives symbolically, we evaluate them 
           numerically, is the numerical-differentiation faulty by definition if, after proceeding with the same process 
           as above, we get a resulting norm(L_{numerical}) that differs substantially from ε?
           -> Yes, the symbolic residual is the right one
        2. Why is it that a numerical solver starting with a smaller starting numerically-computed MSE can take longer to converge than a guess that has a larger starting numerically-computed MSE?
            -> Might be in a different basin of attraction
            -> A lot of newton/quasi-newton solvers compute/approximate a Jacobian, which may be ill-conditioned depending on the initial seed. For quasi-newton especially, the Jacobian approximate could be poor if the true-Jacobian is ill-conditioned
        3. Does there usually/~always exist a sufficiently accurate numerical scheme that gets the close enough to the symbolically computed MSE 
            3 a. If not, then how can we modify the SR search to penalize solutions that would be difficult to use numerically?
                -> Add a term to the loss that penalizes the condition-number of the Jacobian 
        4. Does it make more sense to go the direction of reducing the error of the SR f or to explore the numerics side? 
            -> We definitely want to reduce the SR error as much as possible given that the SR-solution MSE is ~ mesh-independent
                -> Right now the numerics is nice but not central if we can get the SR loss down sufficiently in a mesh-independent manner
            -> Makes sense to see if we can use our solution for the mu=1, nu=1 case as an initial seed for a general function solution f(r, theta; mu, nu) 
        5. If we do numerical continuation in (mu, nu), does it make more sense to do it from a phenmonelogical perspective (to gain e.g. insight into the "functional forms" of the solutions in the SR) or just in the numerics
            -> Makes sense to see if we can use our solution for the mu=1, nu=1 case as an initial seed for a general function solution f(r, theta; mu, nu) 
        
        https://iopscience.iop.org/article/10.1088/1361-6544/acc508/pdf 
'''
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
PRINT_SH = False
f = None
if GENERIC:
    f = Function('f')(r, theta)
else:
    f =  [-0.998846776839887*0.999950000416665**(r**4)*sin(r)*sin(theta) + 2.71782596428238e-13*10.36319**(r + 0.01) + 0.00164034101997398*theta - (r/(r + 2))**(r + 6.35319) + 0.6448561035289, \
        sin(r)*sin(theta)+0.604, \
        sin(r)*sin(theta), \
        0.998846776839887*0.999950000416665**(r**4)*sin(r)*sin(theta) + 0.604, 0.88898139159952*0.999884875453817**(r**4.03)*sqrt(1 - cos(r)**2)*sin(theta) + 0.760176150613572,\
        -0.28580222883408**(r + 10)*(1.01 - sin(theta))*(167.620651926117*r**7.38905609893065 + 0.000105912014609458) + 0.833098208613807*0.999884875453817**(r**4.25)*sqrt(1 - cos(r)**2)*sin(theta) + 0.797073913381706,\
        -182.159206127457*0.28580222883408**(r + 10)*(1.02 - sin(theta))*(r + 0.00999991666708333)**7.50905609893065 + 0.745258709383936*0.999884875453817**(r**4.25)*sqrt(1 - cos(r)**2)*sin(theta + 6.28319)  + 0.815307524508096][-1] \
        if PERIODIC_IN_THETA else \
        (((0.148475282221305 * theta) - (sin(theta) * (1.0000132758892615 * sin(r)))) - 0.0922858190550785)
#11284
#38469
#-0.998846776839887*0.999950000416665**(r**4)*sin(r)*sin(theta) + 2.71782596428238e-13*10.36319**(r + 0.01) + 0.00164034101997398*theta - (r/(r + 2))**(r + 6.35319) + 0.6448561035289
#A(r)*f(r,theta) + const + hoc. (higher-order corrections)


print(f"f = {f}\n")
latex_f = sp.latex(f)
latex_f = latex_f.replace(r"(r", r"(\sqrt{x^2 + y^2}")
latex_f = latex_f.replace(r"\theta", r"\arctan{\dfrac{y}{x}}")
#print(latex_f)

# Calculate the first Laplacian (Laplacian of f)
laplacian_f = diff(f, r, 2) + (1/r) * diff(f, r) + (1/(r**2)) * diff(f, theta, 2)

# Calculate the double Laplacian (Laplacian of the first Laplacian)
double_laplacian_f = diff(laplacian_f, r, 2) + (1/r) * diff(laplacian_f, r) + (1/(r**2)) * diff(laplacian_f, theta, 2)

swift_hohenberg = mu*f + nu*f*f - f*f*f - (f + 2*laplacian_f + double_laplacian_f)
if PRINT_SH:
    print(f"swift_hohenberg = {str(swift_hohenberg.evalf()).replace('r','r_val').replace('theta', 'theta_val')}\n")

# print(*swift_hohenberg.args, sep="\n")
r_vals, theta_vals = [None]*2
func_vals = None
N = 1000
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
