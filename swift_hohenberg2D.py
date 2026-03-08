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
from warnings import filterwarnings
from sympy.utilities.autowrap import ufuncify

filterwarnings('ignore')
#sech=lambda x:1/cosh(x)

# -------------------------
# Helpers: parameterization
# -------------------------
def extract_parameter_atoms(expr, max_params=12):
    floats = list(expr.atoms(sp.Float))
    floats = [c for c in floats if float(c) not in (0.0, 1.0)]
    # frequency heuristic
    floats_sorted = sorted(floats, key=lambda c: expr.count(c), reverse=True)
    picked, seen = [], set()
    for c in floats_sorted:
        if c in seen:
            continue
        seen.add(c)
        picked.append(c)
        if len(picked) >= max_params:
            break
    return picked

def make_parametrized_expr(expr, atoms):
    params = [sp.Symbol(f"p{i}", real=True) for i in range(len(atoms))]
    subs_map = {atoms[i]: params[i] for i in range(len(atoms))}
    expr_param = expr.xreplace(subs_map)
    p0 = np.array([float(a) for a in atoms], dtype=float)
    return expr_param, params, p0, subs_map

def build_sh_residual(f_expr_param, r, theta, mu=1.0, nu=1.0):
    laplacian_f = diff(f_expr_param, r, 2) + (1/r) * diff(f_expr_param, r) + (1/(r**2)) * diff(f_expr_param, theta, 2)
    double_laplacian_f = diff(laplacian_f, r, 2) + (1/r) * diff(laplacian_f, r) + (1/(r**2)) * diff(laplacian_f, theta, 2)
    swift_hohenberg = mu*f_expr_param + nu*f_expr_param*f_expr_param - f_expr_param**3 - (f_expr_param + 2*laplacian_f + double_laplacian_f)
    return swift_hohenberg

def make_grids(N, r_max, stride=1):
    r_lin = np.linspace(0.01, r_max, N)[::stride]
    th_lin = np.linspace(0.0, 2*pi, N)[::stride]
    return np.meshgrid(r_lin, th_lin)

def sh_metrics(sh_num, p, r_vals, th_vals):
    res = np.asarray(sh_num(r_vals, th_vals, *p), dtype=float).ravel()
    sqnorm = float(LA.norm(res)**2)
    mse = sqnorm / res.size
    return sqnorm, mse

def variance_of_f(f_num, p, r_vals, th_vals):
    vals = np.asarray(f_num(r_vals, th_vals, *p), dtype=float)
    return float(np.var(vals))

# -------------------------
# Naive random search
# -------------------------
def naive_random_optimize(
    f_expr,
    r, theta,
    f_thresh,
    N=1000,
    mu=1.0, nu=1.0,
    # compute cost controls:
    opt_stride=5,          # evaluate candidates on strided grids
    max_params=12,
    n_iters=2000,
    step_scale=0.05,       # relative perturbation size
    additive_step=1e-3,    # absolute perturbation fallback
    # acceptance:
    use_annealing=False,
    T0=0.1,
    Tf=1e-4,
    # variance handling:
    reject_if_var_below=False,
    var_penalty_weight=1e6,
    rng_seed=0,
    print_every=50
):
    rng = np.random.default_rng(rng_seed)

    # 1) Parameterize expression
    atoms = extract_parameter_atoms(f_expr, max_params=max_params)
    f_param, params, p0, subs_map = make_parametrized_expr(f_expr, atoms)

    # 2) Build SH residual (symbolic) and lambdify
    sh_param = build_sh_residual(f_param, r, theta, mu=mu, nu=nu)
    f_num = lambdify((r, theta, *params), f_param, modules="numpy")
    sh_num = lambdify((r, theta, *params), sh_param, modules="numpy")

    # 3) Grids
    r1_opt, th1_opt = make_grids(N, r_max=10.0, stride=opt_stride)
    r2_opt, th2_opt = make_grids(N, r_max=100.0, stride=opt_stride)

    # Full grids only for initial + final reporting (expensive at N=1000)
    r1_full, th1_full = make_grids(N, r_max=10.0, stride=1)
    r2_full, th2_full = make_grids(N, r_max=100.0, stride=1)

    # 4) Initial residuals (stored!)
    initial_sq1, initial_mse1 = sh_metrics(sh_num, p0, r1_full, th1_full)
    initial_sq2, initial_mse2 = sh_metrics(sh_num, p0, r2_full, th2_full)
    initial_var1 = variance_of_f(f_num, p0, r1_full, th1_full)
    initial_var2 = variance_of_f(f_num, p0, r2_full, th2_full)

    initial_residual = {
        "grid1_rmax10_sqnorm": initial_sq1,
        "grid1_rmax10_mse": initial_mse1,
        "grid2_rmax100_sqnorm": initial_sq2,
        "grid2_rmax100_mse": initial_mse2,
        "combined_mse": (initial_mse1 + initial_mse2) / 2.0,
        "var_grid1": initial_var1,
        "var_grid2": initial_var2,
    }

    # Objective on OPT grids (fast)
    def objective(p):
        # base: mean of MSEs across the two grids
        _, mse1 = sh_metrics(sh_num, p, r1_opt, th1_opt)
        _, mse2 = sh_metrics(sh_num, p, r2_opt, th2_opt)
        base = 0.5 * (mse1 + mse2)

        v1 = variance_of_f(f_num, p, r1_opt, th1_opt)
        v2 = variance_of_f(f_num, p, r2_opt, th2_opt)

        if reject_if_var_below and (v1 < f_thresh[0] or v2 < f_thresh[1]):
            return np.inf, v1, v2

        # soft penalty (still useful even if not rejecting)
        pen = 0.0
        if v1 < f_thresh[0]:
            pen += (f_thresh[0] - v1)
        if v2 < f_thresh[1]:
            pen += (f_thresh[1] - v2)
        base_plus = base + var_penalty_weight * pen
        return base_plus, v1, v2

    # 5) Initialize best
    p_best = p0.copy()
    best_obj, best_v1, best_v2 = objective(p_best)

    # 6) Random search loop
    for k in range(1, n_iters + 1):
        # annealing temperature schedule
        if use_annealing:
            t = (k - 1) / max(1, n_iters - 1)
            T = T0 * (Tf / T0) ** t
        else:
            T = 0.0

        # propose perturbation: mix relative + absolute
        rel = step_scale * (np.abs(p_best) + 1.0)
        delta = rng.normal(0.0, rel) + rng.normal(0.0, additive_step, size=p_best.shape)
        p_try = p_best + delta

        obj_try, v1_try, v2_try = objective(p_try)

        accept = False
        if obj_try < best_obj:
            accept = True
        elif use_annealing and np.isfinite(obj_try) and np.isfinite(best_obj) and T > 0:
            # accept worse move with probability exp(-(Δ)/T)
            d = obj_try - best_obj
            if rng.random() < np.exp(-d / max(1e-12, T)):
                accept = True

        if accept:
            p_best = p_try
            best_obj, best_v1, best_v2 = obj_try, v1_try, v2_try

        if (k % print_every) == 0 or k == 1:
            print(f"[{k:5d}/{n_iters}] best_obj={best_obj:.6e} var1={best_v1:.6e} var2={best_v2:.6e}")

    # 7) Build optimized expression back in SymPy
    back_map = {params[i]: sp.Float(p_best[i]) for i in range(len(params))}
    f_optimized_expr = sp.simplify(f_param.subs(back_map))

    # 8) Final full-grid reporting
    final_sq1, final_mse1 = sh_metrics(sh_num, p_best, r1_full, th1_full)
    final_sq2, final_mse2 = sh_metrics(sh_num, p_best, r2_full, th2_full)
    final_var1 = variance_of_f(f_num, p_best, r1_full, th1_full)
    final_var2 = variance_of_f(f_num, p_best, r2_full, th2_full)

    final_residual = {
        "grid1_rmax10_sqnorm": final_sq1,
        "grid1_rmax10_mse": final_mse1,
        "grid2_rmax100_sqnorm": final_sq2,
        "grid2_rmax100_mse": final_mse2,
        "combined_mse": (final_mse1 + final_mse2) / 2.0,
        "var_grid1": final_var1,
        "var_grid2": final_var2,
    }

    return {
        "atoms_optimized": atoms,
        "params": params,
        "p0": p0,
        "p_best": p_best,
        "initial_residual": initial_residual,
        "final_residual": final_residual,
        "f_optimized_expr": f_optimized_expr,
    }

# Define the polar coordinates
SH = symbols('\\text{SwiftHohenberg} r theta mu nu')
r, theta = symbols('r theta', real = True, positive = True)
mu, nu = 1, 1
# Define the function f as a function of r and theta
GENERIC = False
PERIODIC_IN_THETA = True
COMPUTE_NUMERIC = False
PRINT_SH = False
f = None
f_per_idx = 8

if GENERIC:
    f = Function('f')(r, theta)
else:
    f =  [sin(r)*sin(theta), \
    
          sin(r)*sin(theta)+0.604, \
          
          0.998846776839887*0.999950000416665**(r**4)*sin(r)*sin(theta) + 0.604, \
          
              0.88898139159952*0.999884875453817**(r**4.03)*sqrt(1 - cos(r)**2)*sin(theta) + 0.760176150613572, \
              
          -0.28580222883408**(r + 10)*(1.01 - sin(theta))*(167.620651926117*r**7.38905609893065 + 0.000105912014609458) + 0.833098208613807*0.999884875453817**(r**(17/4))*sqrt(1 - cos(r)**2)*sin(theta)  + 0.797073913381706, \
          
          -182.159206127457*0.28580222883408**(r + 10)*(1.02 - sin(theta))*(r + 0.00999991666708333)**7.50905609893065 + 0.745258709383936*0.999884875453817**(r**4.25)*sqrt(1 - cos(r)**2)*sin(theta + 6.28319)  + 0.815307524508096, \
          
          -0.28580222883408**(r + 10.02)*(2*r + 0.0137395477321287)**(0.01**(6.28319/(r + 0.01)) + 7.57016955826421)*(0.01**r - sin(theta) + 1) + 0.708762837941528*0.999884875453817**(1.3213487088109*r**4*(1.6*(tanh(.6*r))))*sqrt(1 - cos(r)**2)*sin(0.999999999988989*theta) + 0.845330825627302, \
          
          -0.285806921654494**(r + 10.0100907998593)*(r**1.00009081398177 + r**((r + 1)**0.0100001666741671))**(0.015**(6.28319/(r + 2)) + 7.59399521303526)*(- sin(theta + cos(theta) + 6.28319) + sin(log(r)))
           + 0.612417858855597*0.999884875453817**(1.4426686039141*(r + 0.01)**4.03*(1.58*(tanh(.59*r))))*sqrt(1 - cos(r)**2)*sin(theta) + 0.879032771381193, \
           
           -0.285806921654494**(r + 10.0139164646307)*(r + (r**0.999993025405072 - 5.00008333556817e-5)**(r**0.01))**(0.0166848951652189**(6.23978883640503/(r + 2)) + 0.000631778468553939*r + 7.59291602260893)*(-sin(theta + cos(theta) + 1/r) + sin(log(r)))
#            + 10**(-10**(theta + 10) + r)
#            - ((r + 0.0100001666741671*sin(theta) + 0.34334238565416)**0.980198015679673/(0.01**r*(11*r + 30) + r + 1.67000612012792 + 10**(-r)))**(((0.0100001666741671*0.01**theta + 9.99)/(r + 0.240154824221454) + sin(theta + cos(theta + 0.01) + 0.05))*(r + sin(r) + sech(sin(theta))**(r*theta + r) + 0.35862981013366))
#             + (2.71406347200553e-13 + 9.86961177255443e-20/r)*(2*r + 3*theta + exp(r) + 10.6223826114788)**(cos(theta - 0.01) + sech(theta))
#             + (2.71406357200553e-13 + 6.12323399573677e-17/(2.00090909090909 - 6.29319469282041*r + 1e-6))*(r + cos(sin(theta) - 38.41228718056) + 0.455196050124308)**(sin(sqrt(r)) + 11.259320763461)
             + 0.606923362578475*sqrt(1 - cos(r)**2)*(sech(r + 10) + 0.999884875453817)**((r + 0.02)**4.03*(1.58*(tanh(.59*r)))/(sin(sech(r)) + 0.693147180559945))*sin(theta + 6.28319)
#             - (-2*r - 8.60517018598809)*(-6.28319**r + theta + 20.29319)/(-r**2 + 54.5981500331442*(r + 6.28319)**(theta + 1)*exp(r) + 364525919796.747)
#             - (-log(r) + 1.01005016708417*sin(r) + tanh(sin(r)) + 17.9095021851957)**(0.714028539196511*r - 9.36056423787)
             + 0.886342906953379
#             - 16.047123831843/(r + theta - 10405.8182267205)
             , \
             
            -0.285806921654494**(r + 10.0101815997187)*(r**1.00009081398177 + r**((r + 1)**0.01))**(0.0150907998593378**(6.28319/(r + 2)) + 7.59399990585568)*(-sin(theta + cos(theta) + 0.01) + sin(log(r)) + sech(r)**(r/10))
            + 0.612309082637831*0.999884875453817**(1.4426686039141*(r + 0.01)**4*asin(tanh(r)))*sqrt(1 - cos(r)**2)*sin(theta - 4.69282041378069e-6)
            - 5.48657829636844e-12*theta*(-r + theta + 2)
            + 0.000194759452736256*theta
            - ((r + 0.333333333333333)**0.98019801980198/(0.01**r*(r + 30) + r + 1.61454687435024))**((sin(theta + cos(theta) + 0.01) + 10/(r + 0.179154824221454))*(r + sin(r) + 0.30789579291938)) - (-9.56638*theta - 28.626681002505)/(theta + (0.01/tanh(r))**(r**1.56079616012073) - sin(theta) - 1920153.09449961)
            + 2.71406357200553e-13*(theta + 8.29319)**(cos(theta) + sech(theta))
            + 2.71406347200553e-13*(r + cos(sin(theta) - 0.693147180559945) + 0.355680581247829)**(sin(sqrt(r)) + 11.1765281936143)
            + 0.879248603142091
            - (0.01*theta + 10.7615941559558)/(1.90929742682568*r - 10426.8045639011)][f_per_idx] \
            if PERIODIC_IN_THETA else \
            (((0.148475282221305 * theta) - (sin(theta) * (1.0000132758892615 * sin(r)))) - 0.0922858190550785)


f_thresh = (0.13014006102898107, 0.014037022014875142)
optimize = False

if optimize:
    info = naive_random_optimize(f, r, theta, f_thresh=f_thresh, N=1000, mu=mu, nu=nu, opt_stride=1, max_params=12)

    #Access stored initial residuals:
    initial_residual = info["initial_residual"]
    print("Initial residuals:", initial_residual)

    #See optimized expression:
    print("Optimized f:", info["f_optimized_expr"])

    #See final metrics:
    print("Final residuals:", info["final_residual"])
    exit()

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
    terms = sp.Add.make_args(f)  # f is your full expression
    term_funcs = [sp.lambdify((r, theta), t, "numpy") for t in terms]

    bad = []
    for k, tf in enumerate(term_funcs):
        v = tf(r_vals, theta_vals)
        imag = np.max(np.abs(np.imag(v))) if np.iscomplexobj(v) else 0.0
        n_nan = np.isnan(v).sum()
        n_inf = np.isinf(v).sum()
        if imag > 1e-12 or n_nan or n_inf:
            bad.append((k, imag, n_nan, n_inf))
    print(*bad, " ... total bad:", len(bad), sep='\n')
    
    
    f_SR = lambdify((r, theta), f)
    f_SR_r = lambdify((r, theta), f_r := diff(f, r))
    f_SR_theta = lambdify((r, theta), f_theta := diff(f, theta))

    print(f"Variance of f = {np.var(f_SR_vals:=f_SR(r_vals, theta_vals))}")
    print(f"||f|| = {LA.norm(f_SR_vals)}")
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
N = 1000
Nr = N
Nth = N
th_vec = np.linspace(0.0, 2.0*np.pi, Nth, endpoint=False)  # periodic, no duplicate endpoint
r_edges = np.linspace(0.0, r_vals.max(), Nr + 1)                   # edges include r=0
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
print(f"Initial mean-squared residual = {float(np.dot(R0, R0))/R0.size:.3f}")
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
