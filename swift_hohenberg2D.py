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

def phi(z):
    return Piecewise((0, z <= 0), (exp(-1/z), True))

def smooth_step(z):
    return phi(z)/(phi(z) + phi(1 - z))

def cutoff(r, r0, r1):
    return 1 - smooth_step((r - r0)/(r1 - r0))

def sech_stable(x):
    ax = np.abs(x)
    out = np.empty_like(ax, dtype=np.float64)
    # for large |x|, sech(x) ≈ 2*exp(-|x|) (never overflows)
    big = ax > 20
    out[big] = 2.0*np.exp(-ax[big])
    out[~big] = 1.0/np.cosh(ax[~big])
    return np.maximum(out, np.finfo(np.float64).tiny)  # <-- key line
    
class SympyDagEvaluator:
    """
    Build a DAG from a SymPy Expr by hash-consing subexpressions,
    then evaluate bottom-up on NumPy arrays/scalars.

    Node format:
        (op, data, children)

    where:
        op       : string like "const", "symbol", "add", "mul", ...
        data     : constant value or symbol name or None
        children : tuple of child node ids
    """

    def __init__(self, expr):
        self.expr = expr
        self.nodes = []
        self.root = -1

        # SymPy expr -> node id
        self._intern_expr = {}

        # (op, data, children) -> node id
        self._intern_node = {}

        self.root = self._build(expr)

    def _intern(self, op, data, children):
        key = (op, data, children)
        if key in self._intern_node:
            return self._intern_node[key]
        idx = len(self.nodes)
        self.nodes.append((op, data, children))
        self._intern_node[key] = idx
        return idx

    def _build(self, expr):
        if expr in self._intern_expr:
            return self._intern_expr[expr]

        if expr.is_Symbol:
            node_id = self._intern("symbol", str(expr), ())

        elif expr.is_Integer:
            node_id = self._intern("const", float(int(expr)), ())

        elif expr.is_Rational:
            node_id = self._intern("const", float(expr), ())

        elif expr.is_Float:
            node_id = self._intern("const", float(expr), ())

        elif expr.is_Number:
            node_id = self._intern("const", float(expr.evalf()), ())

        else:
            args = tuple(self._build(a) for a in expr.args)

            if expr.func is sp.Add:
                node_id = self._intern("add", None, args)

            elif expr.func is sp.Mul:
                node_id = self._intern("mul", None, args)

            elif expr.func is sp.Pow:
                node_id = self._intern("pow", None, args)

            elif expr.func is sp.sin:
                node_id = self._intern("sin", None, args)

            elif expr.func is sp.cos:
                node_id = self._intern("cos", None, args)

            elif expr.func is sp.exp:
                node_id = self._intern("exp", None, args)

            elif expr.func is sp.log:
                node_id = self._intern("log", None, args)

            elif expr.func is sp.sqrt:
                node_id = self._intern("sqrt", None, args)

            elif expr.func is sp.tanh:
                node_id = self._intern("tanh", None, args)

            elif expr.func.__name__ == "sech":
                node_id = self._intern("sech", None, args)

            elif expr.func is sp.asin:
                node_id = self._intern("asin", None, args)

            elif expr.func is sp.acos:
                node_id = self._intern("acos", None, args)

            elif expr.func is sp.Abs:
                node_id = self._intern("abs", None, args)

            else:
                raise NotImplementedError(
                    "Unsupported SymPy node: func=%r expr=%r" % (expr.func, expr)
                )

        self._intern_expr[expr] = node_id
        return node_id

    def evaluate(self, env, shape=None, dtype=np.float64):
        """
        env maps symbol names to numpy arrays/scalars, e.g.
            {"r": r_vals, "theta": theta_vals}

        shape is only needed if expression is constant and you want an array result.
        """
        values = [None] * len(self.nodes)

        for i, node in enumerate(self.nodes):
            op, data, children = node

            if op == "symbol":
                if data not in env:
                    raise KeyError("Missing value for symbol '%s'" % data)
                values[i] = env[data]

            elif op == "const":
                c = np.array(data, dtype=dtype)
                if shape is None:
                    values[i] = c
                else:
                    values[i] = np.full(shape, c, dtype=dtype)

            else:
                ch = [values[j] for j in children]

                if op == "add":
                    out = ch[0]
                    for x in ch[1:]:
                        out = out + x
                    values[i] = out

                elif op == "mul":
                    out = ch[0]
                    for x in ch[1:]:
                        out = out * x
                    values[i] = out

                elif op == "pow":
                    base, expo = ch
                    values[i] = np.power(base, expo)

                elif op == "sin":
                    values[i] = np.sin(ch[0])

                elif op == "cos":
                    values[i] = np.cos(ch[0])

                elif op == "exp":
                    values[i] = np.exp(ch[0])

                elif op == "log":
                    values[i] = np.log(ch[0])

                elif op == "sqrt":
                    values[i] = np.sqrt(ch[0])

                elif op == "tanh":
                    values[i] = np.tanh(ch[0])

                elif op == "sech":
                    values[i] = sech_stable(ch[0])

                elif op == "asin":
                    values[i] = np.arcsin(ch[0])

                elif op == "acos":
                    values[i] = np.arccos(ch[0])

                elif op == "abs":
                    values[i] = np.abs(ch[0])

                else:
                    raise RuntimeError("Unknown op '%s'" % op)

        return values[self.root]
        
# Define the polar coordinates
SH = symbols('\\text{SwiftHohenberg} r theta mu nu')
r, theta = symbols('r theta', real = True, positive = True)
mu, nu = 1, 1
# Define the function f as a function of r and theta
GENERIC = False
PERIODIC_IN_THETA = True
COMPUTE_NUMERIC = False
DEBUG_NAN = False
PRINT_SH = False
f = None
f_per_idx = 10
feta = 0

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
             -0.285806921654494**(r + sech(r) + 10.0540737282843)*(r + (r**0.999993025405072 - 0.00373485491171487)**(r**0.01) + 0.00880895839265857)**(0.0166848851652189**(6.52587306756706/(r - 0.00781876960101768)) + 0.000702853356876745*r + 7.57798844600458)*(-sin(theta + cos(theta) + 1.25218047866732/r) + sin(log(r)) + sech(r)**(0.0611608608465381*r))
#              + 10**(-10**(theta + 10) + r)
#              - ((r + 0.0675028199851666*sin(theta) + 0.319527238502232)**0.980198011557366/(0.0117005728144073**r*(r + 55.7585786235571) + r + 1.81013830026433 + (r + 10.6189532795815)**(-r)))**((sin(theta + cos(theta + 0.452450872037153) + 5.99600860790768) + 9.35715943594639/(r + 0.52683483997909))*(r + sin(r) + sech(sin(theta))**(r + theta) + 0.425613409800106))
#              + (2.71406357200553e-13 + 6.12323399573677e-17/(2.00090909090909 - 7.78794109346426*r))*(r + cos(sin(theta) - 38.4131109482099) + 0.499577688583474)**(sin(sqrt(r)) + 11.3049428452759)
#              + (2.71408016417197e-13 + 3.35149368755653e-18/sqrt(r))*(2*theta + 5.19646483481694*exp(r) + tanh(theta) + 7.82150631202911)**(cos(theta - 0.0325614982719992) + sech(theta))
              + 0.604143623800264*sqrt(1 - cos(r)**2)*(sech(r + cos(r) + 8.39984108535765) + 0.999884875453817)**((r + 0.0308839840501129)**4.03*(1.58*(tanh(.59*r)))/(tanh(sech(r)) + 0.657323425267437))*sin(theta + 6.32621184985741)
#              - (-6.28613844557487**r - 167.513687684933)*(-2.64384776247393*r - 25.9576392396297)/(-r**2 - r + 13.9321568185087*(r + 8.88197588332913)**(theta + 0.978878224067382)*exp(r) + 364525919766.594)
#              - (-log(r) + 2.55087081076035*sin(r) + tanh(sin(r)) + 12.6004110087972)**(0.656205870576171*r - 9.67665833558963)
              + 0.890778403056392
#              - 50.236703696356/(r - theta - 10631.0099021024)
              , \
#              -6.97457203102658e-10*r
#              + 7.96110622688787e-5*theta
#              + 5.7581511072593*theta/(1.25423469760191*r - 21948.0619370553)
                - ((1e-10 + (r + 0.0675028199851666*sin(theta) + 0.315589358780667)**(sqrt(r)*(r + 2.87892339678315)*(5953.65096806617 - r)/((5953.65096806617 - r)**2 + 1e10) + 0.980141037771426)/(0.013519701745416**r*(43.687622183442*r + 8.33814060981745) + r + 0.02*sin(r) + 1.82648253593279))**(((9.2348889286512)/(r + 0.55183450665909) + sin(theta + cos(theta + 0.519039044087815) + 5.8847752990135)*(.5*(1-tanh(1.025*(r-21.2)))))*(r + sin(r - 0.01) + (1.0e-10 + cos(sin(theta)))**(r - 1.58074387559245) + 0.37384427398835 + tanh(r)/(r + 7.97723076614237))))*((.5*(1-tanh(1.775e3*(r-10.01)))))*(1)
                
              + sqrt(1 - cos(r)**2)*(1.0e-10*0.68688067225485**(8.16109249232708*r) + 0.854229974212735)*(sech(r + cos(r) + 8.39614384384391) + 0.999884853180843)**(1.58799646315658*(r + 0.0308839840501129)**4.01549520152667*(1.57*(tanh(.62*r))))*(0.0100048594945809**(2*r + 5.29438341416157) + 0.7011748940086 - 45.6560816728088/(21934.7382737552))*sin(theta + 18.8962439891879)*(1)
#              - (-6.31938761555448**(r + 0.01) + 5.92927345732395*theta*tanh(r))*(-3.07571474874356*r - theta - 48.7005902035467)/(8.51330104620672**r + 8.99586054074951**r + theta**2*(2*r)**(theta + 0.86387229970376)*exp(r) + exp(r) + 364526023290.415)
#              + (5.52136873685152e55*6.45127057152239**(r + 0.0565869958743546) + 5.52136873685152e55*cos(theta - 1.14278632914743))*(4.69282041378069e-6*exp(theta) + 0.478323918013976)**(-11.2069358951144 - theta/1.96953999447147**theta)
#              - (r + 0.141913233757233)**(2*theta)*(r + 4.43794472272825)*log(r)*sech(exp(10 - theta))
#              + (8.133634781186*theta + 210.8664358206)/(sech(1.23159415646033/r) + 22208.9094631345)
              - ((1.5707963267949)**(-18.3837681892581) + 0.285811651486423)**(r + sech(r + 0.0561717295263584) + 10.0547134992516)*(r + (r**0.999950000416665 - 0.00364405505237706)**((0.376065617272839**r + r)**0.00999966667999946) + 0.0106243230277353)**(-r**2*exp(-r)/(1 + 1358.42254658947*exp(-r)) + (0.000469282041378069*r + 0.0160184860388267)**((sin(r) + 6.78974430415452)/(r - 0.00781876960101768)) + 7.58897670822754)*(-sin(theta + cos(theta - 0.0144023112886078) + 0.105413950813453) + sin(log(r + 0.390458429297535)) + ((-tanh(0.62*r)+1.01)*(pi/2))**(0.061275433230159*r + 0.00061275433230159))
#              - (tanh(2*r)**(26.0630000590122*r**6.29029115986841) + 95.2404187164865)/(r + (0.0053984397980632*r + 5.3984397980632e-5)*log(tanh(r)) - 10335.0958606709)
              + (0.00273233753019377**(6.19641677671904 - sin(theta + 0.089280925720443)) + 2.79499001433555e-13 + (6.12323399573677e-17)/(2.19270786451049 - 6.28221254344588*r))*(0.999329299739067*r + 0.453212918064574)**(sin(sqrt(r + 9.07998593378172e-5)) + 11.4130415650481 + 0.00010001/(1.01005016708417 - cos(theta)))*((.5*(1-tanh(1.775e3*(r-10.01)))))
#              + (0.0198370015775754**(tanh(theta) + 7.93024265961469) + 4.85851653532341e-19*r*theta**2/(5.73576501270149 - 2*r) + 2.89036725439893e-13)*(1.87368352123689*r + 10.5926268755992*theta + (theta + 6.6244829732113)*exp(r) + 2.57539899861567)**(cos(tanh(sin(theta))) + sech(theta + 0.114076364228401))
#              - ((sin(theta) + 1.81465628877088)*sin(r + 0.0428431433987448) - log(r) + tanh(sin(r)) + 11.8460597445485)**(0.74666154308685*r - 9.55656216208119)*(0)
              + 0.0101001582000134*tanh(10.0327249667171*r + 11.9956250261793)
              + 0.863191833358681
#              https://chatgpt.com/c/69cb3325-be28-8328-a8d3-c503e0ef5021
              ][f_per_idx] \
            if PERIODIC_IN_THETA else \
            (((0.148475282221305 * theta) - (sin(theta) * (1.0000132758892615 * sin(r)))) - 0.0922858190550785)

print(f"f = {f}\n")
#print(f"sp.expand(f) = {sp.expand(f)}")
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

#{r<=10, r<=100, r<=1000, r<=10000, r<=100000} threshholds:
#2e3: {0.014644639263880545, 0.010045543165114933, 0.009417682123248308, 0.009346834502440376, 0.009337827805946159}
#1.8e3: {0.014645134264879296, 0.010322621280987975, 0.009418247925166098, 0.009347111268533347,  0.009337827805946159}
#1.7e3: {0.014654068054416325, 0.016449047261037967, ❌}
#1.75e3: {..., 0.011385481562662716, ❌}
#1.775e3: {0.014645675282139147, 0.010652997285208078, 0.009418752758094926, 0.009347510063070913, 0.009337827805946159}
#1.76e3: {0.0146460922350811, 0.011019015269055561 ❌}

# print(*swift_hohenberg.args, sep="\n")
r_vals, theta_vals = [None]*2
func_vals = None
N = 1000 #echo $?
if not GENERIC:
    r_vals, theta_vals = np.meshgrid(np.linspace(0.01, 10, N), np.linspace(0, 2*pi, N))
    terms = sp.Add.make_args(diff(f, r))  # f is your full expression
    term_funcs = [sp.lambdify((r, theta), t, modules=[{"sech": sech_stable}, "numpy"]) for t in terms]

    if DEBUG_NAN:
        bad = []
        for k, tf, term in zip(range(len(terms)), term_funcs, terms):
            v = tf(r_vals, theta_vals)
            imag = np.max(np.abs(np.imag(v))) if np.iscomplexobj(v) else 0.0
            n_nan = np.isnan(v).sum()
            n_inf = np.isinf(v).sum()
            if imag > 1e-12 or n_nan or n_inf:
                bad.append((k, imag, n_nan, n_inf, term))
        print(*bad, " ... total bad:", len(bad), sep='\n')
    
#
#    f_SR = lambdify((r, theta), f, modules=[{"sech": sech_stable}, "numpy"])
#    f_SR_r = lambdify((r, theta), f_r := diff(f, r), modules=[{"sech": sech_stable}, "numpy"])
#    f_SR_theta = lambdify((r, theta), f_theta := diff(f, theta), modules=[{"sech": sech_stable}, "numpy"])
#    print(f"f_r = {f_r}");
#    print(f"Variance of f = {np.var(f_SR_vals:=f_SR(r_vals, theta_vals))}")
#    print(f"||f|| = {LA.norm(f_SR_vals)}")
#    print(f"Max(∂f/∂r) = {np.max(f_SR_r_vals:=f_SR_r(r_vals, theta_vals))}")
#    print(f"Max(∂f/∂θ) = {np.max(f_SR_theta_vals:=f_SR_theta(r_vals, theta_vals))}")
#    print(f"Median(∂f/∂r) = {np.median(np.sort(f_SR_r_vals))}")
#    print(f"Median(∂f/∂θ) = {np.median(np.sort(f_SR_theta_vals))}")

    # Evaluate
    dag_eval = SympyDagEvaluator(swift_hohenberg)
    func_vals = dag_eval.evaluate(
        env={"r": r_vals, "theta": theta_vals},
        shape=r_vals.shape
    )

    squared_norm_error = LA.norm(func_vals.ravel())**2
    mean_squared_error = squared_norm_error / func_vals.size

    print(f"num DAG nodes = {len(dag_eval.nodes)}")
    print(f"squared-norm error = {squared_norm_error}")
    print(f"mean-squared error = {mean_squared_error}")
    exit()
    func = lambdify((r, theta), swift_hohenberg, modules=[{"sech": sech_stable}, "numpy"])
#    func_vals = func(r_vals, theta_vals)
#
##    print(f"func_vals.size = {func_vals.size}")
##    print(f"func_vals.shape = {func_vals.shape}")
##    print(f"func_vals = {func_vals}");
##    print(f"diff(func_vals, axis = 0) = {np.diff(func_vals, axis = 0)}") #diff(f, theta)
##    print(f"diff(func_vals, axis = 1) = {np.diff(func_vals, axis = 1)}") #diff(f, r)
#    squared_norm_error = LA.norm(func_vals.flatten())**2
#    print(f"squared-norm error = {squared_norm_error}")
##    print(sp.multiline_latex(SH, swift_hohenberg, 2).replace(r"\frac", r"\dfrac"))
#    mean_squared_error = squared_norm_error / func_vals.size
#    print(f"mean-squared_error = {mean_squared_error}")

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
f_func = lambdify((r, theta), f, modules=[{"sech": sech_stable}, "numpy"])
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

#So the below expression
#```
#  - (((r + 0.0675028199851666*sin(theta) + 0.315589358780667)**(sqrt(r)*(r + 2.87892339678315)/(5953.65096806617 - r) + 0.980141037771426)/(0.013519701745416**r*(43.687622183442*r + 8.33814060981745) + r + 0.02*sin(r) + 1.82648253593279))**(((9.2348889286512)/(r + 0.55183450665909) + sin(theta + cos(theta + 0.519039044087815) + 5.8847752990135))*(r + sin(r - 0.01) + cos(sin(theta))**(r - 1.58074387559245) + 0.37384427398835 + tanh(r)/(r + 7.97723076614237))))
#  + sqrt(1 - cos(r)**2)*(1.0e-10*0.68688067225485**(8.16109249232708*r) + 0.854229974212735)*(sech(r + cos(r) + 8.39614384384391) + 0.999884853180843)**(1.58799646315658*(r + 0.0308839840501129)**4.01549520152667*(1.57*(tanh(.62*r))))*(0.0100048594945809**(2*r + 5.29438341416157) + 0.7011748940086 - 45.6560816728088/(21934.7382737552))*sin(theta + 18.8962439891879)
#  - ((1.5707963267949)**(-18.3837681892581) + 0.285811651486423)**(r + sech(r + 0.0561717295263584) + 10.0547134992516)*(r + (r**0.999950000416665 - 0.00364405505237706)**((0.376065617272839**r + r)**0.00999966667999946) + 0.0106243230277353)**(-r**2/(exp(r) + 1358.42254658947) + (0.000469282041378069*r + 0.0160184860388267)**((sin(r) + 6.78974430415452)/(r - 0.00781876960101768)) + 7.58897670822754)*(-sin(theta + cos(theta - 0.0144023112886078) + 0.105413950813453) + sin(log(r + 0.390458429297535)) + ((-tanh(0.62*r)+1.01)*(pi/2))**(0.061275433230159*r + 0.00061275433230159))
#  + (0.00273233753019377**(6.19641677671904 - sin(theta + 0.089280925720443)) + 2.79499001433555e-13 + (6.12323399573677e-17)/(2.19270786451049 - 6.28221254344588*r))*(0.999329299739067*r + 0.453212918064574)**(sin(sqrt(r + 9.07998593378172e-5)) + 11.4130415650481 + 0.00010001/(1.01005016708417 - cos(theta)))
#  - ((sin(theta) + 1.81465628877088)*sin(r + 0.0428431433987448) - log(r) + tanh(sin(r)) + 11.8460597445485)**(0.74666154308685*r - 9.55656216208119)
#  + 0.0101001582000134*tanh(10.0327249667171*r + 11.9956250261793)*(1)
#  + 0.863191833358681
#```
#
#gives me an mse of 0.0170123 on the `N = 1000, np.meshgrid(np.linspace(0.01, 10, N), np.linspace(0, 2*pi, N))` mesh but inf on the `N = 1000, np.meshgrid(np.linspace(0.01, 100, N), np.linspace(0, 2*pi, N))` so some part of it doesn't decay or blows up I'm guessing since all my past prunes solves hosted here (https://docs.google.com/presentation/d/11YRvkZ2o9TBRSwIOQ4RcDrl1GS1Sbh_QVH6S8en5ngM/edit?slide=id.g3c6d958ab3b_0_28#slide=id.g3c6d958ab3b_0_28) decay out from the center, i.e. localized. so what's the MINIMAL amputation you can make to most likely save the mse?!

#So I replaced `` by cutoff(r,10,14) and it gave me the following error:
#```
#Traceback (most recent call last):
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 368, in <module>
#    dag_eval = SympyDagEvaluator(swift_hohenberg)
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 81, in __init__
#    self.root = self._build(expr)
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 112, in _build
#    args = tuple(self._build(a) for a in expr.args)
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 112, in <genexpr>
#    args = tuple(self._build(a) for a in expr.args)
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 112, in _build
#    args = tuple(self._build(a) for a in expr.args)
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 112, in <genexpr>
#    args = tuple(self._build(a) for a in expr.args)
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 112, in _build
#    args = tuple(self._build(a) for a in expr.args)
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 112, in <genexpr>
#    args = tuple(self._build(a) for a in expr.args)
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 112, in _build
#    args = tuple(self._build(a) for a in expr.args)
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 112, in <genexpr>
#    args = tuple(self._build(a) for a in expr.args)
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 112, in _build
#    args = tuple(self._build(a) for a in expr.args)
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 112, in <genexpr>
#    args = tuple(self._build(a) for a in expr.args)
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 112, in _build
#    args = tuple(self._build(a) for a in expr.args)
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 112, in <genexpr>
#    args = tuple(self._build(a) for a in expr.args)
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 112, in _build
#    args = tuple(self._build(a) for a in expr.args)
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 112, in <genexpr>
#    args = tuple(self._build(a) for a in expr.args)
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 112, in _build
#    args = tuple(self._build(a) for a in expr.args)
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 112, in <genexpr>
#    args = tuple(self._build(a) for a in expr.args)
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 112, in _build
#    args = tuple(self._build(a) for a in expr.args)
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 112, in <genexpr>
#    args = tuple(self._build(a) for a in expr.args)
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 112, in _build
#    args = tuple(self._build(a) for a in expr.args)
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 112, in <genexpr>
#    args = tuple(self._build(a) for a in expr.args)
#  File "/Users/edwardfinkelstein/alpha-zero-general/swift_hohenberg2D.py", line 154, in _build
#    raise NotImplementedError(
#NotImplementedError: Unsupported SymPy node: func=<class 'sympy.core.relational.GreaterThan'> expr=r >= 14
#```
