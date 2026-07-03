import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import sympy as sp
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from math import pi
from numpy import linalg as LA
from pathlib import Path
from os import system
from matplotlib.backends.backend_pdf import PdfPages

def sech_stable(x):
    ax = np.abs(x)
    out = np.empty_like(ax, dtype=np.float64)
    big = ax > 20
    out[big] = 2.0*np.exp(-ax[big])
    out[~big] = 1.0/np.cosh(ax[~big])
    return np.maximum(out, np.finfo(np.float64).tiny)


class SympyDagEvaluator:
    """
    Hash-consed DAG evaluator for SymPy expressions.
    Evaluates only nodes reachable from the root in postorder.
    """

    def __init__(self, expr, use_cse=False):
        self.original_expr = expr
        self.nodes = []
        self.root = -1
        self._intern_expr = {}
        self._intern_node = {}

        if use_cse:
            replacements, reduced = sp.cse(expr, optimizations='basic')
            if len(reduced) != 1:
                raise RuntimeError("Expected one reduced expression from cse.")
            expr2 = reduced[0]
            for sym, rhs in replacements[::-1]:
                expr2 = expr2.xreplace({sym: rhs})
            self.expr = expr2
        else:
            self.expr = expr

        self.root = self._build(self.expr)
        self.postorder = []
        self._make_postorder()

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

    def _make_postorder(self):
        seen = set()
        out = []

        def dfs(i):
            if i in seen:
                return
            seen.add(i)
            op, data, children = self.nodes[i]
            for c in children:
                dfs(c)
            out.append(i)

        dfs(self.root)
        self.postorder = out

    def evaluate(self, env):
        values = {}

        for i in self.postorder:
            op, data, children = self.nodes[i]

            if op == "symbol":
                if data not in env:
                    raise KeyError("Missing value for symbol '%s'" % data)
                values[i] = env[data]

            elif op == "const":
                values[i] = float(data)

            else:
                ch = [values[c] for c in children]

                if op == "add":
                    out = ch[0]
                    for x in ch[1:]:
                        out = out + x

                elif op == "mul":
                    out = ch[0]
                    for x in ch[1:]:
                        out = out * x

                elif op == "pow":
                    base, expo = ch
                    out = np.power(base, expo)

                elif op == "sin":
                    out = np.sin(ch[0])

                elif op == "cos":
                    out = np.cos(ch[0])

                elif op == "exp":
                    out = np.exp(ch[0])

                elif op == "log":
                    out = np.log(ch[0])

                elif op == "sqrt":
                    out = np.sqrt(ch[0])

                elif op == "tanh":
                    out = np.tanh(ch[0])

                elif op == "sech":
                    out = sech_stable(ch[0])

                elif op == "asin":
                    out = np.arcsin(ch[0])

                elif op == "acos":
                    out = np.arccos(ch[0])

                elif op == "abs":
                    out = np.abs(ch[0])

                else:
                    raise RuntimeError("Unknown op '%s'" % op)

                values[i] = out

        return values[self.root]

show = False
np.sech = lambda x: 1.0/np.cosh(x)

PERIODIC_IN_THETA = True
PRINT_LATEX_ONLY = False
PlotType = "2D"
N = 1000
r_max = 10
r_edges = np.linspace(0.01, r_max, N)
theta_edges = np.linspace(0, 2*np.pi, N, endpoint=False)

round_floats = lambda expr, ndigits: expr.xreplace({f: Float(round(float(f), ndigits)) for f in expr.atoms(Float)})
f_per_idx = 10
mu_equals_nu = False
mu, nu = (1, 1) if mu_equals_nu else symbols('mu nu', real = True)
f_eqn, r, theta = symbols('f r theta')
f = None
if mu_equals_nu:
    f =  [sin(r)*sin(theta), \
          sin(r)*sin(theta)+0.604, \
          0.998846776839887*0.999950000416665**(r**4)*sin(r)*sin(theta) + 0.604, \
              0.88898139159952*0.999884875453817**(r**4.03)*sqrt(1 - cos(r)**2)*sin(theta) + 0.760176150613572, \
          -0.28580222883408**(r + 10)*(1.01 - sin(theta))*(167.620651926117*r**7.38905609893065 + 0.000105912014609458) + 0.833098208613807*0.999884875453817**(r**(17/4))*sqrt(1 - cos(r)**2)*sin(theta)  + 0.797073913381706, \
          -182.159206127457*0.28580222883408**(r + 10)*(1.02 - sin(theta))*(r + 0.00999991666708333)**7.50905609893065 + 0.745258709383936*0.999884875453817**(r**4.25)*sqrt(1 - cos(r)**2)*sin(theta + 6.28319)  + 0.815307524508096, \
          -0.28580222883408**(r + 10.02)*(2*r + 0.0137395477321287)**(0.01**(6.28319/(r + 0.01)) + 7.57016955826421)*(0.01**r - sin(theta) + 1) + 0.708762837941528*0.999884875453817**(1.3213487088109*r**4*(1.6*(tanh(.6*r))))*sqrt(1 - cos(r)**2)*sin(0.999999999988989*theta) + 0.845330825627302, \
            -0.285806921654494**(r + 10.0100907998593)*(r**1.00009081398177 + r**((r + 1)**0.0100001666741671))**(0.015**(6.28319/(r + 2)) + 7.59399521303526)*(- sin(theta + cos(theta) + 6.28319) + sin(log(r))) + 0.612417858855597*0.999884875453817**(1.4426686039141*(r + 0.01)**4*(1.6*(tanh(.6*r))))*sqrt(1 - cos(r)**2)*sin(theta) + 0.879032771381193, \
           -0.285806921654494**(r + 10.0139164646307)*(r + (r**0.999993025405072 - 5.00008333556817e-5)**(r**0.01))**(0.0166848951652189**(6.23978883640503/(r + 2)) + 0.000631778468553939*r + 7.59291602260893)*(-sin(theta + cos(theta) + 1/r) + sin(log(r))) + 0.606923362578475*sqrt(1 - cos(r)**2)*(sech(r + 10) + 0.999884875453817)**((r + 0.02)**4.03*(1.58*(tanh(.59*r)))/(sin(sech(r)) + 0.693147180559945))*sin(theta + 6.28319) + 0.886342906953379, \
            -0.285806921654494**(r + sech(r) + 10.0540737282843)*(r + (r**0.999993025405072 - 0.00373485491171487)**(r**0.01) + 0.00880895839265857)**(0.0166848851652189**(6.52587306756706/(r - 0.00781876960101768)) + 0.000702853356876745*r + 7.57798844600458)*(-sin(theta + cos(theta) + 1.25218047866732/r) + sin(log(r)) + sech(r)**(0.0611608608465381*r)) + 0.604143623800264*sqrt(1 - cos(r)**2)*(sech(r + cos(r) + 8.39984108535765) + 0.999884875453817)**((r + 0.0308839840501129)**4.03*(1.58*(tanh(.59*r)))/(tanh(sech(r)) + 0.657323425267437))*sin(theta + 6.32621184985741) + 0.890778403056392, \
            6.41032241779392/(1.28262426683707*r - 21890.2917193253) - ((1e-10 + (r + 0.0675028199851666*sin(theta) + 0.315589358780667)**(sqrt(r)*(r + 2.87892339678315)*(5836.48192235195 - r)/((5836.48192235195 - r)**2 + 1e10) + 0.980141037771426)/(0.013519701745416**r*(43.6481950729572*r + 8.49265186112673) + r + 0.02*sin(r) + 1.82648253593279))**(((9.23274577772655)/(r + 0.55183450665909) + sin(theta + cos(theta + 0.520501469061028) + 5.88393395176072)*(.5*(1-tanh(1.025*(r-21.2)))))*(r + sin(r - 0.01) + (1.0e-10 + cos(sin(theta)))**(r - 1.58836944951552) + 0.37384427398835 + tanh(r)/(r + 7.97719844610347))))*((.5*(1-tanh(2e3*(r-10.01))))) + sqrt(1 - cos(r)**2)*(1.0e-10*0.68688067225485**(8.47949129673737*r) + 0.854229974212735)*(sech(r + cos(r) + 8.39614384384391) + 0.999884853180843)**(1.58799646315658*(r + 0.0308839840501129)**4.01549520152667*(1.57*(tanh(.62*r))))*(0.0100048594945809**(2*r + 5.29438341416157) + 0.7011748940086 - 47.9851001514431/(21870.6640272238))*sin(theta + 18.8962439891879) + (223.780532729645)/(sech(1.24592745883222/r) + 22066.7916781747) - ((1.5707963267949)**(-18.3890761348268) + 0.285811651486423)**(r + sech(r + 0.058634538594504) + 10.0547134992516)*(r + (r**0.999950000416665 - 0.00364405505237706)**((0.367957484641359**r + r)**0.00999966667999946) + 0.0106243230277353)**(-r**2*exp(-r)/(1 + 1348.60891792031*exp(-r)) + (0.000497198311398096*r + 0.0160184860388267)**((sin(r) + 6.78974430415452)/(r - 0.00808244126136504)) + 7.58897670822754)*(-sin(theta + cos(theta - 0.0144023112886078) + 0.104063940713704) + sin(log(r + 0.390458429297535)) + ((-tanh(0.62*r)+1.01)*(pi/2))**(0.061275433230159*r + 0.00061275433230159)) - (tanh(2*r)**(27.030153904932*r**6.29794519183292) + 101.657005682759)/(r + (0.0053984397980632*r + 5.3984397980632e-5)*log(tanh(r)) - 10452.4889454854)*((.5*(1-tanh(.1*(r-100.01))))) + (0.00273233753019377**(6.1870938127573 - sin(theta + 0.0821478246163881)) + 2.79499001433555e-13 + (6.12323399573677e-17)/(2.19282503136287 - 6.28215572322383*r))*(0.999329299739067*r + 0.455361131407934)**(sin(sqrt(r + 9.07998593378172e-5)) + 11.4146131594409 + 0.00010001/(1.01005016708417 - cos(theta)))*((.5*(1-tanh(2e3*(r-10.01))))) + 0.0101001582000134*tanh(10.1589286596707*r + 11.9949097863706) + 0.861955660617334][f_per_idx] \
            if PERIODIC_IN_THETA else \
            (((0.148475282221305 * theta) - (sin(theta) * (1.0000132758892615 * sin(r)))) - 0.0922858190550785)
else:
    f = [\
     -1.02043668920688*mu
     + 0.976265145088285*nu
     + 8.59989931879868*r
     - 0.0203834157158624*theta*(mu - 0.37792964035845)*(nu + cos(mu) + 20.8711034252206)
     + 1.02373485491171*theta
     - (6.66863975473491 - 1.12736995112076*theta)*(10.1684549022942 - 0.01*mu**2)
     + (-mu - 10.8078598550297)*sech(mu)
     + (0.01*mu + 1.70574681933198)*(theta + 0.419460833028891)*(cos(sqrt(mu)) + 1.48262246565637)
     - (mu + 0.01)*(0.01*theta + 1.07153419386493)*sin(mu)
     - (mu + 0.0126443120008097)*(r - 6.53552077810625)*sin(0.01*mu)
     - 0.130002166764172*(-nu + r)*tanh(mu)
     + (2.58445747306298e-5*theta + 3.52449144346226e-5)*(sin(cos(theta)) + 0.292574181673163*cos(theta))
     - (theta + 13.3170149415013)*tanh(r - 8.08011643682698)
     + 0.01*((0.0001*mu + 0.00147062016648684)*(-r + 2*theta + 6.83807539219667) - 0.0310159232024146)*(-0.0918464964600025*mu + sin(theta) + 2.42649277694376)*cos(theta + 0.441471131955851)
     + ((nu - 3.38720803159771)*asin(0.01*mu) + tanh(r + 8.28319))*(0.02*mu**2 + acos(sin(theta)) + 2*sech(mu) + 2.04196555729524)
     + (0.0001*tanh(mu) + 0.229736644002127)*(mu - 1.08049890084678*r + 2.84816074457674*theta + 6.35987497675188)
     - (-0.0566277722928855*tanh(2*r - 8.10519982851059) - 0.0287447493517572)*(-0.01*mu*r*(mu + 42.2909621522195)*tanh(theta) - (2.74539306159939 - sin(0.01*mu))*(11.9705677807914*theta + 54.9484044192348) + 14.44606544794*(2*mu + theta - 9.4470135686607)*sin(mu))
     - (-0.476296284520996*mu + sech(mu) + 18.4917355215373)*(0.0101001666741671*nu - 0.0102127550390257*r + 1.17863649307223)*(0.02*nu + theta + cos(0.0181619186960565*mu**2 + 0.00540115517597472*nu - 10.0382467432885) - 1.51242673800041)
     + (8.91241641691062e-6*mu*(1047.19894466968 - 103.107145789561*mu) + 0.000249300182601612*tanh(theta + 0.0381866192361131) + 0.353063588652603)*(-1.33622595207659*r - 2.4784926602329*theta - sech(mu) - 17.0617608724563)
     + (1.00003333511123e-6*theta - (0.0101847738826694 - 1.66305743820599e-6*cos(theta))*(0.0201003350150759*mu + r + 0.0766142017330207) - 0.000100003333511123*tanh(mu) + 4.99994167653742e-7*sech(tanh(theta)) + 0.00718179435113738)*(-mu + 0.975405044737502*nu - 3*theta - (0.0001*sin(theta) + 0.00999966667999946*cos(mu) + 0.998552911879501)*(0.01*nu + r + tanh(theta - 9.41797714882564) - 27.9497372299844) + tanh(theta) + acos(sech(mu)) + 94.5127309793458)
     + ((9.99950001972143e-5*r - 1.99990000394429e-6)*(6.28287585358945*theta + 0.394765027345147)*(-0.01*nu - 0.0200909665335049*theta + 5.95284780272031) + 0.00168753213792146*sin(r) + 0.0168753213792146*sech(mu) + 0.07*sech(mu - 0.399619592949299) + 0.629496940343797)*(4.63734251914198*mu + 0.99*r + 8.33220171459921*theta + sin(mu) + cos(mu) + sech((0.996469636845584 - mu)*(1.03596136665519 - mu)) + 73.7463877563006)
     + 2*sin(mu)
     + sin(mu + 1.44434977098051)
     + sin(mu + 7.68950568510422)
     + 3.46417227282007*cos(0.00999983333416666*r*(2*r + theta + 0.01))
     - tanh(0.02*theta)
     - 0.000907998593378171*tanh(theta)
     + acos(0.0947140398686842*r)
     + 2*acos(sin(theta))
     - sech(-0.204245759737047*r + 1.72063673930557*theta + (0.0100909665335049 - tanh(r))*(-0.998687620305263*theta - tanh(theta) + 1.02370492421771) - (0.0154873527439205*r + 0.0998966701331946)*(mu*r + 5*r + 9.5115459200379) + 8.61038625622014)
     - 84.0357720038456,
     (0.000214601556558127*mu + 1.34828171950016e-6*nu + 0.000207597442231019*sin(mu) - 1.00526293689344)*(0.00877926734854302*mu - 0.16515718756188*nu - 0.000653938060863869*sin(mu)**2 + 3.66064988347847)*sin(r - 1.70585404191244e-10)*sin(5.30179718552688*nu - theta + sin(10.6889050130482*mu + 7.33505942049553e-8) + 72.4312091689447),
     -(0.0100000249223089*nu - 0.852354395503058)*(-0.165057185695183*nu - 0.165057185695183*sin(mu) + 4.0792426600352)*sin(r)*sin(mu*nu + 0.0503322423527055*nu + theta + 137.263108969405),
      -1.00526293689344*(-0.00982457832153368*mu*(mu*sin(nu) + 2*mu + 3.65979113490899)*sin(mu) + 2.25358925601765)*sin(r - 1.70585404191244e-10)*sin(0.000207410181763234*mu*nu + nu**2*(mu - 5.93918060635909e-7) - nu + theta + sin(mu*nu + 10.6889050130482)),
     ][2]
'''
best result for mu_equals_nu == False
=====================================
f = \left(0.86322 - 0.01 \cdot \nu\right) \cdot \left(- 0.01732 \cdot \mu - 0.18238 \cdot \nu + 4.23664\right) \cdot \sin{\left(r \right)} \cdot \sin{\left(\mu \cdot \nu + 0.05033 \cdot \nu + \theta + 130.97992 \right)}
SH DAG nodes = 88
f DAG nodes = 22
mse = 218.6430061206957; Added mu=0.01, nu=0.01
mse = 213.19952195116238; Added mu=0.01, nu=5
mse = 156.24335118293612; Added mu=0.01, nu=10
mse = 51.81448226072274; Added mu=5, nu=0.01
mse = 168.06877497506633; Added mu=5, nu=5
mse = 140.55527229864572; Added mu=5, nu=10
mse = 62.9062082915746; Added mu=10, nu=0.01
mse = 211.81758453457297; Added mu=10, nu=5
mse = 162.94956407482636; Added mu=10, nu=10
Average mse = 154.02197396557813
'''
formula_label = latex(f_float_rounded:=round_floats(f, 5), mul_symbol='dot')
print(f'f = {formula_label}')

if PRINT_LATEX_ONLY:
    print("MULTILINE f")
    print("===========")
    print(sp.multiline_latex(f_eqn, (sp.expand(f_float_rounded)), 1))
    exit()
# Calculate the first Laplacian (Laplacian of f)
laplacian_f = diff(f, r, 2) + (1/r) * diff(f, r) + (1/(r**2)) * diff(f, theta, 2)

# Calculate the double Laplacian (Laplacian of the first Laplacian)
double_laplacian_f = diff(laplacian_f, r, 2) + (1/r) * diff(laplacian_f, r) + (1/(r**2)) * diff(laplacian_f, theta, 2)

swift_hohenberg = mu*f + nu*f*f - f*f*f - (f + 2*laplacian_f + double_laplacian_f)

sh_eval = SympyDagEvaluator(swift_hohenberg, use_cse=False)

if mu_equals_nu:
    # Preserve old behavior exactly
    r_vals, theta_vals = np.meshgrid(
        np.linspace(0.01, r_max, N),
        np.linspace(0, 2*pi, N)
    )

    func_vals = sh_eval.evaluate({
        "r": r_vals,
        "theta": theta_vals
    })

    squared_norm_error = LA.norm(func_vals.ravel())**2
    mean_squared_error = squared_norm_error / func_vals.size

else:
    # Do not build a 4D mesh. Use scalar mu, nu per plot.
    mu_plot_vals = np.linspace(0.01, 10, 3)
    nu_plot_vals = np.linspace(0.01, 10, 3)

    # Reasonable MSE/plot mesh for the parameter sweep
    N_sweep = 1000
    r_vals, theta_vals = np.meshgrid(
        np.linspace(0.01, r_max, N_sweep),
        np.linspace(0, 2*pi, N_sweep)
    )

    mean_squared_error = None

print("SH DAG nodes =", len(sh_eval.nodes))

# Compute centers
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
R, Theta = np.meshgrid(r_centers, theta_centers)

# Evaluate f on cell centers using the same DAG idea
# Evaluate f on cell centers using the same DAG idea
f_eval = SympyDagEvaluator(f, use_cse=False)

print("f DAG nodes =", len(f_eval.nodes))

# Convert to Cartesian
X = R * np.cos(Theta)
Y = R * np.sin(Theta)

formula_label = r"$f(r,\theta) = " + formula_label + "$"

def plot_one(mu0=None, nu0=None):
    global mean_squared_error
    mse = 0
    if mu_equals_nu:
        env_plot = {"r": R, "theta": Theta}
        mse = mean_squared_error
    else:
        env_plot = {"r": R, "theta": Theta, "mu": mu0, "nu": nu0}

        env_mse = {
            "r": r_vals,
            "theta": theta_vals,
            "mu": mu0,
            "nu": nu0,
        }

        func_vals = sh_eval.evaluate(env_mse)
        squared_norm_error = LA.norm(np.nan_to_num(func_vals).ravel())**2
        mse = squared_norm_error / func_vals.size
        print(f"mse = {mse}", end = "; ")

    Z = f_eval.evaluate(env_plot)

    if PlotType == "3D":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(r"$f(r,\theta)$")

        if mu_equals_nu:
            ax.set_title(f"Swift-Hohenberg 2D Pattern, MSE = {mse:.3f}")
        else:
            ax.set_title(rf"$\mu={mu0:.3g}$, $\nu={nu0:.3g}$, MSE = {mse:.3e}")

        fig.colorbar(surf, shrink=0.5, aspect=10, label=formula_label)
        ax.view_init(elev=35, azim=235)

    else:
        fig, ax = plt.subplots()

        vmin, vmax = -1, 1
        min_, max_ = vmin, vmax
        levels = np.linspace(vmin, vmax, 100)

        Xc = np.vstack([X, X[0:1, :]])
        Yc = np.vstack([Y, Y[0:1, :]])
        Zc = np.vstack([Z, Z[0:1, :]])

        contour = ax.contourf(
            Xc, Yc, Zc,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            extend="both"
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if mu_equals_nu:
            ax.set_title(f"Swift-Hohenberg 2D Pattern, MSE = {mse:.3f}")
        else:
            ax.set_title(rf"$\mu={mu0:.3g}$, $\nu={nu0:.3g}$, MSE = {mse:.3e}")

        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label(formula_label, fontsize=7)
        cbar.set_ticks([min_, 0, max_])

        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    return fig, mse


Path("SwiftHohenbergPlots").mkdir(exist_ok=True)

domain_str = f"_r_{int(r_edges[0])}_{int(r_edges[-1])}_theta_{int(theta_edges[0])}_{int(theta_edges[-1])}_N_{N}"

if mu_equals_nu:
    fig, mse = plot_one()

    if show:
        plt.show()
    else:
        filename = f"SwiftHohenbergPlots/SwiftHohenberg2D{'Periodic'+str(f_per_idx)+domain_str if PERIODIC_IN_THETA else 'NonPeriodic'}.pdf"
        fig.savefig(filename)
        print(f"Saved {filename}")

        filename_no_ext = filename[:-4]
        system(f"open {filename_no_ext}.pdf")
        system(f"sips -s format png -s dpiWidth 480 -s dpiHeight 480 -z 2400 2400 {filename_no_ext}.pdf --out {filename_no_ext}.png")

else:
    filename = f"SwiftHohenbergPlots/SwiftHohenberg2D_mu_nu_sweep.pdf"
    total_mse = 0
    count = 0
    with PdfPages(filename) as pdf:
        for mu0 in mu_plot_vals:
            for nu0 in nu_plot_vals:
                fig, mse = plot_one(float(mu0), float(nu0))
                count += 1
                total_mse += mse
                pdf.savefig(fig)
                plt.close(fig)
                print(f"Added mu={mu0:.3g}, nu={nu0:.3g}")

    print(f"Average mse = {total_mse/count}")
    print(f"Saved {filename}")
    system(f"open {filename}")
