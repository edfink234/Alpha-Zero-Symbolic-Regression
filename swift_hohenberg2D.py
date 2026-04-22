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
r, theta = symbols('r theta', real = True, positive = True)
mu_equals_nu = True
mu, nu = (1, 1) if mu_equals_nu else symbols('mu nu', real = True)
# Define the function f as a function of r and theta
GENERIC = False
COMPUTE_NUMERIC = False
COMPUTE_INIT_NUMERIC_ONLY = False
DEBUG_NAN = True
PRINT_SH_AND_EXIT = False
PERIODIC_IN_THETA = True
f = None
f_per_idx = 3

if GENERIC:
    f = Function('f')(r, theta)
elif mu_equals_nu:
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
#              -6.97457203102658e-10*r #❌
#              + 0.000113400864086264*sin(theta) #❌
              + 6.41032241779392/(1.28262426683707*r - 21890.2917193253) #✅
              - ((1e-10 + (r + 0.0675028199851666*sin(theta) + 0.315589358780667)**(sqrt(r)*(r + 2.87892339678315)*(5836.48192235195 - r)/((5836.48192235195 - r)**2 + 1e10) + 0.980141037771426)/(0.013519701745416**r*(43.6481950729572*r + 8.49265186112673) + r + 0.02*sin(r) + 1.82648253593279))**(((9.23274577772655)/(r + 0.55183450665909) + sin(theta + cos(theta + 0.520501469061028) + 5.88393395176072)*(.5*(1-tanh(1.025*(r-21.2)))))*(r + sin(r - 0.01) + (1.0e-10 + cos(sin(theta)))**(r - 1.58836944951552) + 0.37384427398835 + tanh(r)/(r + 7.97719844610347))))*((.5*(1-tanh(2e3*(r-10.01)))))*(1)
              
              + sqrt(1 - cos(r)**2)*(1.0e-10*0.68688067225485**(8.47949129673737*r) + 0.854229974212735)*(sech(r + cos(r) + 8.39614384384391) + 0.999884853180843)**(1.58799646315658*(r + 0.0308839840501129)**4.01549520152667*(1.57*(tanh(.62*r))))*(0.0100048594945809**(2*r + 5.29438341416157) + 0.7011748940086 - 47.9851001514431/(21870.6640272238))*sin(theta + 18.8962439891879)
              
#              - (-6.31938761555448**(r + 0.01) + 4.70690542793667*theta*tanh(r))*(-3.07571474874356*r - theta - 50.0195801090766)/(8.54730467651944**r + 9.01195165841846**r + theta**2*(2*r)**(theta + 0.847534706330229)*exp(r) + exp(r) + 364526023284.345) #❌
              
#              + (5.52136873685152e-15*6.45127057152239**(r + 0.0565869958743546) + 5.52136873685152e-15*cos(theta - 1.13992602268042))*(4.69282041378069e-6*exp(theta) + 0.478323918013976)**(-11.3510427141294 - theta/2.09369070528888**theta) #❌
              
#              - (r + 0.149682587698298)**(2*theta)*(r + 5.65960416070114)*log(r)*sech(exp(10 - theta)) #❌
              
              + (223.780532729645)/(sech(1.24592745883222/r) + 22066.7916781747) #✅
              
              - ((1.5707963267949)**(-18.3890761348268) + 0.285811651486423)**(r + sech(r + 0.058634538594504) + 10.0547134992516)*(r + (r**0.999950000416665 - 0.00364405505237706)**((0.367957484641359**r + r)**0.00999966667999946) + 0.0106243230277353)**(-r**2*exp(-r)/(1 + 1348.60891792031*exp(-r)) + (0.000497198311398096*r + 0.0160184860388267)**((sin(r) + 6.78974430415452)/(r - 0.00808244126136504)) + 7.58897670822754)*(-sin(theta + cos(theta - 0.0144023112886078) + 0.104063940713704) + sin(log(r + 0.390458429297535)) + ((-tanh(0.62*r)+1.01)*(pi/2))**(0.061275433230159*r + 0.00061275433230159))
              
              - (tanh(2*r)**(27.030153904932*r**6.29794519183292) + 101.657005682759)/(r + (0.0053984397980632*r + 5.3984397980632e-5)*log(tanh(r)) - 10452.4889454854)*((.5*(1-tanh(.1*(r-100.01)))))
              
              + (0.00273233753019377**(6.1870938127573 - sin(theta + 0.0821478246163881)) + 2.79499001433555e-13 + (6.12323399573677e-17)/(2.19282503136287 - 6.28215572322383*r))*(0.999329299739067*r + 0.455361131407934)**(sin(sqrt(r + 9.07998593378172e-5)) + 11.4146131594409 + 0.00010001/(1.01005016708417 - cos(theta)))*((.5*(1-tanh(2e3*(r-10.01)))))*(1)
              
#              + (0.0344385979027009**(tanh(theta) + 10.844100609133) + 4.52001918428856e-19*r*theta**2/(5.73576501270149 - 2*r) + 2.89036725439893e-13)*(2.5585250149695*r + 11.6654449158227*theta + (theta + 7.6153808633482)*exp(r) + 1.05244931134205)**(cos(tanh(sin(theta))) + sech(theta + 0.192752728875375)) #❌
              
#              - ((sin(theta) + 1.75395898033809)*sin(r + 0.0428431433987448) - log(r) + tanh(sin(r)) + 11.7842644412664)**(0.74910346555205*r - 9.55266145443371) #❌
              
              + 0.0101001582000134*tanh(10.1589286596707*r + 11.9949097863706)
              
              + 0.861955660617334][f_per_idx] \
            if PERIODIC_IN_THETA else \
            (((0.148475282221305 * theta) - (sin(theta) * (1.0000132758892615 * sin(r)))) - 0.0922858190550785)
else:
    f = [\
         -mu*(r - 1)*tanh(0.01*mu)
         - 1.00999966255769*mu*sin(mu)
         + 0.149911373204456*mu
         + 1.02753406857375*nu
         + 8.36750647973528*r
#         - 0.0199999999175539*theta*(mu + 0.01)*(nu + cos(mu) + 10)
#         + 1.05373485491171*theta
         - (6.29319 - 0)*(-0.01*mu*nu + 7.20248742682568)
         - (11.6592038398793 - 0.2*mu)*(0.0100907998593378*nu - 0.0100907998593378*r + 1.45052799763474)*(0.02*nu + 0 + cos(0.02*mu**2 + 0.01*mu - 10) - 1.37000200009001)
         - 0.130002166764172*(-mu + r)*tanh(mu)
         - (mu + 0.01)*tanh(mu)
         + (4.69282041378069e-5*0 + 4.69282041378069e-5)*(sin(cos(theta)) + 125.8638)
#         - (theta + 10)*tanh(r - 10)
         + (1.58079649346906*0 + 0.0159515010339393)*(cos(sqrt(mu)) + 1)
         + ((nu + 6.28319)*asin(0.01*mu) + tanh(2*mu))*(0.02*mu**2 + 0.01*r*sech(mu) + 4.01000016667417)
         - (-0.0559725865983503*tanh(2*r - 10) - 0.0327317333341667)*(-0.0100001666741671*mu + 0.02*nu + 9.65048123121691)*(-2*0 + tanh(mu - 2) - 31.03) + 0.0100909665335049*(-0.01*mu + sin(theta) + 2.01)*(0.02*0 + (0.0001*mu + 0.001)*(2*0 + 4) - 0.0312159232024146)*cos(theta + 0.01)
         + (9.38564082756138e-6*mu*(1000 - 100*mu) + 0.000244363708390108*tanh(0 + 0.01) + 0.332651435448151)*(-1.24850674624177*r - 2.42001867002747*0 - sech(mu) - 12.846846649608)
         + (-4.69282041378069e-6*nu - (0.0101213400253702 - 1.66305743820599e-6*cos(theta))*(0.0201*mu + r + 0.0602917998593378) - 0.000100003333511123*tanh(nu) + 0.00753195968505202)*(-mu + nu - 3*0 - (9.07998593378172e-5*sin(theta) + 0.0100001666741671*cos(mu) + 0.998652910212759)*(tanh(r) + tanh(0 - 10) - 22.44) + tanh(0) + acos(sech(mu)) + 86.5977291892448)
         + ((9.99950001972143e-5*r - 1.99990000394429e-6)*(6.28314617729489*0 + 0.394782012297175)*(-0.01*nu - 0.0200001666741671*0 + 5.9633881682509) + 0.00158653047117979*sin(r) + 0.0158653047117979*sech(mu) + 0.06*sech(mu - 0.01) + 0.632188968952088)*(4.69110706287247*mu + 0.989999833325833*r + 8*0 + sin(mu) + cos(mu) + sech((1 - mu)**2) + 73.3658280287198)
         + 2*sin(mu)
         + sin(mu + 10)
         + 3.46159415595577*cos(0.0100001666741671*r*(2*r + theta + 0.01))
         + tanh(2*mu)
         - tanh(nu)
#         - tanh(0.02*theta)
#         - 0.000907998593378172*tanh(theta)
         - 0.228351234805663*tanh(sech(nu))
         + acos(0.0829319*r)
#         + 1.5*sin(theta+pi) + 1.5
         + 2*sech(mu)
         - sech(-0.137385897666266*r + 1.79261410233373*0 + (0.01 - sin(r))*(-0 - tanh(0) + 1.03) - (0.0200003333483342*r + 0.0999966667999946)*(mu*r + 5*r + 10) + 8.48770383987927)
         - 92.338744596324
         ][0]

#
print(f"f = {f}\n")
#print(f"sp.expand(f) = {sp.expand(f)}")
#latex_f = sp.latex(f)
#latex_f = latex_f.replace(r"(r", r"(\sqrt{x^2 + y^2}")
#latex_f = latex_f.replace(r"\theta", r"\arctan{\dfrac{y}{x}}")
#print(latex_f)

# Calculate the first Laplacian (Laplacian of f)
laplacian_f = diff(f, r, 2) + (1/r) * diff(f, r) + (1/(r**2)) * diff(f, theta, 2)

# Calculate the double Laplacian (Laplacian of the first Laplacian)
double_laplacian_f = diff(laplacian_f, r, 2) + (1/r) * diff(laplacian_f, r) + (1/(r**2)) * diff(laplacian_f, theta, 2)

swift_hohenberg = mu*f + nu*f*f - f*f*f - (f + 2*laplacian_f + double_laplacian_f)
if PRINT_SH_AND_EXIT:
#    print(f"swift_hohenberg = {str(swift_hohenberg.evalf()).replace('r','r_val').replace('theta', 'theta_val')}"+'\n'*20)
    from sympy.printing.pycode import pycode;
    from sympy.printing.pycode import PythonCodePrinter
    class P(PythonCodePrinter):
        def _print_sech(self, expr):
            return f"(sp.sech({self._print(expr.args[0])}))"
    printer = P();
    sh_str = printer.doprint(swift_hohenberg.evalf()).replace('math','sp')
    if len(sh_str) > 500:
        sh_filename = "sh_str.txt"
        with open(sh_filename, "w") as f:
            f.write(sh_str+'\n')
        print(f"done writing sh(f) to {sh_filename}")
    exit()

#{r<=10, r<=100, r<=1000, r<=10000, r<=100000} threshholds:
#2e3: {0.014644639263880545, 0.010045543165114933, 0.009417682123248308, 0.009346834502440376, 0.009337827805946159}
#1.8e3: {0.014645134264879296, 0.010322621280987975, 0.009418247925166098, 0.009347111268533347,  0.009337827805946159}
#1.7e3: {0.014654068054416325, 0.016449047261037967, ❌}
#1.75e3: {..., 0.011385481562662716, ❌}
#1.775e3: {0.014645675282139147, 0.010652997285208078, 0.009418752758094926, 0.009347510063070913, 0.009337827805946159}
#1.76e3: {0.0146460922350811, 0.011019015269055561 ❌}

#f_per_idx = 0:
#{0.2013604673907694, 0.2077459139516579, 0.20668954496625658, 0.20654875329411565, 0.20659748320406293}

#f_per_idx = 10:
#{0.014264540268755435, 0.008889513560533092, 0.009074180998223862, 0.009103185436913842, 0.009051156332784069}

# print(*swift_hohenberg.args, sep="\n")
r_vals, theta_vals = [None]*2
mu_vals, nu_vals = [None]*2
func_vals = None
N = 50 #echo $?
if mu_equals_nu:
    r_vals, theta_vals = np.meshgrid(np.linspace(0.01, 10, N), np.linspace(0, 2*pi, N))
else:
    r_vals, theta_vals, mu_vals, nu_vals = np.meshgrid(np.linspace(0.01, 10000000, N), np.linspace(0, 2*pi, N), np.linspace(0.0, 100, N), np.linspace(0.0, 10000, N))
#    print(f"r_vals.shape = {r_vals.shape}, theta_vals.shape = {theta_vals.shape}, mu_vals.shape = {mu_vals.shape}, nu_vals.shape = {nu_vals.shape}")
#    exit()
if not GENERIC and not COMPUTE_INIT_NUMERIC_ONLY:
    
    terms = sp.Add.make_args(diff(f, theta, 1))  # f is your full expression
    term_funcs = [sp.lambdify(((r, theta) if mu_equals_nu else (r, theta, mu, nu)), t, modules=[{"sech": sech_stable}, "numpy"])  for t in terms]

    if DEBUG_NAN:
        bad = []
        for k, tf, term in zip(range(len(terms)), term_funcs, terms):
            v = tf(r_vals, theta_vals) if mu_equals_nu else tf(r_vals, theta_vals, mu_vals, nu_vals)
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
        env = ({"r": r_vals, "theta": theta_vals} if mu_equals_nu else {"r": r_vals, "theta": theta_vals, "mu": mu_vals, "nu": nu_vals}),
        shape=r_vals.shape
    )

    squared_norm_error = LA.norm(func_vals.ravel())**2
    mean_squared_error = squared_norm_error / func_vals.size

    print(f"num DAG nodes = {len(dag_eval.nodes)}")
    print(f"squared-norm error = {squared_norm_error}")
    print(f"mean-squared error = {mean_squared_error}")
    exit()
#    func = lambdify((r, theta), swift_hohenberg, modules=[{"sech": sech_stable}, "numpy"])
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
#func_vals = func(r_vec, th_vec)
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
