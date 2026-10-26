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
from warnings import filterwarnings
from time import time
filterwarnings('ignore')
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
SAVE_HIGH_RESIDUAL_POINTS = False
N_HIGH_RESIDUAL_POINTS = 512
HIGH_RESIDUAL_FILE = "highest_residual_points_case_1.csv"

high_residual_candidates = []
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
         0.0480340324897402*r*sin(1.84080793412163*r)*sin(tanh(mu)) + (0.208036361383951 - 0.0655337432507525*tanh(mu - 0.887908976814556))*sin((r - 5.07551711479456e-5)*sin(tanh(0.542772471705644*r + 3.3615800975248e-5))*sech(tanh(tanh(tanh(mu))))) + (49.5484592544836 - 18.6908937587448*mu)*(0.195872680642965*mu*nu + 1.88578587659083*nu + 0.001408497686864) + 0.977842800665703*(0.00286990448387994*nu - 0.727733476654549)*(-0.166195579263443*nu - 0.215161826706427*sech(r) + 3.56281508996299)*sin(r)*sin(0.361043526014667*mu*nu*(nu + 5.57633302952875) - 0.794533617008966*r*sin(theta) - theta + sin(mu)*tanh(nu) + sin(mu) + tanh(nu) + sech(mu) + 8.47173548516884) + (-7.25540455008624*nu*(-mu - 9.61733762658069)*(mu - 2.65094348313446) - tanh(11.1022140247273*sech(nu)))*(-1.56607067660726e-6*nu*(mu - 12.99306758379) - 0.00180882131212939*sech(nu) + 0.504783188229183) + sech(nu + 2.58992970880596*sech(nu) - 2.28768232271063),
         0.0050776408050461*mu + 0.0050776408050461*(0.786383478861041*nu - 2.75850432660879)*(cos(0.671173820630643*mu) + 0.85362774913407)*cos(sin(nu)) - (sech(sech(mu)) - 0.329119561570178)*sin(0.011992005218829*nu + 0.011992005218829*sin(nu) + 0.385467498104794) + 0.0050776408050461*(-0.503497619162544*mu + cos(r) + 2.81305898081788)*cos(0.314026883634774*r) + (0.00246757454807468*nu + 0.00246757454807468*sech(mu - 4.60491978372231) - 0.0109613237361406)*cos(mu**2*(nu + 0.969543958634128)*cos(cos(cos(cos(nu)))))*sech(0.891957264478116*mu*sech(tanh(cos(r))) - mu + nu - sin(2*nu) + tanh(nu - 2.65357174828569)) + 0.581416912663555*(0.000821840221246166*mu**2 + 0.00164368044249233*mu + 0.000821840221246166*nu - 0.374713341093)*sin(r)*sin(0.440995297287277*mu*nu*(nu + 5.47707127413877) + nu - 0.737149487785967*r*sin(theta) - theta + 2*sin(mu) + tanh(sech(mu)) + 1.9119566515564) - 0.00716204372407795*sin(2.58238095103209*mu*nu)*tanh(nu) + 0.0050776408050461*sin(mu + 0.908912983334334) + 0.227244207469321
     ][1]
'''
best result for mu_equals_nu == False, threshold = 1
====================================================
    0.01 <= r <= 10, N = 5^2 = 25:
        f = 0.04587 \cdot r \cdot \sin{\left(1.83655 \cdot r \right)} \cdot \sin{\left(\tanh{\left(\mu \right)} \right)} + \left(0.50478 - 0.00137 \cdot \operatorname{sech}{\left(\nu \right)}\right) \cdot \left(- 7.2555 \cdot \nu \cdot \left(- \mu - 9.61757\right) \cdot \left(\mu - 2.65102\right) + \tanh{\left(\left(\mu - 31.52717\right) \cdot \operatorname{sech}{\left(\nu \right)} \right)}\right) + \left(49.5447 - 18.69024 \cdot \mu\right) \cdot \left(0.19587 \cdot \mu \cdot \nu + 1.88578 \cdot \nu + 0.00041\right) + \left(0.00237 \cdot \nu - 0.71353\right) \cdot \left(- 0.16553 \cdot \nu - 0.11703 \cdot \operatorname{sech}{\left(r \right)} + 3.585\right) \cdot \sin{\left(r \right)} \cdot \sin{\left(0.36104 \cdot \mu \cdot \nu \cdot \left(\nu + 5.57812\right) - 0.80324 \cdot r \cdot \sin{\left(\theta \right)} - \theta + \sin{\left(\mu \right)} \cdot \tanh{\left(\nu \right)} + \sin{\left(\mu \right)} + \operatorname{sech}{\left(\mu \right)} + 9.52102 \right)} + 0.11258 \cdot \sin{\left(r \cdot \sin{\left(\tanh{\left(0.63095 \cdot r + 2.0 \cdot 10^{-5} \right)} \right)} \cdot \operatorname{sech}{\left(\tanh{\left(\tanh{\left(\tanh{\left(\mu \right)} \right)} \right)} \right)} \right)} + \operatorname{sech}{\left(\left(0.32925 \cdot \mu + 1.08246 \cdot \nu - 3.39742\right) \cdot \left(0.00788 \cdot r \cdot \sin{\left(r \right)} + 1.0 \cdot 10^{-5} \cdot r + 0.05834 \cdot \operatorname{sech}{\left(r^{2} \right)} + 0.50774\right) \right)}
        f = 0.0458681047812681*r*sin(1.83655262875572*r)*sin(tanh(mu)) + (49.544698162166 - 18.6902444543227*mu)*(0.195870814803681*mu*nu + 1.88577663800403*nu + 0.000409231205202852) + (0.00236655123170145*nu - 0.713526969318837)*(-0.165532751254607*nu - 0.117027010143442*sech(r) + 3.5850015290037)*sin(r)*sin(0.361043526014667*mu*nu*(nu + 5.57811844158534) - 0.803239861860498*r*sin(theta) - theta + sin(mu)*tanh(nu) + sin(mu) + sech(mu) + 9.52101806405529) + (-7.25550417367005*nu*(-mu - 9.6175651258302)*(mu - 2.65102198105384) + tanh((mu - 31.5271687214139)*sech(nu)))*(-1.51120203929791e-6*nu*(mu - 11.1278689469132) - 0.00137279748418124*sech(nu) + 0.504783188229183) + 0.112579193587659*sin(r*sin(tanh(0.630950373076642*r + 2.04353573584371e-5))*sech(tanh(tanh(tanh(mu))))) + sech((0.329247344664438*mu + 1.08245985893031*nu - 3.39741587551312)*(0.00787678685370564*r*sin(r) + 1.03392948384204e-5*r + 0.0583414912755357*sech(r**2) + 0.507741265674475))
        SH DAG nodes = 736
        f DAG nodes = 106
        mse = 64.16378811953324; Added mu=0.01, nu=0.01
        mse = 19.27437339832776; Added mu=0.01, nu=2.51
        mse = 29.14414195937142; Added mu=0.01, nu=5
        mse = 43.7157650282404; Added mu=0.01, nu=7.5
        mse = 37.48410596735406; Added mu=0.01, nu=10
        mse = 31.32375942904362; Added mu=2.51, nu=0.01
        mse = 14.402558921341972; Added mu=2.51, nu=2.51
        mse = 43.71788935712376; Added mu=2.51, nu=5
        mse = 48.62743504462797; Added mu=2.51, nu=7.5
        mse = 42.91614242965212; Added mu=2.51, nu=10
        mse = 22.055173091020833; Added mu=5, nu=0.01
        mse = 23.42580194392449; Added mu=5, nu=2.51
        mse = 45.95345120852104; Added mu=5, nu=5
        mse = 50.63287062040184; Added mu=5, nu=7.5
        mse = 41.61210918458266; Added mu=5, nu=10
        mse = 24.30724640153968; Added mu=7.5, nu=0.01
        mse = 31.510803089176747; Added mu=7.5, nu=2.51
        mse = 47.48279273342308; Added mu=7.5, nu=5
        mse = 50.359010314300086; Added mu=7.5, nu=7.5
        mse = 39.76901404149203; Added mu=7.5, nu=10
        mse = 46.90570761866368; Added mu=10, nu=0.01
        mse = 45.19840958208546; Added mu=10, nu=2.51
        mse = 49.36570727081509; Added mu=10, nu=5
        mse = 49.50392241066862; Added mu=10, nu=7.5
        mse = 37.96322117755442; Added mu=10, nu=10
        Average mse = 39.232608013711435

    0.01 <= r <= 100, N = 5^2 = 25:
        f = \left(17.29628 - 4.06985 \cdot \mu\right) \cdot \left(0.39693 \cdot \mu \cdot \nu + 14.45471 \cdot \nu\right) + \left(\left(0.01085 \cdot \nu - 0.10741\right) \cdot \operatorname{sech}{\left(0.19868 \cdot r \right)} + 0.11377\right) \cdot \left(r^{2} \cdot \left(r - 1.6682\right) \cdot \left(r^{2} + 0.42152\right) \cdot \operatorname{sech}{\left(r + 4.04605 \right)} + \operatorname{sech}{\left(r \cdot \cos{\left(\theta \right)} \right)} - 0.56584\right) \cdot \sin{\left(1.36207 \cdot \nu^{2} \cdot \left(\mu + 2 \cdot \nu + 0.84907\right) - 1.14619 \cdot r \cdot \sin{\left(\theta \right)} + \left(\mu + \cos{\left(\mu \right)}\right) \cdot \left(\nu + \tanh{\left(\nu \right)}\right) + \cos{\left(0.97686 \cdot \nu + \cos{\left(\nu \right)} \right)} \right)} + \left(5.0 \cdot 10^{-5} \cdot \nu - 0.0065 \cdot \operatorname{sech}{\left(\nu \right)} + 1.87178\right) \cdot \left(- 0.86295 \cdot \nu \cdot \left(- \mu - 36.41249\right) \cdot \left(\mu - 4.25414\right) + 0.1589 \cdot \nu - 0.12947 \cdot \tanh{\left(\nu \right)} - 0.0038\right) - \sin{\left(1.46724 \cdot r - 7.41432 \right)} \cdot \tanh{\left(r + 0.46697 \right)} + \operatorname{sech}{\left(\left(0.29253 \cdot r - \left(0.31516 \cdot r - 0.42167\right) \cdot \sin{\left(r \right)} + 1.95622\right) \cdot \left(- 0.02109 \cdot r + \left(0.09478 \cdot r + 2.692\right) \cdot \operatorname{sech}{\left(r \right)} + \left(r + 0.26721\right) \cdot \tanh{\left(r \right)} - 2.30946\right) \right)}
        f = (17.2962780501224 - 4.06985085625717*mu)*(0.396933785904778*mu*nu + 14.4547089071054*nu) + ((0.0108535632425481*nu - 0.107406820397611)*sech(0.198678097617435*r) + 0.113768887771362)*(r**2*(r - 1.66820355779866)*(r**2 + 0.421524856200736)*sech(r + 4.04605348273777) + sech(r*cos(theta)) - 0.565836931471571)*sin(1.36207314760604*nu**2*(mu + 2*nu + 0.849072179572089) - 1.14618824682589*r*sin(theta) + (mu + cos(mu))*(nu + tanh(nu)) + cos(0.976858180269465*nu + cos(nu))) + (4.70971237114259e-5*nu - 0.00649789086170966*sech(nu) + 1.8717818515922)*(-0.86294709029387*nu*(-mu - 36.4124859344195)*(mu - 4.25413819872072) + 0.15890113130736*nu - 0.129471459028129*tanh(nu) - 0.00379556650571147) - sin(1.46724087211556*r - 7.41431501980945)*tanh(r + 0.466968782321199) + sech((0.292531947653479*r - (0.315155263064677*r - 0.421665217126719)*sin(r) + 1.95621568318622)*(-0.0210904407123763*r + (0.0947825788162892*r + 2.69199903846609)*sech(r) + (r + 0.267207992092303)*tanh(r) - 2.3094595467324))
        SH DAG nodes = 935
        f DAG nodes = 119
        mse = 2.6624788622638995; Added mu=0.01, nu=0.01
        mse = 1.291638302653289; Added mu=0.01, nu=2.51
        mse = 7.575004802774878; Added mu=0.01, nu=5
        mse = 18.559373606416653; Added mu=0.01, nu=7.5
        mse = 35.13626420389161; Added mu=0.01, nu=10
        mse = 0.3775002943013825; Added mu=2.51, nu=0.01
        mse = 2.550882819172672; Added mu=2.51, nu=2.51
        mse = 9.410437206763786; Added mu=2.51, nu=5
        mse = 20.917582721240134; Added mu=2.51, nu=7.5
        mse = 37.136902750475116; Added mu=2.51, nu=10
        mse = 4.284923633851683; Added mu=5, nu=0.01
        mse = 2.7335939381952197; Added mu=5, nu=2.51
        mse = 8.922048668556517; Added mu=5, nu=5
        mse = 20.601969105627614; Added mu=5, nu=7.5
        mse = 36.86008250248359; Added mu=5, nu=10
        mse = 14.381205785166756; Added mu=7.5, nu=0.01
        mse = 3.457489238589667; Added mu=7.5, nu=2.51
        mse = 6.936971859767226; Added mu=7.5, nu=5
        mse = 17.773259096991612; Added mu=7.5, nu=7.5
        mse = 33.66894914571926; Added mu=7.5, nu=10
        mse = 30.66999943283832; Added mu=10, nu=0.01
        mse = 12.123106338621042; Added mu=10, nu=2.51
        mse = 6.187531242145154; Added mu=10, nu=5
        mse = 14.131999579023168; Added mu=10, nu=7.5
        mse = 30.482113592467737; Added mu=10, nu=10
        Average mse = 15.153332349199918
                
best result for mu_equals_nu == False, threshold = 0.1
======================================================
    0.01 <= r <= 10, N = 5^2 = 25:
        f = - 0.00133 \cdot \mu \cdot \left(\mu + \nu \cdot \left(\operatorname{sech}{\left(\mu \right)} - 1.09129\right) + \tanh{\left(\sin{\left(\sin{\left(\nu \right)} \right)} \right)}\right) \cdot \tanh{\left(\mu \right)} + 0.01839 \cdot \mu + 0.00507 \cdot \nu \cdot \operatorname{sech}{\left(\nu \right)} - 0.01447 \cdot \nu + \left(- 0.0005 \cdot \mu + 0.00507 \cdot \cos{\left(r \right)}\right) \cdot \cos{\left(0.31424 \cdot r \right)} + \left(0.06193 \cdot \sin{\left(\mu \right)} - 0.00406\right) \cdot \operatorname{sech}{\left(5.41099 \cdot \sin{\left(\mu \right)} \right)} + \left(0.00048 \cdot \mu^{2} + 0.00096 \cdot \mu + 0.00048 \cdot \nu - 0.21812\right) \cdot \sin{\left(r \right)} \cdot \sin{\left(0.441 \cdot \mu \cdot \nu \cdot \left(\nu + 5.47708\right) + \nu - 0.73842 \cdot r \cdot \sin{\left(\theta \right)} - \theta + 2 \cdot \sin{\left(\mu \right)} + \tanh{\left(\operatorname{sech}{\left(\mu \right)} \right)} + 1.91197 \right)} + 0.00824 \cdot \sin{\left(0.55829 \cdot \mu \right)} \cdot \sin{\left(\nu^{2} \right)} + 0.0347 \cdot \cos{\left(\mu^{2} \cdot \left(\nu + 0.97276\right) \cdot \cos{\left(\cos{\left(\cos{\left(\cos{\left(\nu \right)} \right)} \right)} \right)} \right)} \cdot \operatorname{sech}{\left(0.00186 \cdot \nu + \left(\mu - 0.24614\right) \cdot \sin{\left(0.99181 \cdot \mu \right)} + \sin{\left(\nu - 2.02017 \right)} - \sin{\left(\nu + 1.11142 \right)} - 3.69735 \right)} - 0.16485 \cdot \tanh{\left(\mu + \left(0.04564 - 0.09632 \cdot \mu\right) \cdot \left(\nu - 4.84735\right) - 2.47188 \cdot \tanh{\left(\nu \right)} + 1.44519 \right)} + 0.105
        f = -0.00133497016106323*mu*(mu + nu*(sech(mu) - 1.0912942894365) + tanh(sin(sin(nu))))*tanh(mu) + 0.0183854246994495*mu + 0.00507258934054959*nu*sech(nu) - 0.0144683372562269*nu + (-0.000497105673639101*mu + 0.00507258934054959*cos(r))*cos(0.314241967598318*r) + (0.0619315177730212*sin(mu) - 0.00406486979727307)*sech(5.41098838615827*sin(mu)) + (0.000477831804139679*mu**2 + 0.000955663608279356*mu + 0.000477831804139679*nu - 0.218123135841441)*sin(r)*sin(0.440995297287277*mu*nu*(nu + 5.4770836680218) + nu - 0.73842432386237*r*sin(theta) - theta + 2*sin(mu) + tanh(sech(mu)) + 1.91196677508862) + 0.00824041153212626*sin(0.558290192162773*mu)*sin(nu**2) + 0.0346955899496206*cos(mu**2*(nu + 0.972758670012948)*cos(cos(cos(cos(nu)))))*sech(0.00185777567632561*nu + (mu - 0.246138476715741)*sin(0.991806324327853*mu) + sin(nu - 2.0201694612072) - sin(nu + 1.11142288850713) - 3.69735237624645) - 0.16485046624973*tanh(mu + (0.0456428865825602 - 0.096322166419162*mu)*(nu - 4.84734769242716) - 2.47188431452055*tanh(nu) + 1.44518696385851) + 0.105004083704277
        SH DAG nodes = 287
        f DAG nodes = 117
        mse = 0.0848668416042077; Added mu=0.01, nu=0.01
        mse = 0.07246518815898788; Added mu=0.01, nu=2.51
        mse = 0.06634935952606504; Added mu=0.01, nu=5
        mse = 0.06121379084240027; Added mu=0.01, nu=7.5
        mse = 0.07583270076595396; Added mu=0.01, nu=10
        mse = 0.04328924268414407; Added mu=2.51, nu=0.01
        mse = 0.042920599503471046; Added mu=2.51, nu=2.51
        mse = 0.04434101492530157; Added mu=2.51, nu=5
        mse = 0.05639968889128841; Added mu=2.51, nu=7.5
        mse = 0.07119514918742492; Added mu=2.51, nu=10
        mse = 0.12076818355700185; Added mu=5, nu=0.01
        mse = 0.12138276553319019; Added mu=5, nu=2.51
        mse = 0.1532521134281109; Added mu=5, nu=5
        mse = 0.10354412661039226; Added mu=5, nu=7.5
        mse = 0.0946219357771868; Added mu=5, nu=10
        mse = 0.3141539961386264; Added mu=7.5, nu=0.01
        mse = 0.27139754337310606; Added mu=7.5, nu=2.51
        mse = 0.257948322701641; Added mu=7.5, nu=5
        mse = 0.24616555480830962; Added mu=7.5, nu=7.5
        mse = 0.2893284725842479; Added mu=7.5, nu=10
        mse = 0.4713906046392527; Added mu=10, nu=0.01
        mse = 0.446575975508407; Added mu=10, nu=2.51
        mse = 0.48892923338068406; Added mu=10, nu=5
        mse = 0.4612037608024117; Added mu=10, nu=7.5
        mse = 0.4240492638535515; Added mu=10, nu=10
        Average mse = 0.1953434171514146
'''
formula_label = latex(f_float_rounded:=round_floats(f, 5), mul_symbol='dot')
def sh_residual(g):
    start=time()
    lap = (
        diff(g, r, 2)
        + diff(g, r) / r
        + diff(g, theta, 2) / r**2
    )

    bilap = (
        diff(lap, r, 2)
        + diff(lap, r) / r
        + diff(lap, theta, 2) / r**2
    )
    print(f"sh_residual time = {time() - start}")
    return mu*g + nu*g**2 - g**3 - g - 2*lap - bilap


sh_exact = sh_residual(f)
sh_rounded = sh_residual(f_float_rounded)

exact_eval = SympyDagEvaluator(sh_exact, use_cse=False)
rounded_eval = SympyDagEvaluator(sh_rounded, use_cse=False)


print(f'f = {formula_label}')
print(f'f = {f}')

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
    N_vals = 5
    # Do not build a 4D mesh. Use scalar mu, nu per plot.
    mu_plot_vals = np.linspace(0.01, 10, N_vals)
    nu_plot_vals = np.linspace(0.01, 10, N_vals)

    # Reasonable MSE/plot mesh for the parameter sweep
    N_sweep = 1000
    r_vals, theta_vals = np.meshgrid(
        np.linspace(0.01, r_max, N_sweep),
        np.linspace(0, 2*pi, N_sweep, endpoint=False)
    )

    mean_squared_error = None

print("SH DAG nodes =", len(sh_eval.nodes))
#env = {
#    "r": r_vals,
#    "theta": theta_vals,
#    "mu": 0.01,
#    "nu": 0.01,
#}
#
#res_exact = exact_eval.evaluate(env)
#res_rounded = rounded_eval.evaluate(env)
#
#print("exact MSE:", np.mean(np.square(res_exact)))
#print("rounded MSE:", np.mean(np.square(res_rounded)))
#print("difference MSE:", np.mean(np.square(res_exact - res_rounded)))
#print("max exact:", np.max(np.abs(res_exact)))
#print("max rounded:", np.max(np.abs(res_rounded)))
#print("max difference:", np.max(np.abs(res_exact - res_rounded)))
#
#lap_eval = SympyDagEvaluator(laplacian_f, use_cse=False)
#bilap_eval = SympyDagEvaluator(double_laplacian_f, use_cse=False)
#f_eval_test = SympyDagEvaluator(f, use_cse=False)
#
#f_values = f_eval_test.evaluate(env)
#lap_values = lap_eval.evaluate(env)
#bilap_values = bilap_eval.evaluate(env)
#
#algebraic_values = (
#    env["mu"] * f_values
#    + env["nu"] * f_values**2
#    - f_values**3
#    - f_values
#)
#
#components = {
#    "f": f_values,
#    "mu*f": env["mu"] * f_values,
#    "nu*f^2": env["nu"] * f_values**2,
#    "-f^3": -f_values**3,
#    "-f": -f_values,
#    "-2*lap": -2 * lap_values,
#    "-bilap": -bilap_values,
#    "algebraic total": algebraic_values,
#    "full residual": algebraic_values - 2*lap_values - bilap_values,
#}
#
#for name, values in components.items():
#    values = np.asarray(values)
#    print(
#        f"{name:16s}",
#        f"min={np.min(values): .12e}",
#        f"max={np.max(values): .12e}",
#        f"rms={np.sqrt(np.mean(values**2)): .12e}",
#    )
#
#residual = components["full residual"]
#idx = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
#
#print("maximum index:", idx)
#print("r:", r_vals[idx])
#print("theta:", theta_vals[idx])
#print("f:", f_values[idx])
#print("laplacian:", lap_values[idx])
#print("bilaplacian:", bilap_values[idx])
#print("algebraic:", algebraic_values[idx])
#print("residual:", residual[idx])

# Compute centers
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
R, Theta = np.meshgrid(r_centers, theta_centers)

# Evaluate f on cell centers using the same DAG idea
f_eval = SympyDagEvaluator(f, use_cse=False)

print("f DAG nodes =", len(f_eval.nodes))

# Convert to Cartesian
X = R * np.cos(Theta)
Y = R * np.sin(Theta)

formula_label = r"$f(r,\theta) = " + formula_label + "$"
makeCalendar = False

def mse_at(mu0, nu0):
    env = {
        "r": r_vals,
        "theta": theta_vals,
        "mu": float(mu0),
        "nu": float(nu0),
    }
    residual = sh_eval.evaluate(env)
    residual = np.nan_to_num(residual)
    return float(np.mean(residual**2))


def save_calendar_pattern(mu0=5.0, nu0=5.0):
    env = {
        "r": R,
        "theta": Theta,
        "mu": float(mu0),
        "nu": float(nu0),
    }

    Z = f_eval.evaluate(env)
    mse = mse_at(mu0, nu0)

    fig, ax = plt.subplots(figsize=(6.2, 5.6))

    zmax = np.nanmax(np.abs(Z))
    if not np.isfinite(zmax) or zmax == 0:
        zmax = 1.0

    # close periodic theta seam
    Xc = np.vstack([X, X[0:1, :]])
    Yc = np.vstack([Y, Y[0:1, :]])
    Zc = np.vstack([Z, Z[0:1, :]])

    im = ax.pcolormesh(
        Xc, Yc, Zc,
        shading="nearest",
        cmap="viridis",
        vmin=-zmax,
        vmax=zmax,
        rasterized=True,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x=r\cos\theta$")
    ax.set_ylabel(r"$y=r\sin\theta$")
    ax.set_title(rf"$\mu=\nu=5$, residual MSE $\approx {mse:.2f}$")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$f(r,\theta;\mu,\nu)$")

    fig.tight_layout()
    out = CAL_DIR / "pattern_mu5_nu5.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out}")
    return mse


def save_calendar_mse_heatmap(n_param=5):
    vals = np.linspace(0.01, 10.0, n_param)
    mse_grid = np.zeros((n_param, n_param))

    # rows = nu, columns = mu
    for i, mu0 in enumerate(vals):
        for j, nu0 in enumerate(vals):
            mse_grid[j, i] = mse_at(mu0, nu0)
            print(f"mu={mu0:.3g}, nu={nu0:.3g}, mse={mse_grid[j, i]:.6g}")

    avg_mse = float(np.mean(mse_grid))

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    dmu = vals[1] - vals[0]
    dnu = vals[1] - vals[0]

    im = ax.imshow(
        mse_grid,
        origin="lower",
        cmap="viridis",
        extent=[
            vals[0] - 0.5*dmu,
            vals[-1] + 0.5*dmu,
            vals[0] - 0.5*dnu,
            vals[-1] + 0.5*dnu,
        ],
        aspect="equal",
    )

    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\nu$")
    ax.set_title(rf"$5\times5$ validation MSE, average $\approx {avg_mse:.2f}$")

    ax.set_xticks(vals)
    ax.set_yticks(vals)
    ax.tick_params(axis="x", rotation=45)
    
    ax.set_xlim(vals[0] - 0.5*dmu, vals[-1] + 0.5*dmu)
    ax.set_ylim(vals[0] - 0.5*dnu, vals[-1] + 0.5*dnu)

    for i, mu0 in enumerate(vals):
        for j, nu0 in enumerate(vals):
            ax.text(
                mu0,
                nu0,
                f"{mse_grid[j, i]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("MSE")

    fig.tight_layout()
    out = CAL_DIR / "mse_heatmap_5x5.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out}")
    print(f"Average MSE = {avg_mse}")
    return mse_grid, avg_mse

CAL_DIR = None
if makeCalendar:
    # Calendar outputs
    from pathlib import Path

    CAL_DIR = Path("SwiftHohenbergCalendar")
    CAL_DIR.mkdir(exist_ok=True)
    hero_mse = save_calendar_pattern(mu0=5.0, nu0=5.0)
    mse_grid, avg_mse = save_calendar_mse_heatmap(n_param=5)
    exit()

def plot_one(mu0=None, nu0=None):
    global mean_squared_error, r_vals, theta_vals
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
        if SAVE_HIGH_RESIDUAL_POINTS:
            abs_residual = np.abs(func_vals.ravel())

            print(
                "quantiles:",
                np.quantile(abs_residual, [0.5, 0.9, 0.99, 0.999, 1.0])
            )

            squared = abs_residual**2
            order = np.sort(squared)[::-1]

            print(
                "SSE fractions from top 512, 1%, 5%:",
                order[:512].sum() / squared.sum(),
                order[:10_000].sum() / squared.sum(),
                order[:50_000].sum() / squared.sum(),
            )
            residual_flat = np.asarray(func_vals, dtype=np.float64).ravel()
            finite_idx = np.flatnonzero(np.isfinite(residual_flat))

            k = min(N_HIGH_RESIDUAL_POINTS, finite_idx.size)
            if k:
                finite_abs = np.abs(residual_flat[finite_idx])
                local = np.argpartition(finite_abs, -k)[-k:]
                idx = finite_idx[local]

                high_residual_candidates.append(
                    np.column_stack((
                        r_vals.ravel()[idx],
                        theta_vals.ravel()[idx],
                        np.full(k, float(mu0)),
                        np.full(k, float(nu0)),
                        residual_flat[idx],
                    ))
                )
        squared_norm_error = LA.norm(np.nan_to_num(func_vals).ravel())**2
        mse = squared_norm_error / func_vals.size
        print(f"mse = {mse}", end = "; ")
#        row_mses = np.mean(np.square(func_vals), axis=1)
#        for rv, row_mse, row_max in zip(
#            np.linspace(0.01, r_max, N_sweep),
#            row_mses,
#            np.max(np.abs(func_vals), axis=1),
#        ):
#            print(
#                f"r={rv:.8g}: "
#                f"row MSE={row_mse:.12e}, "
#                f"max |residual|={row_max:.12e}"
#            )
#        r_grid = np.linspace(0.01, r_max, N_sweep)
#        theta_grid = np.linspace(0, 2*pi, N_sweep)
#
#        r_vals, theta_vals = np.meshgrid(r_grid, theta_grid)
#        
#        print(
#            "MSE excluding r=0.01:",
#            np.mean(np.square(func_vals[:, 1:]))
#        )
#
#        for j, rv in enumerate(r_grid):
#            column = func_vals[:, j]
#            print(
#                f"r={rv:.8g}: "
#                f"MSE={np.mean(column**2):.12e}, "
#                f"max |residual|={np.max(np.abs(column)):.12e}"
#            )

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

        finite_Z = Z[np.isfinite(Z)]

        if finite_Z.size == 0:
            raise ValueError("Z contains no finite values")

        # Robust dynamic range; use np.max(...) instead if no clipping is desired.
        zmax = np.percentile(np.abs(finite_Z), 99.5)
        zmax = max(float(zmax), np.finfo(float).eps)

        vmin, vmax = -zmax, zmax
        levels = np.linspace(vmin, vmax, 100)

        is_clipped = np.max(np.abs(finite_Z)) > zmax

        Xc = np.vstack([X, X[0:1, :]])
        Yc = np.vstack([Y, Y[0:1, :]])
        Zc = np.vstack([Z, Z[0:1, :]])

        pcolormesh = 1
        plot = ax.pcolormesh(
            X,
            Y,
            Z,
            cmap="RdBu_r",  # better for a signed field; viridis also works
            vmin=vmin,
            vmax=vmax,
            shading="nearest",
            rasterized=True,
        ) if pcolormesh else ax.contourf(
            Xc,
            Yc,
            Zc,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            extend="both" if is_clipped else "neither",
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if mu_equals_nu:
            ax.set_title(f"Swift-Hohenberg 2D Pattern, MSE = {mse:.3f}")
        else:
            ax.set_title(rf"$\mu={mu0:.3g}$, $\nu={nu0:.3g}$, MSE = {mse:.3e}")

        cbar = fig.colorbar(
            plot,
            ax=ax,
            extend="both" if is_clipped else "neither",
        )

        cbar.set_label(formula_label, fontsize=7)
        cbar.set_ticks([vmin, 0.0, vmax])
        cbar.set_ticklabels([
            f"{vmin:.3g}",
            "0",
            f"{vmax:.3g}",
        ])

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
#    total_sse_all = 0.0
#    total_sse_without_inner = 0.0
#    total_sse_inner_zeroed = 0.0
#    total_count_all = 0
#    total_count_without_inner = 0
#
#    for mu0 in mu_plot_vals:
#        for nu0 in nu_plot_vals:
#            env_mse = {
#                "r": r_vals,
#                "theta": theta_vals,
#                "mu": float(mu0),
#                "nu": float(nu0),
#            }
#
#            residual = sh_eval.evaluate(env_mse)
#
#            total_sse_all += np.sum(residual**2)
#            total_count_all += residual.size
#
#            residual_without_inner = residual[:, 1:]
#            total_sse_without_inner += np.sum(residual_without_inner**2)
#            total_count_without_inner += residual_without_inner.size
#
#            residual_inner_zeroed = residual.copy()
#            residual_inner_zeroed[:, 0] = 0.0
#            total_sse_inner_zeroed += np.sum(residual_inner_zeroed**2)
#
#    print("all-point SSE:", total_sse_all)
#    print("all-point MSE:", total_sse_all / total_count_all)
#
#    print("exclude-r=.01 SSE:", total_sse_without_inner)
#    print(
#        "exclude-r=.01 MSE:",
#        total_sse_without_inner / total_count_without_inner,
#    )
#
#    print("zero-r=.01 SSE:", total_sse_inner_zeroed)
#    print(
#        "zero-r=.01 but retain 6561 denominator:",
#        total_sse_inner_zeroed / total_count_all,
#    )
#    exit()

#    point = {
#        r: 0.01,
#        theta: 3.9269908169872414,
#        mu: 0.01,
#        nu: 0.01,
#    }
#
#    expressions = {
#        "f": f,
#        "fr": diff(f, r),
#        "frr": diff(f, r, 2),
#        "frrr": diff(f, r, 3),
#        "frrrr": diff(f, r, 4),
#        "ftt": diff(f, theta, 2),
#        "frtt": diff(f, theta, 2, r),
#        "frrtt": diff(f, theta, 2, r, 2),
#        "ftttt": diff(f, theta, 4),
#    }
#
#    for name, expression in expressions.items():
#        print(
#            f"{name:8s}",
#            f"{float(expression.evalf(17, subs=point)):.17g}",
#        )
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
    if SAVE_HIGH_RESIDUAL_POINTS and high_residual_candidates:
        candidates = np.vstack(high_residual_candidates)

        k = min(N_HIGH_RESIDUAL_POINTS, len(candidates))
        idx = np.argpartition(np.abs(candidates[:, 4]), -k)[-k:]
        highest = candidates[idx]

        # Save in descending |residual| order.
        highest = highest[
            np.argsort(np.abs(highest[:, 4]))[::-1]
        ]

        np.savetxt(
            f'{HIGH_RESIDUAL_FILE.replace(".csv","")}_with_res.csv',
            highest,
            delimiter=",",
            header="r,theta,mu,nu,residual",
            comments="",
        )
        
        np.savetxt(
            HIGH_RESIDUAL_FILE,
            highest[:,:-1],
            delimiter=",",
            header="r,theta,mu,nu",
            comments="",
        )

        print(f"Saved {k} highest-residual points to {HIGH_RESIDUAL_FILE}")

    print(f"Average mse = {total_mse/count}")
    print(f"Saved {filename}")
    system(f"open {filename}")
