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
r_max = 100
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
     0.750670805917889*((-0.000141707574371161*nu - 0.165067789235591)*(1.0001257169309*nu + 3.09202110589853) + 0.120947989281476*(sech(r) - 5.11586605849299)*sech(r) + 4.09696360042431)*sin(r)*sin(0.361048148966913*mu*nu*(nu + 5.57242342973361) - 0.787021991993375*r*sin(theta) - theta + sin(mu)*sech(mu) + sin(mu) + sech(mu) + 9.47594015796012) + sin(0.471863704218753*mu - 0.27656177041966)*sin(mu - sin(1.03980442827475*mu))*tanh(0.453846343161461*tanh(sin(mu))) + sin(sech(mu)) + sin(sech(sech(r))) + sech(sech(mu)) - 1.97117788657096,
     (4.12599679465333 - 1.62549402814435*mu)*(0.188484936229902*mu*nu + 3.24376155812461*nu) + (-0.66541325925397*nu*(-mu - 17.0705294022282)*(mu - 2.54266023358934) + tanh(0.802153102627979*mu - 7.03712393934889))*(7.67693825438223e-6*(nu - 369.520730863518)*sech(mu) - 0.0276293194324254*sech(nu) + 0.462459369255155) + 0.617890629351021*(0.000892567772483054*nu + 0.00178513554496611*r - tanh(sin(mu)) - 0.000976456951950205)*sin(r)*sin(-1.35332054484561*mu*nu*(1.00081738195407*nu + 4.87647929091422) + 1.16541731522825*mu + nu + 0.716230318263056*r*sin(theta) + 0.767254900695391*tanh(nu))*sin(tanh(r**2*sin(mu))) + sin(1.44839792605419*r)*tanh(r) + sech((-0.0471939200816425*mu - sech(mu) + 0.296339609565172)*(mu + nu + sech(nu) + sech(r) - 0.0959427010002434 + sech(1)))
     ][3]
'''
best result for mu_equals_nu == False
=====================================
    0.01 <= r <= 10, N = 3^2 = 9:
        f = 0.18248 \cdot \mu \cdot \nu \cdot \operatorname{sech}{\left(\mu \right)} \cdot \operatorname{sech}{\left(r \right)} \cdot \operatorname{sech}{\left(\mu \cdot \sin{\left(\mu \right)} \right)} + 0.08098 \cdot \mu + 0.08098 \cdot \nu + \left(3.1899 - 1.70304 \cdot \mu\right) \cdot \left(0.08355 \cdot \mu \cdot \nu + 1.61025 \cdot \nu + 0.06418\right) + \left(- 0.01108 \cdot \mu - 0.03343\right) \cdot \sin{\left(\sin{\left(\mu \right)} + 3.2632 \cdot \tanh{\left(\mu \right)} + 1.63789 \right)} \cdot \tanh{\left(\mu \cdot \nu \cdot \sin{\left(\nu \right)} \right)} \cdot \operatorname{sech}{\left(\tanh{\left(\nu \right)} \right)} + \left(- 1.4318 \cdot \nu + \tanh{\left(\mu \right)}\right) \cdot \left(0.05097 \cdot \nu + 0.07448\right) \cdot \left(- 0.11094 \cdot \mu + \nu + 2.56224\right) \cdot \left(0.36957 \cdot \mu - 1.73341 \cdot \nu + 4.00746\right) \cdot \sin{\left(\operatorname{sech}{\left(1.70855 \cdot \nu \right)} \right)} - \left(- 0.03169 \cdot r - 1.0 \cdot 10^{-5}\right) \cdot \sin{\left(1.8426 \cdot r \right)} + \left(- 1.0 \cdot 10^{-5} \cdot \nu \cdot \left(\nu + 3.2739\right) + 0.39218\right) \cdot \left(- 0.36359 \cdot \nu \cdot \left(- \mu - 19.31789\right) \cdot \left(\mu - 1.86873\right) + \tanh{\left(\left(\mu - 48.17516\right) \cdot \operatorname{sech}{\left(\nu \right)} \right)}\right) + \left(\nu + \sin{\left(\nu \right)} - 1.13148\right) \cdot \operatorname{sech}{\left(1.8691 \cdot \nu \right)} + \left(\left(- 0.04803 \cdot \mu + \sin{\left(\nu \right)}\right) \cdot \left(\nu - 9.84045\right) \cdot \left(\nu + 145.92019\right) \cdot \left(\tanh{\left(\nu \right)} - 1.03912\right) \cdot \sin{\left(\mu \right)} \cdot \tanh{\left(\nu \right)} + \sin{\left(\sin{\left(\nu \right)} \right)} + \operatorname{sech}{\left(\mu \right)}\right) \cdot \left(- 0.00015 \cdot \mu + 0.00203 \cdot \sin{\left(\left(\mu + 1.05586\right) \cdot \tanh{\left(\mu \right)} \right)} - 0.17255 \cdot \tanh{\left(7.23679 \cdot \mu \right)} + 0.17253\right) - \left(- 0.1221 \cdot \nu + 0.72766 \cdot \left(0.10871 \cdot \operatorname{sech}{\left(r \right)} - 0.72956\right) \cdot \operatorname{sech}{\left(r \right)} + 0.72766 \cdot \operatorname{sech}{\left(\nu + 12.24657 \right)} + 2.60282\right) \cdot \sin{\left(r \right)} \cdot \sin{\left(0.36107 \cdot \mu \cdot \nu \cdot \left(\nu + 5.57474\right) - 0.82061 \cdot r \cdot \sin{\left(\theta \right)} - \theta + \sin{\left(\mu \right)} \cdot \tanh{\left(\nu \right)} + \sin{\left(\mu \right)} + \operatorname{sech}{\left(\mu \right)} + 9.50239 \right)} + \left(0.01007 \cdot \sin{\left(\nu \right)} + 0.01205 \cdot \sin{\left(1.82368 \cdot \mu + 1.4033 \right)} \cdot \tanh{\left(\sin{\left(\nu \right)} \right)} + 0.14028 \cdot \operatorname{sech}{\left(\nu \right)} \cdot \operatorname{sech}{\left(0.75402 \cdot r \right)} + 0.08823\right) \cdot \left(- \mu - \nu - \sin{\left(\sin{\left(\tanh{\left(\nu \right)} \right)} \right)} - \operatorname{sech}{\left(\mu - 0.33124 \right)} + 7.38655\right) - 0.08098 \cdot \sin{\left(\sin{\left(\nu \right)} \right)} + \operatorname{sech}{\left(\left(\nu + \left(0.42939 \cdot \mu + \operatorname{sech}{\left(r \right)}\right) \cdot \left(0.05143 \cdot \nu + 0.46824\right) + \tanh{\left(\mu \right)} - 3.97193\right) \cdot \left(- 2.0 \cdot 10^{-5} \cdot r \cdot \tanh{\left(\nu \right)} + 3.0 \cdot 10^{-5} \cdot r + \operatorname{sech}{\left(\mu \cdot \nu + 1.66047 \right)} + 0.47767\right) \right)} + \operatorname{sech}{\left(\nu^{2} \cdot \left(\mu + 20.22899\right) \cdot \left(1.23326 \cdot \nu - 3.6156\right) \right)} - 0.64388
        f = -1.37327324963925e-7*mu*nu*r + 0.182480361704898*mu*nu*sech(mu)*sech(r)*sech(mu*sin(mu)) + 0.0809768987063034*mu + 0.0809768987063034*nu - 3.95046794508987e-6*r + (3.18990041457538 - 1.70303592965154*mu)*(0.0835470372689606*mu*nu + 1.61025089416828*nu + 0.0641763007787589) + (-0.011083281470764*mu - 0.0334301321341606)*sin(sin(mu) + 3.26320098861293*tanh(mu) + 1.63789409341858)*tanh(mu*nu*sin(nu))*sech(tanh(nu)) + (-1.43180243117527*nu + tanh(mu))*(0.0509701397441228*nu + 0.0744787222733847)*(-0.110939485063588*mu + nu + 2.56223860235202)*(0.369570977726661*mu - 1.7334102029509*nu + 4.00746342208323)*sin(sech(1.7085475986183*nu)) - (-0.0316870068074179*r - 5.18998989257778e-6)*sin(1.84259626519388*r) + (-6.84598129579738e-6*nu*(nu + 3.27390357946227) + 0.392177170119054)*(-0.363591809862454*nu*(-mu - 19.3178924832079)*(mu - 1.86872973306398) + tanh((mu - 48.1751633038631)*sech(nu))) + (nu + sin(nu) - 1.131475374541)*sech(1.86910084328575*nu) + ((-0.0480328328947595*mu + sin(nu))*(nu - 9.84044500704665)*(nu + 145.920185584766)*(tanh(nu) - 1.03911597475944)*sin(mu)*tanh(nu) + sin(sin(nu)) + sech(mu))*(-0.000149594348314223*mu + (0.00202946307526777 - 5.89717545906865e-8*r)*sin((mu + 1.0558595938956)*tanh(mu)) - 0.172547821493815*tanh(7.23678710160448*mu) + 0.172533993291247) - (-0.122098591510301*nu + 0.727655794026031*(0.108711530496075*sech(r) - 0.729556641950236)*sech(r) + 0.727655794026031*sech(nu + 12.2465744204158) + 2.60281500316559)*sin(r)*sin(0.361068519008264*mu*nu*(nu + 5.57474342469274) - 0.820610053684433*r*sin(theta) - theta + sin(mu)*tanh(nu) + sin(mu) + sech(mu) + 9.50238891655399) + (0.0100656714997877*sin(nu) + 0.0120496037709609*sin(1.82367743949262*mu + 1.40330122075086)*tanh(sin(nu)) + 0.14027775287786*sech(nu)*sech(0.754017988040545*r) + 0.0882256014306345)*(-mu - nu - sin(sin(tanh(nu))) - sech(mu - 0.331244207234973) + 7.38654588359899) - 0.0809768987063034*sin(sin(nu)) + sech((nu + (0.429394529241447*mu + sech(r))*(0.0514320583581283*nu + 0.46824156212343) + tanh(mu) - 3.97193001789239)*(-2.3417347056994e-5*r*tanh(nu) + 2.5221095402203e-5*r + sech(mu*nu + 1.66047009911464) + 0.477670844764273)) + sech(nu**2*(mu + 20.2289900527862)*(1.2332556018553*nu - 3.61560033839232)) - 0.643880090362268
        SH DAG nodes = 722
        f DAG nodes = 225
        mse = 66.12790579272901; Added mu=0.01, nu=0.01
        mse = 32.34157405782723; Added mu=0.01, nu=5
        mse = 43.754559947123; Added mu=0.01, nu=10
        mse = 26.031306353809477; Added mu=5, nu=0.01
        mse = 48.17336196498448; Added mu=5, nu=5
        mse = 44.593146862748874; Added mu=5, nu=10
        mse = 40.89267390993185; Added mu=10, nu=0.01
        mse = 56.540526172071; Added mu=10, nu=5
        mse = 41.06601076022739; Added mu=10, nu=10
        Average mse = 44.39122953571692

    0.01 <= r <= 10, N = 5^2 = 25:
        f = 0.04202 \cdot r \cdot \sin{\left(1.84203 \cdot r \right)} \cdot \sin{\left(\tanh{\left(\nu \right)} \right)} + \left(3.66045 - 1.59695 \cdot \mu\right) \cdot \left(0.14716 \cdot \mu \cdot \nu + 2.86176 \cdot \nu + 0.02221\right) + \left(- 0.5455 \cdot \nu \cdot \left(- \mu - 19.28838\right) \cdot \left(\mu - 2.2933\right) + \tanh{\left(\left(\mu - 32.26396\right) \cdot \operatorname{sech}{\left(\nu \right)} \right)}\right) \cdot \left(- 2.0 \cdot 10^{-5} \cdot \nu \cdot \left(\mu - 9.80888\right) - 0.01071 \cdot \operatorname{sech}{\left(\nu \right)} + 0.433\right) + \left(0.00079 \cdot \nu + 0.00079 \cdot r - 0.72662\right) \cdot \left(- 0.16509 \cdot \nu + \left(0.10836 \cdot \operatorname{sech}{\left(r \right)} - 0.60432\right) \cdot \operatorname{sech}{\left(r \right)} + 0.00143 \cdot \tanh{\left(r \right)} + 3.58987\right) \cdot \sin{\left(r \right)} \cdot \sin{\left(0.36107 \cdot \mu \cdot \nu \cdot \left(\nu + 5.57927\right) + \nu - 0.80527 \cdot r \cdot \sin{\left(\theta \right)} - \theta + \sin{\left(\mu \right)} \cdot \tanh{\left(\nu \right)} + \sin{\left(\mu \right)} + 9.50348 \right)} + \operatorname{sech}{\left(\left(\nu + \left(0.35207 \cdot \mu + \operatorname{sech}{\left(r \right)}\right) \cdot \left(0.05625 \cdot \nu + 0.54293\right) + \tanh{\left(\mu \right)} - 4.00376\right) \cdot \left(- 0.00207 \cdot \mu + 0.00198 \cdot r \cdot \tanh{\left(r \right)} + 1.0 \cdot 10^{-5} \cdot r - 0.00207 \cdot \sin{\left(\mu \right)} + 0.50878\right) \right)}
        f = (3.66044940646846 - 1.59694854303305*mu)*(0.147161851352679*mu*nu + 2.86175962110052*nu + 0.0222110061657081) - (-0.0420236894767088*r - 4.76308017692989e-6)*sin(1.84203491117177*r)*sin(tanh(nu)) + (-0.545503734946519*nu*(-mu - 19.2883836821421)*(mu - 2.2933036404516) + tanh((mu - 32.2639623115106)*sech(nu)))*(-1.64861515435008e-5*nu*(mu - 9.80888437164882) - 0.0107129827508127*sech(nu) + 0.432995632472821) + (0.000790449536029959*nu + 0.000790449536029959*r - 0.726616118554963)*(-0.165088531998991*nu + (0.108359319038353*sech(r) - 0.604320114497875)*sech(r) + 0.00143035805032774*tanh(r) + 3.58987282517415)*sin(r)*sin(0.361068078737719*mu*nu*(nu + 5.57926688644434) + nu - 0.805270356178479*r*sin(theta) - theta + sin(mu)*tanh(nu) + sin(mu) + 9.50347512804185) + sech((nu + (0.352071372430706*mu + sech(r))*(0.0562468962873796*nu + 0.542932774769565) + tanh(mu) - 4.0037646627928)*(-0.00207270390469954*mu + 0.00198265580477616*r*tanh(r) + 6.44122891442208e-6*r - 0.00207270390469954*sin(mu) + 0.508780736608017))
        SH DAG nodes = 630
        f DAG nodes = 104
        mse = 69.30870637900397; Added mu=0.01, nu=0.01
        mse = 22.70836210537753; Added mu=0.01, nu=2.51
        mse = 37.484827138237904; Added mu=0.01, nu=5
        mse = 51.18779313851249; Added mu=0.01, nu=7.5
        mse = 37.677895840178145; Added mu=0.01, nu=10
        mse = 30.920612812778597; Added mu=2.51, nu=0.01
        mse = 16.863266284710168; Added mu=2.51, nu=2.51
        mse = 42.03935167381561; Added mu=2.51, nu=5
        mse = 52.25725448406311; Added mu=2.51, nu=7.5
        mse = 50.6341039148913; Added mu=2.51, nu=10
        mse = 17.536342186512552; Added mu=5, nu=0.01
        mse = 24.95644316763983; Added mu=5, nu=2.51
        mse = 51.332196268019295; Added mu=5, nu=5
        mse = 58.24262172303627; Added mu=5, nu=7.5
        mse = 48.384189656277016; Added mu=5, nu=10
        mse = 22.52397457283456; Added mu=7.5, nu=0.01
        mse = 33.77022351098843; Added mu=7.5, nu=2.51
        mse = 50.499043553050285; Added mu=7.5, nu=5
        mse = 51.867898817387555; Added mu=7.5, nu=7.5
        mse = 46.043892499418945; Added mu=7.5, nu=10
        mse = 43.31773625445272; Added mu=10, nu=0.01
        mse = 38.63558265577251; Added mu=10, nu=2.51
        mse = 55.33219267264568; Added mu=10, nu=5
        mse = 56.043018130881244; Added mu=10, nu=7.5
        mse = 46.9377608249458; Added mu=10, nu=10
        Average mse = 42.26021161061726

    0.01 <= r <= 100, N = 5^2 = 25:
        f = \left(4.126 - 1.62549 \cdot \mu\right) \cdot \left(0.18848 \cdot \mu \cdot \nu + 3.24376 \cdot \nu\right) + \left(- 0.66541 \cdot \nu \cdot \left(- \mu - 17.07053\right) \cdot \left(\mu - 2.54266\right) + \tanh{\left(0.80215 \cdot \mu - 7.03712 \right)}\right) \cdot \left(\left(1.0 \cdot 10^{-5} \cdot \nu - 0.00284\right) \cdot \operatorname{sech}{\left(\mu \right)} - 0.02763 \cdot \operatorname{sech}{\left(\nu \right)} + 0.46246\right) + \left(0.00055 \cdot \nu + 0.0011 \cdot r - 0.61789 \cdot \tanh{\left(\sin{\left(\mu \right)} \right)} - 0.0006\right) \cdot \sin{\left(r \right)} \cdot \sin{\left(- 1.35332 \cdot \mu \cdot \nu \cdot \left(1.00082 \cdot \nu + 4.87648\right) + 1.16542 \cdot \mu + \nu + 0.71623 \cdot r \cdot \sin{\left(\theta \right)} + 0.76725 \cdot \tanh{\left(\nu \right)} \right)} \cdot \sin{\left(\tanh{\left(r^{2} \cdot \sin{\left(\mu \right)} \right)} \right)} + \sin{\left(1.4484 \cdot r \right)} \cdot \tanh{\left(r \right)} + \operatorname{sech}{\left(\left(- 0.04719 \cdot \mu - \operatorname{sech}{\left(\mu \right)} + 0.29634\right) \cdot \left(\mu + \nu + \operatorname{sech}{\left(\nu \right)} + \operatorname{sech}{\left(r \right)} - 0.09594 + \operatorname{sech}{\left(1 \right)}\right) \right)}
        f = (4.12599679465333 - 1.62549402814435*mu)*(0.188484936229902*mu*nu + 3.24376155812461*nu) + (-0.66541325925397*nu*(-mu - 17.0705294022282)*(mu - 2.54266023358934) + tanh(0.802153102627979*mu - 7.03712393934889))*((7.67693825438223e-6*nu - 0.00283678783455342)*sech(mu) - 0.0276293194324254*sech(nu) + 0.462459369255155) + (0.000551509262677993*nu + 0.00110301852535599*r - 0.617890629351021*tanh(sin(mu)) - 0.000603343600574692)*sin(r)*sin(-1.35332054484561*mu*nu*(1.00081738195407*nu + 4.87647929091422) + 1.16541731522825*mu + nu + 0.716230318263056*r*sin(theta) + 0.767254900695391*tanh(nu))*sin(tanh(r**2*sin(mu))) + sin(1.44839792605419*r)*tanh(r) + sech((-0.0471939200816425*mu - sech(mu) + 0.296339609565172)*(mu + nu + sech(nu) + sech(r) - 0.0959427010002434 + sech(1)))
        SH DAG nodes = 621
        f DAG nodes = 91
        mse = 12.161225963889237; Added mu=0.01, nu=0.01
        mse = 10.327657527607041; Added mu=0.01, nu=2.51
        mse = 16.701431312102265; Added mu=0.01, nu=5
        mse = 32.74555287578382; Added mu=0.01, nu=7.5
        mse = 105.40085718766315; Added mu=0.01, nu=10
        mse = 21.70551697680684; Added mu=2.51, nu=0.01
        mse = 13.741895352056163; Added mu=2.51, nu=2.51
        mse = 36.08553540684015; Added mu=2.51, nu=5
        mse = 48.93151500780082; Added mu=2.51, nu=7.5
        mse = 79.67022515850911; Added mu=2.51, nu=10
        mse = 59.960448384630475; Added mu=5, nu=0.01
        mse = 22.826314691038988; Added mu=5, nu=2.51
        mse = 27.179846266314975; Added mu=5, nu=5
        mse = 93.70364448401985; Added mu=5, nu=7.5
        mse = 250.36076850441373; Added mu=5, nu=10
        mse = 195.1056148937996; Added mu=7.5, nu=0.01
        mse = 10.945206660129136; Added mu=7.5, nu=2.51
        mse = 148.52264134061116; Added mu=7.5, nu=5
        mse = 35.46419728031543; Added mu=7.5, nu=7.5
        mse = 101.45884852158026; Added mu=7.5, nu=10
        mse = 56.989026912463906; Added mu=10, nu=0.01
        mse = 23.73950449157789; Added mu=10, nu=2.51
        mse = 71.53036551901148; Added mu=10, nu=5
        mse = 41.80361397161569; Added mu=10, nu=7.5
        mse = 44.691821086100106; Added mu=10, nu=10
        Average mse = 62.47013103106726
'''
formula_label = latex(f_float_rounded:=round_floats(f, 5), mul_symbol='dot')
def sh_residual(g):
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

        vmin, vmax = -1, 1
        min_, max_ = vmin, vmax
        levels = np.linspace(vmin, vmax, 100)

        Xc = np.vstack([X, X[0:1, :]])
        Yc = np.vstack([Y, Y[0:1, :]])
        Zc = np.vstack([Z, Z[0:1, :]])

        pcolormesh = 1
        plot = ax.pcolormesh(
            X,
            Y,
            Z,
            cmap="viridis",
            vmin=-1,
            vmax=1,
            shading="nearest",
            rasterized=True
        ) if pcolormesh else ax.contourf(
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

        cbar = fig.colorbar(plot, ax=ax, extend = "both" if pcolormesh else "neither")
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
