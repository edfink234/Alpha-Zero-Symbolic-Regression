from interval import interval
from interval.imath import *
from sympy import symbols
import sympy as sp
from numpy import linspace
from itertools import product
import math


class Interval:
    @staticmethod
    def _as_interval(v):
        if isinstance(v, interval):
            return v
        return interval[v, v]

    @staticmethod
    def pow(x, y):
        xI = Interval._as_interval(x)
        yI = Interval._as_interval(y)

        # integer exponent fast path
        if len(yI) == 1 and yI[0][0] == yI[0][1]:
            yy = yI[0][0]
            if float(yy).is_integer():
                n = int(yy)

                if n == 0:
                    return interval[1.0, 1.0]
                if n < 0:
                    return Interval.pow(interval[1.0, 1.0] / xI, -n)

                out = xI
                for _ in range(n - 1):
                    out = out * xI
                return out

        return exp(yI * log(xI))


class IntervalBackend:
    def const(self, c):
        return Interval._as_interval(float(c))

    def add(self, a, b):
        return a + b

    def sub(self, a, b):
        return a - b

    def mul(self, a, b):
        return a * b

    def div(self, a, b):
        return a / b

    def pow(self, a, b):
        return Interval.pow(a, b)

    def sin(self, x):
        return sin(x)

    def cos(self, x):
        return cos(x)

    def exp(self, x):
        return exp(x)

    def log(self, x):
        return log(x)

    def sqrt(self, x):
        return sqrt(x)

    def tanh(self, x):
        return tanh(x)

    def abs(self, x):
        return abs(x)

    def neg(self, x):
        return -x


class IntervalDagEvaluator:
    """
    DAG evaluator for SymPy expressions using interval arithmetic backend.
    Repeated subexpressions are hash-consed and evaluated once per point.
    """

    def __init__(self, expr, use_cse=False):
        self.original_expr = expr
        self.nodes = []
        self.root = -1
        self._intern_expr = {}
        self._intern_node = {}

        if use_cse:
            repls, reduced = sp.cse(expr, optimizations="basic")
            if len(reduced) != 1:
                raise RuntimeError("Expected a single reduced expression from cse.")
            expr2 = reduced[0]
            for sym, rhs in repls[::-1]:
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

            elif expr.func is sp.Abs:
                node_id = self._intern("abs", None, args)

            elif expr.func is sp.sech:
                # represent sech(x) as 1/cosh(x) = 2/(exp(x)+exp(-x))
                # rewrite and build the rewritten expr
                rewritten = 2 / (sp.exp(expr.args[0]) + sp.exp(-expr.args[0]))
                node_id = self._build(rewritten)

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

    def evaluate(self, env, backend):
        values = {}

        for i in self.postorder:
            op, data, children = self.nodes[i]

            if op == "symbol":
                if data not in env:
                    raise KeyError("Missing value for symbol '%s'" % data)
                values[i] = env[data]

            elif op == "const":
                values[i] = backend.const(data)

            else:
                ch = [values[c] for c in children]

                if op == "add":
                    out = ch[0]
                    for x in ch[1:]:
                        out = backend.add(out, x)

                elif op == "mul":
                    out = ch[0]
                    for x in ch[1:]:
                        out = backend.mul(out, x)

                elif op == "pow":
                    out = backend.pow(ch[0], ch[1])

                elif op == "sin":
                    out = backend.sin(ch[0])

                elif op == "cos":
                    out = backend.cos(ch[0])

                elif op == "exp":
                    out = backend.exp(ch[0])

                elif op == "log":
                    out = backend.log(ch[0])

                elif op == "sqrt":
                    out = backend.sqrt(ch[0])

                elif op == "tanh":
                    out = backend.tanh(ch[0])

                elif op == "abs":
                    out = backend.abs(ch[0])

                else:
                    raise RuntimeError("Unknown op '%s'" % op)

                values[i] = out

        return values[self.root]


def square_interval(x):
    return x * x


# -------------------------
# problem setup
# -------------------------

N = 1000
PERIODIC_IN_THETA = True

r_float_vals = linspace(0.01, 100000, N).tolist()
theta_float_vals = linspace(0, 2 * math.pi, N).tolist()

eps = 1e-14
r_vals = [interval[v - eps, v + eps] for v in r_float_vals]
theta_vals = [interval[v - eps, v + eps] for v in theta_float_vals]

SH = symbols('\\text{SwiftHohenberg} r theta mu nu')
r, theta = symbols('r theta')
mu, nu = 1, 1

# -----------------------------------------------------------
# PUT YOUR SYMPY EXPRESSION HERE
#
# This should be the same expression that your current huge
# f_val_func lambda is computing, but as a SymPy expression.
#
# Example placeholder:
# -----------------------------------------------------------
r_val, theta_val = r, theta
expr = (
    -(sp.sin(r) - sp.cos(r)/r + 3.0*sp.sin(r)/r**2 + 6.0*sp.cos(r)/r**3 - 6.0*sp.sin(r)/r**4)*sp.sin(theta) - sp.sin(r)**3*sp.sin(theta)**3 + sp.sin(r)**2*sp.sin(theta)**2 + 2.0*sp.sin(r)*sp.sin(theta) - (-sp.sin(theta)*sp.cos(r) - sp.sin(r)*sp.sin(theta)/r - 2.0*sp.sin(theta)*sp.cos(r)/r**2 + 2.0*sp.sin(r)*sp.sin(theta)/r**3)/r - 2.0*sp.sin(theta)*sp.cos(r)/r - (sp.sin(r) - sp.cos(r)/r + sp.sin(r)/r**2)*sp.sin(theta)/r**2 + 2.0*sp.sin(r)*sp.sin(theta)/r**2,
    eval(open("sh_str.txt").read())
)[0]

print(f"expr instantiated")

# If you already have a SymPy SH residual expression from the symbolic script,
# replace expr above with that exact SymPy expression.


# -------------------------
# build DAG once
# -------------------------

backend = IntervalBackend()
dag = IntervalDagEvaluator(expr, use_cse=False)

print("num DAG nodes =", len(dag.nodes))
print(r_vals[0], r_vals[0][0][0], r_vals[0][0][1])

# -------------------------
# evaluate over interval grid
# -------------------------

f_vals = []
for r_val, theta_val in product(r_vals, theta_vals):
    val = dag.evaluate({"r": r_val, "theta": theta_val}, backend)
    f_vals.append(val)

print("\nlen(f_vals) =", len(f_vals))

f_vals_squared = [square_interval(v) for v in f_vals]

total = interval[0, 0]
for v2 in f_vals_squared:
    total += v2

print("total =", total)
#print("zip(product(r_vals, theta_vals), f_vals_squared) = ", *list(zip(product(r_vals, theta_vals), f_vals_squared)), sep='\n')

"""
Stats
=====

f = sin(r)*sin(theta):
    (1000 x 1000), (0.01 <= r <= 10, 0 <= θ <= 2π): interval([201360.4665727735, 201360.46821284745])
    (1000 x 1000), (0.01 <= r <= 100, 0 <= θ <= 2π): interval([207745.91327080742, 207745.9146365905])
    (1000 x 1000), (0.01 <= r <= 1000, 0 <= θ <= 2π): interval([206689.54428604964, 206689.54565055823])
    (1000 x 1000), (0.01 <= r <= 10000, 0 <= θ <= 2π): interval([206548.75261390474, 206548.75397840884])
    (1000 x 1000), (0.01 <= r <= 100000, 0 <= θ <= 2π): interval([206597.48252385668, 206597.48388835927])

f = 6.41032241779392/(1.28262426683707*r - 21890.2917193253) - ((1e-10 + (r + 0.0675028199851666*sin(theta) + 0.315589358780667)**(sqrt(r)*(r + 2.87892339678315)*(5836.48192235195 - r)/((5836.48192235195 - r)**2 + 1e10) + 0.980141037771426)/(0.013519701745416**r*(43.6481950729572*r + 8.49265186112673) + r + 0.02*sin(r) + 1.82648253593279))**(((9.23274577772655)/(r + 0.55183450665909) + sin(theta + cos(theta + 0.520501469061028) + 5.88393395176072)*(.5*(1-tanh(1.025*(r-21.2)))))*(r + sin(r - 0.01) + (1.0e-10 + cos(sin(theta)))**(r - 1.58836944951552) + 0.37384427398835 + tanh(r)/(r + 7.97719844610347))))*((.5*(1-tanh(2e3*(r-10.01))))) + sqrt(1 - cos(r)**2)*(1.0e-10*0.68688067225485**(8.47949129673737*r) + 0.854229974212735)*(sech(r + cos(r) + 8.39614384384391) + 0.999884853180843)**(1.58799646315658*(r + 0.0308839840501129)**4.01549520152667*(1.57*(tanh(.62*r))))*(0.0100048594945809**(2*r + 5.29438341416157) + 0.7011748940086 - 47.9851001514431/(21870.6640272238))*sin(theta + 18.8962439891879) + (223.780532729645)/(sech(1.24592745883222/r) + 22066.7916781747) - ((1.5707963267949)**(-18.3890761348268) + 0.285811651486423)**(r + sech(r + 0.058634538594504) + 10.0547134992516)*(r + (r**0.999950000416665 - 0.00364405505237706)**((0.367957484641359**r + r)**0.00999966667999946) + 0.0106243230277353)**(-r**2*exp(-r)/(1 + 1348.60891792031*exp(-r)) + (0.000497198311398096*r + 0.0160184860388267)**((sin(r) + 6.78974430415452)/(r - 0.00808244126136504)) + 7.58897670822754)*(-sin(theta + cos(theta - 0.0144023112886078) + 0.104063940713704) + sin(log(r + 0.390458429297535)) + ((-tanh(0.62*r)+1.01)*(pi/2))**(0.061275433230159*r + 0.00061275433230159)) - (tanh(2*r)**(27.030153904932*r**6.29794519183292) + 101.657005682759)/(r + (0.0053984397980632*r + 5.3984397980632e-5)*log(tanh(r)) - 10452.4889454854)*((.5*(1-tanh(.1*(r-100.01))))) + (0.00273233753019377**(6.1870938127573 - sin(theta + 0.0821478246163881)) + 2.79499001433555e-13 + (6.12323399573677e-17)/(2.19282503136287 - 6.28215572322383*r))*(0.999329299739067*r + 0.455361131407934)**(sin(sqrt(r + 9.07998593378172e-5)) + 11.4146131594409 + 0.00010001/(1.01005016708417 - cos(theta)))*((.5*(1-tanh(2e3*(r-10.01))))) + 0.0101001582000134*tanh(10.1589286596707*r + 11.9949097863706) + 0.861955660617334

    (33 x 33), (0.01 <= r <= 10, 0 <= θ <= 2π): interval([36.36755788914332, 36.819654105452344])
"""
