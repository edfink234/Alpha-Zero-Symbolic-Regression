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

r_float_vals = linspace(0.01, 100, N).tolist()
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
    -(sp.sin(r) - sp.cos(r)/r + 3.0*sp.sin(r)/r**2 + 6.0*sp.cos(r)/r**3 - 6.0*sp.sin(r)/r**4)*sp.sin(theta) - sp.sin(r)**3*sp.sin(theta)**3 + sp.sin(r)**2*sp.sin(theta)**2 + 2.0*sp.sin(r)*sp.sin(theta) - (-sp.sin(theta)*sp.cos(r) - sp.sin(r)*sp.sin(theta)/r - 2.0*sp.sin(theta)*sp.cos(r)/r**2 + 2.0*sp.sin(r)*sp.sin(theta)/r**3)/r - 2.0*sp.sin(theta)*sp.cos(r)/r - (sp.sin(r) - sp.cos(r)/r + sp.sin(r)/r**2)*sp.sin(theta)/r**2 + 2.0*sp.sin(r)*sp.sin(theta)/r**2
)

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
#print(f_vals_squared)

"""
Stats
=====

f = sin(r)*sin(theta):
    (1000 x 1000), (0.01 <= r <= 10, 0 <= θ <= 2π): interval([201360.4665727735, 201360.46821284745])


"""
