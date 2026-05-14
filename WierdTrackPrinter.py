from sympy.printing.pycode import PythonCodePrinter
class P(PythonCodePrinter):
 def _print_sech(self, expr):
    return f"(sp.sech({self._print(expr.args[0])}))"
    
printer = P();
x='((((((arccos(x0) * ((x0 * x0) * (-28.634358501022977 * x0))) - ((9.66280879483071 - x0) * ((((((0.8552973869520635 - x0) * 3.568722346616963) * sin((14.90894268961133 * x0))) - (((((-213.86221612749122 * x0) * (((-0.5068565405297647 * (x0 * x0)) + x0) - 0.5829025447792208)) - (-27.641995509517496 * sech((7.952617048054308 * x0)))) * -4.000013222386744) * x0)) - (0.5047715716106072 * arccos(sin((((16.31919008089338 * x0) - 0.9148309250727087) * 0.9747054194493853))))) - (-15.292852392239427 * tanh(arcsin(sech((22.3214968222805 * x0)))))))) + (25.33078500530147 * tanh((37.38613191012289 * x0)))) - (50.58351873102745 * tanh(sin(((8.598311761938547 * x0) + 0.09908393896496029))))) - (-7.29248371624541 * (17.711903096309705 + (66.14426391127957 * x0)))) + x0)'
x=x.replace("^","**").replace("~", "-").replace("x0","s");
from sympy import *;
import sympy as sp;
y = x.replace("arccos","acos").replace("arcsin","asin");
s, z = sp.symbols("s z");
from sympy.printing.pycode import pycode;
print('\n',x:=printer.doprint(eval(y)).replace('math','sp'), end = "");
print(" if fitPlotFunc else lambda s: ", end = "");
print(x.replace("sp","np"));
round_floats = lambda expr, ndigits: expr.xreplace({f: sp.Float(round(float(f), ndigits)) for f in expr.atoms(sp.Float)});
func_sym_r = round_floats(eval(y), 2);
latex_str = sp.multiline_latex(z, (sp.expand(func_sym_r)), 2);
print(latex_str.replace(r'\left', '').replace(r'\right', ''))
