from sympy.printing.pycode import PythonCodePrinter
class P(PythonCodePrinter):
 def _print_sech(self, expr):
    return f"(sp.sech({self._print(expr.args[0])}))"
    
printer = P();
x='(((((((1.5707953724542136 + (0.8636892533604561 - x0)) * ((((x0 * 1.3107010086716608) * ((acos(sin((x0 + 0.9239770784315402))) * 1.790557006750339) - cos((x0 * 12.12832935999498)))) + ((0.06680046789847266 - (-0.1821786277553502 * x0)) * (3.3335225898850376 + sin((x0 * -9.689384313741993))))) - (x0 * ((x0 + -1.3710938332537321) * 17.5971371187159)))) * sin((sqrt(x0) + 0.15140890423163342))) + arccos(sin((9.485318039248959 * sech((x0 + 1.003308333346023)))))) - (-3.8163960662056424 + (~((x0 - acos(x0))) * (x0 * (asin(x0) - (x0 * 9.016764092183)))))) + (-12.853222070148547 * asin(sech((acos(x0) - x0))))) * 4)'
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
