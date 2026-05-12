from sympy.printing.pycode import PythonCodePrinter
class P(PythonCodePrinter):
 def _print_sech(self, expr):
    return f"(sp.sech({self._print(expr.args[0])}))"
    
printer = P();
x='((arccos(arccos(sech((cos(x0) - 0.96402913805394)))) + ((2.0001015095955106 * asin(sqrt(sech(tanh(x0))))) + x0)) * ((((((x0 + 10.495159233974029) * ((((x0 + (x0 + x0)) * (1.3973652876040572 * (asin(cos(x0)) - 0.09755302806371938))) + (((x0 + -0.1354942222053541) - (x0 * 0.3954018528163919)) * (sin((9.68908970669284 * x0)) - (x0 + 0.6568369105700052)))) - (((-0.2660407078087603 * acos(sin((x0 + 0.9210776194541305)))) - sech((-12.293279067565782 * x0))) * asin(x0)))) * tanh(sqrt(x0))) - ((1.3044665510354259 * (-0.20369443374991217 - asin(x0))) * acos(x0))) - (((x0 * (x0 * x0)) - (-8.55371112939692 * (asin(x0) + 0.581674982859202))) * asin(sech((x0 - acos(x0)))))) + acos(sin(((x0 * 1.7737667816101501) - (0.7891489053016416 - (x0 + (x0 - 2))))))))'
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
