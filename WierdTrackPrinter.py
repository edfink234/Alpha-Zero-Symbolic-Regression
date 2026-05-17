from sympy.printing.pycode import PythonCodePrinter
class P(PythonCodePrinter):
 def _print_sech(self, expr):
    return f"(sp.sech({self._print(expr.args[0])}))"
    
printer = P();
x='((((((sqrt(arccos(x0)) - ((9.668070025255174 - x0) * ((((((sin(x0) - 0.8059223819349055) * -4.232056472702239) * sin((14.889176108571794 * x0))) - (((((207.91157245532216 * x0) * (((0.5101617140051913 * (x0 * x0)) - x0) + 0.5630588087853458)) - (-37.627823916623534 * sech((-7.949234728994861 * x0)))) * -4.084477959473518) * x0)) + (-0.4929031615465201 * arccos(sin(((15.886179981865006 * x0) - 0.8361985171697192))))) + (24.107142399070465 * tanh(arcsin(sech((-22.26578044605885 * x0)))))))) - ((acos(~(x0)) * ((4.04506267546967 + (x0 * x0)) * 3.7546140442275933)) * tanh((-50.66503734870332 * x0)))) + (-51.59762098962836 * tanh(sin(((7.332915989982738 * x0) * 1.1986776982768788))))) - (-14.02369942692714 * (((0.2736408649987147 + sqrt(x0)) * sech((((1.71698252603711 * x0) * 30.11221535026118) - 1.805539289001835))) + (-4.488064046756331 * (-4.009891147407424 * (x0 + 0.7933082675522551)))))) + ((arccos(x0) * (x0 * 1.1875808821778755)) * ((-2.377462127030465 * asin(x0)) + (sin(~((x0 * -63.849612789922574))) - ((((x0 * (x0 - 0.8361619531936503)) * 4.451774438265049) - -1.2356316798927098) * ((3.956129533408117 * (x0 + -0.7074253598210013)) * sin((0.6555597857975091 - ((acos(x0) - (~((x0 * acos(x0))) * arccos((x0 * x0)))) + (x0 * -78.81638698902094)))))))))) - tanh((0.2209976698388 * (sin(((0.44265897439025625 - (x0 * 4.026013591935868)) * (3.6574921525716544 * (((x0 * -1.7503815249823074) - acos(x0)) + 9.256013774179497)))) * sech(x0)))))'
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
