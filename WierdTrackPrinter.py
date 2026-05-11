from sympy.printing.pycode import PythonCodePrinter
class P(PythonCodePrinter):
 def _print_sech(self, expr):
    return f"(sp.sech({self._print(expr.args[0])}))"
    
printer = P();
x='((sqrt(asin(sech((cos(x0) - 0.96402913805394)))) + 3.1451397140527875) * ((((((-0.09205885008491689 - x0) + (arcsin(x0) * -1.5903209615834777)) * arccos((x0 - 1))) + (((x0 + 10.898156328158121) * ((((x0 + x0) * (3.3459264020165214 - (2.342339825737655 * x0))) - ((0.12925440287365023 - (0.5564647909380217 * x0)) * (sin((9.68908970669284 * x0)) - 0.6778218207304143))) - (asin(x0) * ((-0.24031543258216334 * acos(sin(((-0.6495132911379564 + x0) + 1.5707963267948966)))) - sech((12.1279362659261 * x0)))))) * tanh((sqrt(x0) + 0.118754696358058)))) - (((((x0 * -0.48946102360867144) - (arcsin(x0) + 0.7284167523309266)) + ~(asin(x0))) * -4.491757886181279) * asin(sech((x0 - acos(x0)))))) + acos(sin(((x0 - 0.7091178650336425) * 4.052075959170716)))))'
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
