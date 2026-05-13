from sympy.printing.pycode import PythonCodePrinter
class P(PythonCodePrinter):
 def _print_sech(self, expr):
    return f"(sp.sech({self._print(expr.args[0])}))"
    
printer = P();
x='((((((asin(x0) * ((x0 * x0) * (23.478360436530366 * x0))) - ((9.662800577200096 - x0) * ((((((1.1238631533429664 - x0) * ((3.074332962643813 + x0) * 1.1432244903209237)) * sin((14.901600210404958 * x0))) + (((((-212.67998405576904 * x0) * (((-0.5091520390764023 * (x0 * x0)) + x0) - 0.5725013770338305)) + (35.55016339107632 * sech((7.999904121671664 * x0)))) * 3.9999974287552513) * x0)) - (-0.3531331789655214 * arcsin(cos(((16.102622061399014 * x0) + 1.4187811019912715))))) - (-19.16135333884757 * tanh(acos(tanh((22.150899710010773 * x0)))))))) + (36.482441882194394 * tanh((36.3354176516382 * x0)))) + (-55.376537037152744 * tanh(sin(((8.69011576262174 * x0) - -0.055285078248569146))))) + (-9.393812523094956 * (-18.502958420648824 - (36.33893951064894 * x0)))) - x0)'
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
