from sympy.printing.pycode import PythonCodePrinter
class P(PythonCodePrinter):
 def _print_sech(self, expr):
    return f"(sp.sech({self._print(expr.args[0])}))"
    
printer = P();
x='(((((sqrt(arccos(x0)) + ((9.667663193613832 - x0) * ((((((-0.8030588555342483 + sin(x0)) * 4.20656796419571) * sin((14.906510133485185 * x0))) + (((((-208.00136666356212 * x0) * (((0.5101104451656328 * (x0 * x0)) - x0) - -0.563518885104333)) - (37.179980144744185 * sech((-7.948117430149992 * x0)))) * 4.039617583794263) * x0)) - (-0.5127238185134793 * arccos(sin(((15.873367204951377 * x0) - 0.8342633799173187))))) - (23.790831494726362 * tanh(arcsin(sech((-22.296377271943083 * x0)))))))) + ((acos(~(x0)) * ((3.338052952086854 + (x0 * x0)) * -4.431373749611782)) * tanh((-51.44268694664276 * x0)))) - (51.392889773280906 * tanh(sin(((7.333794829929451 * x0) * 1.1986483955395455))))) + (14.03092020672828 * (((0.27976986731361486 + sqrt(x0)) * sech((((1.449072284299925 * x0) * 32.91254998325287) - 1.5600756757415084))) - (4 * (-4.489081140155492 * (x0 - -0.7803543388371276)))))) + ((arccos(x0) * arcsin(x0)) * ((sin(x0) - (x0 + x0)) + (sin(~((x0 * -63.6936865867503))) + ((((sqrt(x0) - x0) * -22.260418713780687) + 9.500890845394574) * ((sqrt(x0) - 0.829171852750111) * sin((0.5896569246601726 - ((((((0.9869486052927025 - x0) * (14.25084643949581 + (x0 * -37.044730477958446))) * (x0 - 0.9866567554615902)) - 3.043475698311831) * x0) + (x0 * -78.55359990246922))))))))))'
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
