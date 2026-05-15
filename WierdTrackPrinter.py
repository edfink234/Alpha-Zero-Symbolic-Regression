from sympy.printing.pycode import PythonCodePrinter
class P(PythonCodePrinter):
 def _print_sech(self, expr):
    return f"(sp.sech({self._print(expr.args[0])}))"
    
printer = P();
x='(((((sqrt(acos(x0)) - ((9.667076466723682 - x0) * ((((((sin(x0) + (x0 - 1.6915464226605978)) * -1.9196258890006548) * sin((14.907621418336204 * x0))) - (((((207.70527637650997 * x0) * (((x0 * (0.5107796690717792 * x0)) - x0) - -0.565868046800584)) + (37.116117679518446 * sech((-7.948117430149992 * x0)))) * -3.999277112264214) * x0)) + (-0.5007591123929714 * arccos(sin(((15.873367204951375 * x0) - 0.8372817569720107))))) - (-23.442886127029407 * tanh(arcsin(sech((-22.296377271943083 * x0)))))))) + ((acos(~(x0)) * ((1.995567636989864 + (x0 * x0)) * -6.791049327353052)) * tanh((-51.44268694664276 * x0)))) + (-50.02495479078701 * tanh(sin(((7.333794829929451 * x0) * 1.1985615132595946))))) - (-15.360586988806647 * (((0.31638004513398643 + sqrt(x0)) * sech((((x0 * 1.336917158457956) * 32.91378990478939) - 1.415011883125092))) - (-4.33106707994306 * ((x0 + x0) + (x0 + (x0 + 2.903573757592812))))))) + ((arccos(x0) * asin(x0)) * ((-1.6059720429153002 - x0) + (sin(~((x0 * -63.6936865867503))) + ((2 * 2.541849653393791) * ((sqrt(x0) - 0.7869335141924855) * cos((0.9832128003860027 * ((-0.10764545740464662 - x0) + (x0 * -80.55437697015459))))))))))'
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
