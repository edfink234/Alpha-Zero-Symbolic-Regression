from sympy.printing.pycode import PythonCodePrinter
class P(PythonCodePrinter):
 def _print_sech(self, expr):
    return f"(sp.sech({self._print(expr.args[0])}))"
    
printer = P();
x='(((((((((((((-16.802184072835658 * x0) - 1.0000069155282691) * ((~((((((47.93124981710377 * asin(x0)) + sqrt((0.6593191824101318 * sqrt(sqrt(x0))))) + (arccos(sin((1.1415827136519796 + x0))) * ((1.0236635326485066 - cos(x0)) * (x0 * 20.6364642471473)))) + (x0 * (sech(arcsin(x0)) * 95.03442711438181))) - 7.044963056079994)) + ((sqrt(x0) - 0.6259047246751295) * (((64.55664170785013 * (0.524677403721233 - x0)) + sin((x0 * 11.325957472323513))) * 0.5551159233280556))) - ((sin(x0) * -4.059355627344009) * (29.782056405831057 - (-3.3549442984785216 * asin(x0)))))) - (8.280806063085372 * (0.5519433042781647 - sqrt(x0)))) * (3.999909339109233 * (x0 - 4.000011206247267))) + (1.4396204346924448 * sin((x0 * 42.90928442524282)))) - (-19.78919937629448 * tanh(cos(((-6.682225398424411 * x0) * 2.0000261744689753))))) + (-1.0399778036060425 * cos((36.83770760176538 * x0)))) * arccos(x0)) - (-138.8430513621242 * acos(sech((0.8959354719798385 - acos(x0)))))) - 105.74309109197115) - ((acos(x0) * 0.21537631413252753) * (x0 + acos(sin((82.60893655713191 * (sin(tanh(x0)) - sech(tanh((x0 * x0)))))))))) - (((0.02844166616833611 - ((-0.7385090393637231 + x0) * (x0 * x0))) + (cos((-20.075519005599535 * x0)) * (1.2272210154449184 - (arccos(x0) + (x0 * x0))))) * (x0 * ((0.9264263544714448 - x0) * (15.114055459942811 * ~(sin((x0 * 119.91078812058495))))))))'
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
