from sympy.printing.pycode import PythonCodePrinter
class P(PythonCodePrinter):
 def _print_sech(self, expr):
    return f"(sp.sech({self._print(expr.args[0])}))"
    
printer = P();
x='((((((sqrt(arccos(x0)) + ((9.667784932868097 - x0) * ((((((0.8033433272512843 - sin(x0)) * -4.222103159161761) * sin((14.903274669312749 * x0))) + (((((-208.00249875861766 * x0) * (((0.5101617140051913 * (x0 * x0)) - x0) + 0.5630588813492303)) - (37.486436913611875 * sech((-7.948480508366865 * x0)))) * 4.065809736604756) * x0)) + (0.5061334386832598 * arccos(sin(((15.87647315073732 * x0) - 0.8345679621947273))))) - (23.992675707936076 * tanh(arcsin(sech((-22.295331914142334 * x0)))))))) - ((acos(~(x0)) * ((-4.2003934624384005 - (x0 * x0)) * -3.6433839117767715)) * tanh((-51.4329730645006 * x0)))) + (-51.498152963710936 * tanh(sin(((7.333794829929451 * x0) * 1.1986459487174077))))) + (14.023919771257086 * (((0.376431355430979 - (-2.48160266261318 * x0)) * sech((((1.6022586983991554 * x0) * 32.81945739492898) - 1.8059086816205898))) - (-4.488888900594388 * (3.999991944334897 * (x0 + 0.7814762739553973)))))) + ((arccos(x0) * arcsin(x0)) * (~((x0 * 2.543641706570506)) + (sin(~((x0 * -63.79280213646689))) + (((((x0 * x0) - x0) * -3.2441782103369725) + -1.3074589129736318) * ((4.27476342586453 * (x0 + -0.664245936921775)) * sin((0.7205520068071117 - ((acos(x0) - (~(sqrt(sin(x0))) * (-0.20728752286519678 + arccos(x0)))) + (x0 * -78.59955176646633)))))))))) + arccos((0.20827560795102262 * (sin(((0.43349686111245916 - (x0 * 4.000922086089561)) * (3.6005065350746968 * (((x0 * -1.5855018411557833) - acos(x0)) + 9.25655184291956)))) * sech(x0)))))'
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
