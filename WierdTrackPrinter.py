from sympy.printing.pycode import PythonCodePrinter
class P(PythonCodePrinter):
 def _print_sech(self, expr):
    return f"(sp.sech({self._print(expr.args[0])}))"
    
printer = P();
x='(((((((((((((((-16.777616570512944 * x0) - 0.8306189265029301) * ((~((((((38.40359043830019 * arcsin(x0)) + sqrt(sqrt(sqrt(x0)))) - (arccos(sin((x0 + 1.1415951307864396))) * ((1.0223661471440606 - cos(x0)) * (x0 * -21.13890720974086)))) + (x0 * (sech(arcsin(x0)) * 101.28256646428251))) - 8.259465504453356)) + ((sqrt(x0) - 0.6249489763711806) * (((64.54590489933697 * (0.5683023211239526 - x0)) + sin((x0 * 11.330638763373788))) * 0.5635392600438554))) + ((sin(x0) * 4.0584308964672235) * (26.77386670959251 + (4.217961269161968 * sqrt(x0)))))) + (6.22514055685562 * (-0.595915710381489 + sqrt(x0)))) * (4.0000437572893865 * (x0 + -3.999897110430852))) - (-1.4568317419473362 * sin((x0 * 42.922593655687336)))) + (19.83996633272047 * tanh(cos(((-6.683555321462922 * x0) * 2.0000119786540065))))) + (-1.0503963064944712 * cos((-36.828669687140874 * x0)))) * acos(x0)) + (135.88933938962452 * acos(sech((0.8961900453685253 - acos(x0)))))) + -103.6408388759151) + ((arccos(x0) * 0.22470061145876485) * ~(arccos(sin((82.5216355471646 * (sin(tanh(x0)) - sech(tanh((x0 * x0)))))))))) - (((0.044851480438137546 - ((-0.7078732093049552 + x0) * (x0 * x0))) + (cos((-19.867254923459424 * x0)) * (-0.2880773968715205 - (-0.2706328594194209 * sqrt(x0))))) * (x0 * ((x0 - 0.9279324129934339) * (-15.965336642805138 * ~(sin((x0 * 119.8461111944218)))))))) + (cos((-3.238299931836995 * (sech((1.6562707812807105 * sin(x0))) * (3.9998470155400883 * (3.9999824038768184 * (1.0124551940002344 + (1.0594336770154307 * (4.000044663534493 * x0)))))))) * arcsin((0.8063394511057888 - sech(((x0 - 0.948739678138467) + (x0 * (6.408877170737145 + (x0 * -9.514769954551296))))))))) - ((x0 + -0.5408073272003451) * (4.412171189196541 * (tanh(x0) * ((-0.46525910568678763 - ((2.0054902337430605 - x0) * (tanh((x0 + x0)) * (arcsin(x0) * (x0 - 1.27721603967268))))) * asin(cos(((arcsin(x0) + 0.9640297593509776) * (-951.2504369851667 * (2 * acos(x0)))))))))))'
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
