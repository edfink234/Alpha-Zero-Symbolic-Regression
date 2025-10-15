from interval import interval
from interval.imath import *
from interval.fpu import power_rn
from sympy import Function, symbols
import sympy as sp
from numpy import linspace
from itertools import product
import math

r_float_vals = linspace(0.01, 10, 3).tolist()
theta_float_vals = linspace(0, 2*math.pi, 3).tolist()

r_vals = [interval[i-0.00000000000001, i+0.00000000000001] for i in r_float_vals] #MARK: Changing the starting r to 1e-2 instead of 1e-4 solves the upper-bound blowup issue!
theta_vals = [interval[i-0.00000000000001, i+0.00000000000001] for i in theta_float_vals]

SH = symbols('\\text{SwiftHohenberg} r theta mu nu')
r, theta = symbols('r theta')
mu, nu = 1, 1

f_val_func = lambda r_val, theta_val: -1.00001716621136*(0.124998927125603*theta_val - sin(r_val)*sin(theta_val))**2*(0.125*theta_val - 1.00000858306885*sin(r_val)*sin(theta_val)) + 1.00001716621136*(0.124998927125603*theta_val - sin(r_val)*sin(theta_val))**2 - (-1.00000858306885*sin(r_val) + 1.00000858306885*cos(r_val)/r_val - 3.00002574920654*sin(r_val)/r_val**2 - 6.00005149841309*cos(r_val)/r_val**3 + 6.00005149841309*sin(r_val)/r_val**4)*sin(theta_val) - 2.0000171661377*sin(r_val)*sin(theta_val) - (1.00000858306885*sin(theta_val)*cos(r_val) + 1.00000858306885*sin(r_val)*sin(theta_val)/r_val + 2.0000171661377*sin(theta_val)*cos(r_val)/r_val**2 - 2.0000171661377*sin(r_val)*sin(theta_val)/r_val**3)/r_val + 2.0000171661377*sin(theta_val)*cos(r_val)/r_val - 1.00000858306885*(-sin(r_val) + cos(r_val)/r_val - sin(r_val)/r_val**2)*sin(theta_val)/r_val**2 - 2.0000171661377*sin(r_val)*sin(theta_val)/r_val**2

for (r_val, theta_val) in product(r_float_vals, theta_float_vals):
    print(r_val, theta_val, f_val_func(r_val, theta_val))

exit();

f_vals = []
max_disp = 0
idx = 0
print(r_vals[0], r_vals[0][0][0], r_vals[0][0][1])
for (r_val, theta_val) in product(r_vals, theta_vals):
    f_val = f_val_func(r_val, theta_val)
    #TODO: round up at each computation, store that list in f_vals_up, then do the same rounding down at each computation and storing that list in f_vals_down and then combine the intervals
    f_vals.append(f_val)

print(f"\nlen(f_vals) = {len(f_vals)}")
f_vals_squared = [i**2 for i in f_vals]
total = interval[0,0]
for i, f_val_squared in enumerate(f_vals_squared):
    total += f_val_squared
print(f"total = {total}")

#squared-norm error = 135.36465516771236
#total = interval([135.36443377482726, 135.36487669363538])
#https://pyinterval.readthedocs.io/en/latest/api.html#interval-imath-mathematical-functions-for-intervals

