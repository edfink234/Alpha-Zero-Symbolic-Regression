x=r'''
(AlphaZeroGeneralEnv) edwardfinkelstein@Edwards-MacBook-Pro alpha-zero-general % ./NeuralNetworks_VecSR
this->__unary_operators
cos exp sqrt sin asin ln tanh acos ~ 
this->__binary_operators
+ - * / ^ 
Board::__unary_operators_uset
~ acos ln sqrt exp asin sin tanh cos 
Board::__binary_operators_uset
* ^ - / + 
Board::__unary_operators.size() = 9
Board::__binary_operators.size() = 5
Board::__tokens.size() = 36
cos exp sqrt sin asin ln tanh acos ~ + - * / ^ w_k eta theta gamma epsilon beta_1 beta_2 d_ij value d_ij_nest velocity_k gradient_k g_t_k expt_grad_squared_k delta_w_t_k expt_weight_squared_k delta_w_t_k_ada_delta m_t_k v_t_k m_t_k_hat v_t_k_hat t 
pieces = t sqrt beta_2 expt_weight_squared_k eta t expt_weight_squared_k * ^ ^ / - , score = nan
pieces = delta_w_t_k_ada_delta v_t_k exp value eta theta g_t_k + + + - * , score = nan
pieces = t v_t_k g_t_k epsilon expt_grad_squared_k m_t_k_hat + / * * ^ , score = nan
pieces = w_k beta_2 g_t_k sin velocity_k m_t_k_hat delta_w_t_k + / / - ^ , score = nan
pieces = d_ij tanh beta_2 v_t_k beta_2 gradient_k epsilon / ^ - / + , score = nan
pieces = value m_t_k_hat d_ij expt_grad_squared_k ^ + ~ - m_t_k_hat ^ , score = nan
pieces = beta_2 d_ij / w_k m_t_k gamma d_ij ~ * - / / , score = nan
pieces = epsilon beta_2 velocity_k eta ln * ^ beta_2 * - , score = nan
pieces = t acos t cos gradient_k - ~ t tanh d_ij cos asin ^ / ^ , score = nan
pieces = velocity_k beta_1 delta_w_t_k asin * sin velocity_k t ln value g_t_k + + * * * , score = nan
pieces = expt_weight_squared_k tanh sin ~ d_ij_nest velocity_k exp v_t_k + beta_1 tanh tanh - ^ * , score = nan
pieces = m_t_k exp asin tanh d_ij eta d_ij expt_weight_squared_k gamma * / / + ^ , score = nan
pieces = delta_w_t_k_ada_delta d_ij_nest value * v_t_k_hat epsilon gamma expt_weight_squared_k - * * / - , score = nan
pieces = v_t_k_hat epsilon beta_1 gamma gradient_k m_t_k_hat / / ^ - + , score = nan
pieces = w_k m_t_k_hat d_ij_nest ~ d_ij_nest velocity_k beta_1 * ^ * + * , score = nan
pieces = gamma v_t_k value d_ij v_t_k cos + - * - , score = nan
pieces = v_t_k expt_grad_squared_k epsilon m_t_k ln gradient_k v_t_k_hat * - * ^ - , score = nan
pieces = w_k asin m_t_k d_ij theta * cos beta_2 / * + , score = nan
pieces = expt_weight_squared_k expt_grad_squared_k epsilon / delta_w_t_k * g_t_k - gradient_k d_ij m_t_k expt_grad_squared_k + - / * - , score = nan
pieces = eta d_ij v_t_k gradient_k delta_w_t_k_ada_delta v_t_k - * + ^ / , score = nan
pieces = delta_w_t_k_ada_delta ~ theta m_t_k_hat w_k gamma eta - - / ^ * , score = nan
pieces = velocity_k beta_1 exp * gamma asin expt_weight_squared_k delta_w_t_k expt_weight_squared_k v_t_k / * / + + , score = nan
pieces = gamma delta_w_t_k_ada_delta ^ sqrt value d_ij_nest asin theta ln velocity_k g_t_k * + / + * , score = nan
pieces = d_ij beta_2 tanh m_t_k_hat m_t_k m_t_k_hat sin - * + / , score = nan
pieces = gradient_k acos expt_weight_squared_k * epsilon v_t_k expt_grad_squared_k cos t ^ / - + , score = nan
Best score = 0.000784789, MSE = 1273.23
Best expression = (t * (velocity_k / (~(m_t_k_hat) / (beta_1 / tanh(epsilon)))))
Best expression (original format) = t velocity_k m_t_k_hat ~ beta_1 epsilon tanh / / / *
Best score = 0.00151375, MSE = 659.613
Best expression = (asin(tanh(~(theta))) * (epsilon / cos(asin((m_t_k - eta)))))
Best expression (original format) = theta ~ tanh asin epsilon m_t_k eta - asin cos / *
pieces = velocity_k v_t_k acos velocity_k w_k v_t_k_hat sin * ^ + * , score = nan
pieces = d_ij expt_grad_squared_k v_t_k expt_grad_squared_k cos - epsilon ln sin / / ^ , score = nan
pieces = value w_k - value velocity_k beta_1 acos beta_2 g_t_k + - - ^ * , score = nan
pieces = m_t_k cos sin beta_1 ^ acos epsilon v_t_k_hat theta ^ gamma * beta_1 expt_grad_squared_k delta_w_t_k_ada_delta ^ + - - - , score = nan
pieces = beta_1 expt_weight_squared_k m_t_k_hat sin + m_t_k d_ij_nest acos m_t_k velocity_k * - + ^ * , score = nan
pieces = w_k cos m_t_k_hat beta_1 expt_weight_squared_k beta_2 acos + + ^ / , score = nan
pieces = m_t_k m_t_k velocity_k d_ij / tanh t d_ij sqrt + / ^ * , score = nan
pieces = v_t_k_hat value ln exp m_t_k sin * ln + , score = nan
pieces = beta_1 sin g_t_k v_t_k_hat cos tanh velocity_k eta cos ^ + * * , score = nan
pieces = value cos epsilon value value v_t_k_hat delta_w_t_k_ada_delta * * ^ ^ * , score = nan
pieces = epsilon m_t_k beta_1 / gamma epsilon eta cos * ^ ^ - , score = nan
pieces = velocity_k m_t_k t asin + cos beta_1 tanh m_t_k_hat g_t_k / acos ^ / / , score = nan
pieces = value tanh exp velocity_k gradient_k d_ij v_t_k w_k + / - ^ / , score = nan
pieces = gamma eta epsilon gamma acos m_t_k_hat beta_1 * + - ^ ^ , score = nan
pieces = gradient_k t / expt_grad_squared_k t - v_t_k gamma velocity_k exp / * / / , score = nan
pieces = beta_2 m_t_k velocity_k gradient_k asin gradient_k * - / / , score = nan
pieces = eta velocity_k exp gamma velocity_k tanh epsilon beta_2 / * * ^ - , score = nan
pieces = d_ij acos d_ij_nest d_ij_nest m_t_k sin gamma theta + - / ^ + , score = nan
pieces = expt_weight_squared_k ln gamma sqrt exp + t beta_1 tanh ^ velocity_k gradient_k velocity_k velocity_k - * / ^ / , score = nan
pieces = beta_1 v_t_k_hat epsilon w_k sin - delta_w_t_k_ada_delta * / - , score = nan
pieces = t eta tanh acos velocity_k delta_w_t_k_ada_delta v_t_k_hat tanh + / * * , score = nan
pieces = velocity_k acos v_t_k beta_2 ln beta_2 m_t_k_hat sqrt - + + / , score = nan
pieces = epsilon tanh expt_weight_squared_k epsilon v_t_k / value v_t_k eta - / * + ^ , score = nan
pieces = velocity_k sqrt value m_t_k v_t_k value expt_grad_squared_k ^ ^ ^ * / , score = nan
pieces = g_t_k d_ij_nest ln v_t_k exp delta_w_t_k_ada_delta g_t_k d_ij ^ - / - / , score = nan
pieces = delta_w_t_k_ada_delta v_t_k expt_grad_squared_k beta_1 beta_1 / m_t_k_hat / + / - , score = nan
pieces = velocity_k ~ expt_grad_squared_k sqrt acos sqrt epsilon delta_w_t_k d_ij m_t_k_hat - * / - ^ , score = nan
pieces = v_t_k_hat d_ij_nest gradient_k + ln m_t_k_hat gradient_k delta_w_t_k_ada_delta * sqrt ^ - - , score = nan
pieces = w_k eta * ~ sqrt acos v_t_k_hat gradient_k sin sqrt epsilon beta_1 w_k / ^ / + + , score = nan
pieces = expt_weight_squared_k cos tanh asin ~ v_t_k sin / , score = nan
pieces = d_ij tanh gamma beta_1 m_t_k_hat ln - gradient_k acos t acos / ^ * - , score = nan
pieces = expt_weight_squared_k theta value v_t_k_hat / ~ d_ij expt_weight_squared_k ^ t d_ij + * / ^ / , score = nan
pieces = gamma velocity_k sqrt asin gamma / g_t_k gamma expt_weight_squared_k + v_t_k_hat theta - / / + ^ , score = nan
pieces = gamma expt_weight_squared_k t m_t_k gamma beta_2 + * / ^ - , score = nan
pieces = beta_2 expt_grad_squared_k exp * exp velocity_k beta_2 eta gradient_k asin ^ ^ + - , score = nan
pieces = delta_w_t_k_ada_delta tanh t velocity_k ^ tanh theta beta_1 m_t_k theta ^ - - * * , score = nan
pieces = v_t_k_hat ln expt_grad_squared_k t value / d_ij_nest delta_w_t_k w_k ^ * * + * , score = nan
pieces = d_ij ln v_t_k epsilon delta_w_t_k * m_t_k sqrt gamma acos / ^ * + , score = nan
pieces = epsilon sin cos w_k gamma v_t_k_hat * eta t m_t_k + - * / ^ , score = nan
pieces = d_ij beta_2 delta_w_t_k g_t_k v_t_k velocity_k - ^ * * + , score = nan
pieces = m_t_k delta_w_t_k_ada_delta theta velocity_k - asin ^ ^ v_t_k_hat velocity_k gradient_k ln expt_grad_squared_k d_ij_nest ^ / + / - , score = nan
pieces = beta_2 m_t_k eta ^ acos sqrt m_t_k_hat g_t_k ^ g_t_k beta_1 velocity_k ^ + / ^ / , score = nan
pieces = d_ij ln asin ~ velocity_k d_ij_nest d_ij_nest theta value * - ^ + - , score = nan
pieces = gamma tanh d_ij eta w_k - gamma cos asin / * - , score = nan
pieces = eta expt_grad_squared_k v_t_k beta_2 ^ asin sin - - , score = nan
pieces = expt_weight_squared_k v_t_k_hat m_t_k_hat expt_weight_squared_k v_t_k value + ^ * * * , score = nan
pieces = expt_grad_squared_k ln t d_ij g_t_k sqrt delta_w_t_k sqrt / ^ / / , score = nan
pieces = expt_grad_squared_k cos sqrt exp delta_w_t_k_ada_delta expt_weight_squared_k tanh cos expt_weight_squared_k m_t_k asin - * / / , score = nan
pieces = expt_grad_squared_k sqrt d_ij sin sqrt d_ij_nest eta m_t_k_hat delta_w_t_k_ada_delta * * / * - , score = nan
pieces = velocity_k w_k beta_2 * / asin t delta_w_t_k_ada_delta g_t_k w_k beta_2 - + * ^ - , score = nan
pieces = m_t_k velocity_k acos velocity_k acos exp gamma beta_2 delta_w_t_k + ^ / / / , score = nan
pieces = w_k delta_w_t_k theta exp sqrt cos * - , score = 0.00120744
pieces = value expt_weight_squared_k w_k value sin t beta_2 / / ^ / / , score = nan
pieces = velocity_k sqrt exp expt_weight_squared_k gradient_k gamma delta_w_t_k sin * - - * , score = nan
pieces = gradient_k ln expt_weight_squared_k theta gamma velocity_k ln * / ^ + , score = nan
pieces = v_t_k_hat g_t_k g_t_k expt_grad_squared_k delta_w_t_k m_t_k_hat + + ^ / + , score = nan
pieces = g_t_k g_t_k t - v_t_k_hat value / m_t_k_hat sin ln - ^ - , score = nan
pieces = beta_1 acos sqrt gamma * ~ delta_w_t_k_ada_delta w_k beta_1 w_k sin / * * ^ , score = nan
pieces = expt_weight_squared_k v_t_k / beta_1 theta beta_1 exp v_t_k epsilon ^ - / + + , score = nan
pieces = t tanh sin expt_grad_squared_k expt_weight_squared_k gamma ln d_ij v_t_k_hat ^ ^ + * + , score = nan
pieces = beta_2 eta d_ij gamma gradient_k epsilon * + * / ^ , score = nan
pieces = theta gradient_k d_ij m_t_k_hat value acos * + ^ * , score = nan
pieces = gamma velocity_k m_t_k_hat delta_w_t_k_ada_delta velocity_k d_ij_nest / - ^ + ^ , score = nan
pieces = d_ij_nest expt_weight_squared_k sqrt ~ ~ ^ v_t_k_hat / , score = nan
pieces = eta ln delta_w_t_k_ada_delta d_ij velocity_k d_ij beta_1 / / ^ - / , score = nan
pieces = epsilon m_t_k_hat acos - asin expt_weight_squared_k theta d_ij expt_weight_squared_k delta_w_t_k_ada_delta / + * ^ * , score = nan
pieces = d_ij eta value * w_k acos theta w_k asin / / * / , score = nan
pieces = g_t_k delta_w_t_k d_ij_nest d_ij beta_2 delta_w_t_k * * ^ ^ * , score = nan
pieces = gradient_k expt_weight_squared_k sqrt gradient_k beta_2 ^ theta expt_grad_squared_k expt_weight_squared_k * ^ / + + , score = nan
pieces = velocity_k v_t_k d_ij sin delta_w_t_k gradient_k exp / - ^ / , score = nan
pieces = expt_weight_squared_k t sqrt g_t_k d_ij_nest theta m_t_k * - ^ * * , score = nan
pieces = epsilon v_t_k exp * epsilon ^ m_t_k_hat expt_grad_squared_k sqrt ln beta_1 ^ + - , score = nan
pieces = epsilon m_t_k_hat g_t_k gamma acos beta_1 delta_w_t_k ^ * ^ + + , score = nan
Thread 1 done generating its initial population
(AlphaZeroGeneralEnv) edwardfinkelstein@Edwards-MacBook-Pro alpha-zero-general % 
'''.replace("\nBest expression", ", expression")
y= '\n'.join([i for i in x.split("\n") if "score = nan" not in i and "score =" in i])
print(y)
data = '''
-1.13363     1.66514     2.82955
1.26407    -2.18976     5.12067
0.135599     1.96631  -0.0352809
1.37152    -1.66242     4.00271
-0.529875    0.968843   -0.271913
-0.84093    -1.09009     2.77899
0.907643    -1.18095      1.8092
-1.85626   -0.196468     18.4846
2.30118     2.76557     16.9146
2.91768    -1.07635     49.2865
-1.08054   -0.307229     2.97926
-0.448591     1.85532 -0.00344939
-2.82556     0.24209     86.0865
-0.148662     1.43836   -0.400145
2.16814     2.79672     13.0199
1.46512    -2.48693     7.04215
-2.30058     1.51986     39.8235
2.23444   -0.847853     14.9786
1.31711     2.12575    0.858232
-1.96068    -2.51847     28.0054
'''.split("\n")

def Hemberg_2(x):
    return x[0]*x[0]*x[0]*(x[0]-1.0) + x[1]*(x[1]/2.0 - 1.0);


import numpy as np
data = np.array([[float(j) for j in i.split()] for i in data if i])
print(data)
print(len(data[:, 0]), len(data[:, 1]))
x, y = np.meshgrid(data[:, 0], data[:, 1])
print(x.shape, y.shape)
print(all([np.all(i == x[0]) for i in x]))
#print(*y, sep='\n',end="\n"*5)
#print(*x, sep='\n')

from matplotlib import cbook, cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
ls = LightSource(270, 45)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
z = Hemberg_2((x, y))
rgb = ls.shade(z, cmap=cm.gist_earth)
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
[1, 4, 3, 5, 3, 5]
[1, 4, 3, 5, 3, 5]
[(1, 1), (4, 4), (3, 3), (5, 5), (3, 3), (5, 5)]
def row_in_matrix(row, data):
    for vec in data:
        if all([np.isclose(i, j) for (i, j) in zip(vec, row)]):
            return True
    return False

for i, (x_i, y_i, z_i) in enumerate(zip(x.flatten(), y.flatten(), z.flatten())):
    temp=np.array((x_i, y_i, z_i))
    print(temp)
#    print(np.argwhere(data==temp))
    ax.scatter(x_i, y_i, z_i, c = 'g' if row_in_matrix(temp, data) else 'r')
#plt.savefig("plt.png",dpi=5*96)
plt.show()
print("\n"*10)

