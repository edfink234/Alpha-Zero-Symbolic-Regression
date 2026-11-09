def isFloat(i):
    try:
        float(i)
        return True
    except ValueError:
        return False
        
x='0.0608264056165289 0.0002628145221960258 x0 -9.02879787696337e-05 * + x2 x0 + sech * -0.16268973809482873 + * -0.48850688258587005 x0 0.00011529516196540298 + * sech ~ -0.07132348099435075 1.55879925326467 x2 + * 0.9849622000105944 + + * 0.5821257180149014 x3 + sin x3 1.363188908540914 + x2 x2 * * + 1.6895362373369902 x0 ~ * x1 cos * + x3 x2 cos + x3 1.6337652281212212 * x3 x3 * * * 0.35673911286308896 x3 + sin cos + + sin * x3 x3 * tanh tanh x2 0.6333913456855232 * sech * sech cos 0.9999688722792472 x0 * sin sech tanh sech 0.07238062650379498 x0 cos cos * x0 sech -2.9202637154554423 x0 cos + * * 0.1491743018966428 x0 cos sech * -0.01752214962505137 1.198273736126695 x2 + cos * + + + + + x0 0.48681018200008286 * sech sin ~ sin -0.06463448669724292 * + -0.018960983186408223 0.9475789506415314 0.0377009684815874 x2 + * sin * + -1.443514987378317 x2 x2 + tanh sin 0.03680104049380962 * tanh ~ + +'

x = x.split()
print(len(x))
print(sum(isFloat(i) for i in x))
floats = [i for i in x if isFloat(i)]
print([(i, j) for (i,j) in enumerate(floats)])


