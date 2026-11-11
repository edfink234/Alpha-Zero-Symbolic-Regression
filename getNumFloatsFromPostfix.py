def isFloat(i):
    try:
        float(i)
        return True
    except ValueError:
        return False
        
x='0.0608264056165289 0.0002628169895706945 x0 -9.028573929308642e-05 * + x2 x0 + sech * -0.16268973809482873 + * -0.4885068825858701 x0 0.00011529516196540298 + * sech ~ -0.07132348099435075 1.55879925326467 x2 + * 0.9849622000105944 + + * 0.5821257180149014 x3 + sin x3 1.363188908540914 + x2 x2 * * + 1.6895362373369902 x0 ~ * x1 cos * + x3 x2 cos + x3 1.6337652281212212 * x3 x3 * * * 0.35673911286308896 x3 + sin cos + + sin * x3 x3 * tanh tanh x2 0.5983194305337294 * sech * sech cos 0.9999688722792472 x0 * sin sech tanh sech 0.07238062650379498 x0 cos cos * x0 sech -2.9202637154554423 x0 cos + * * 0.1491743018966428 x0 cos sech * 0.0010999876962990462 1.0548565061677853 x2 + cos * + + + + + x0 0.4868101819999901 * sech sin ~ sin -0.06463448669704745 * + -0.07745324204472348 x2 tanh ~ -0.07839156782201998 * + -0.48041226486321326 x2 x3 + * sin * + -1.427909680918838 0.0131555340890582 x2 + tanh tanh + x2 tanh ~ -1.124118788113707 * tanh ~ + +'

x = x.split()
print(len(x))
print(sum(isFloat(i) for i in x))
floats = [i for i in x if isFloat(i)]
print([(i, j) for (i,j) in enumerate(floats)])


