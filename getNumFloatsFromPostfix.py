def isFloat(i):
    try:
        float(i)
        return True
    except ValueError:
        return False
        
x='0.0608264056165289 0.00026321159738668055 x0 -9.05522134060653e-05 * + x2 x0 + sech * -0.16268973809482873 + * -0.4885068824402988 x0 0.00011523685136330291 + * sech ~ -0.07132348099435075 1.5587992536025452 x2 + * 0.9849622002097792 + + * 0.582125717046152 x3 + sin x3 1.363188908540914 + x2 x2 * * + 1.6895362373369902 x0 ~ * x1 cos * + x3 x2 cos + x3 1.6337652281212212 * x3 x3 * * * 0.35673913711592004 x3 + sin cos + + sin * x3 x3 * tanh tanh x2 0.6641573767927156 * sech * sech cos 0.9999688724923054 x0 * sin sech tanh sech 0.07238062650367569 x0 cos cos * x0 sech -2.9202637154489945 x0 cos + * * 0.14917430193292358 x0 cos sech * -0.001343397064535385 0.04667196784578353 x2 + cos * + + + + + x0 0.48681018216299615 * sech sin ~ sin -0.06463448863299305 * + 0.003764454250415206 0.4101125061044522 1.1328421014560988 x2 + * sin * -1.4740878999734253 + +'

x = x.split()
print(len(x))
print(sum(isFloat(i) for i in x))
floats = [i for i in x if isFloat(i)]
print([(i, j) for (i,j) in enumerate(floats)])


