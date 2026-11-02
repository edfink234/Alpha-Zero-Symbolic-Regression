def isFloat(i):
    try:
        float(i)
        return True
    except ValueError:
        return False
        
x='0.06082663513976989 0.00048380024656259336 x0 -0.00014905998317485088 * + x2 x0 + sech * -0.16269038311932038 + * -0.4985768799034683 x0 0.000190799542958923 + * sech ~ -0.07132270811907154 1.5560518585745082 x2 + * 0.9850219439343351 + + * 0.5791936079149366 x3 + sin x3 1.363180037464436 + x2 x2 * * + 1.6891680024646574 x0 ~ * x1 cos * + x3 x2 cos + x3 1.6337353102329213 * x3 x3 * * * 0.34436942533039044 x3 + sin cos + + sin * x3 x3 * tanh tanh x2 0.6144926805249477 * sech * sech cos x0 0.9999675025864335 * sin sech tanh sech 0.06638486816337649 x0 cos cos * x0 sech -2.7865965051828594 x0 cos + * * 0.14920512217106127 x0 cos sech * -0.0010781170930676113 x0 cos * -1.474917681913715 + + + + + + x0 sech tanh ~ sin -0.050062723381884465 * +'

x = x.split()
print(len(x))
print(sum(isFloat(i) for i in x))
floats = [i for i in x if isFloat(i)]
print([(i, j) for (i,j) in enumerate(floats)])


