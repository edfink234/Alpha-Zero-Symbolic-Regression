after='''before: ^ * 1 + tanh x x 0 
after: 1 

before: ^ 0 * tanh cos x 1 
after: nan 

before: sin arcsin ^ 0 * y y 
after: nan 

before: ^ sin arcsin ^ x * y y 1 
after: ^ x * y y 

before: ^ + x - 0 + x + x x 1 
after: + x ~ + x + x x 

before: ^ 1 + * x x / sin arcsin / ~ * y y x sin arcsin / ~ * y y x 
after: 1 

before: + ^ 1 sin arcsin ^ x * y y 0 
after: 1 

before: cos sin arcsin ^ 0 * y y 
after: nan 

before: cos ^ 0 * tanh cos x 1 
after: nan 

before: cos ~ ^ 0 * tanh cos x 1 
after: nan 

before: sin cos ~ arcsin ^ 0 * y y 
after: sin cos arcsin ^ 0 * y y 

before: sin sin arcsin ^ 0 * y y 
after: nan 

before: sin ^ sin sin arcsin ^ 0 * y y 1 
after: nan 

before: tanh sin sin arcsin ^ 0 * y y 
after: nan 

before: tanh ^ sin sin arcsin ^ 0 * y y 1 
after: nan 

before: sech tanh ^ sin sin arcsin ^ 0 * y y 1 
after: nan 

before: sech sin sin arcsin ^ 0 * y y 
after: nan 

before: tanh / tanh cos x 0 
after: nan 

before: tanh / sech cos + z x 0 
after: nan 

before: tanh / ~ sech cos + z x 0 
after: nan 

before: tanh / ~ tanh cos x 0 
after: nan 

before: tanh ~ inf 
after: -1 

before: tanh - 0 inf 
after: -1 

before: sech / ~ sech cos + z x 0 
after: nan 

before: sech / ~ tanh cos x sin + 0 0 
after: nan 

before: sech ~ / ~ sech cos + z x 0 
after: nan 

before: sech ~ / ~ tanh cos x sin + 0 0 
after: nan 

before: sech + ~ * 0 tanh tanh x x 
after: sech x 

before: / - cos x cos x 0 
after: nan 

before: / - sech cos x sech cos x 0 
after: nan 

before: / - + z nan - nan x 0 
after: nan 

before: tanh / ~ - x nan 0 
after: nan 

before: / sech ~ / ~ tanh cos x sin + 0 0 0 
after: nan 

before: ~ / sech ~ / ~ tanh cos x sin + 0 0 0 
after: nan 

before: ~ / tanh ~ / ~ tanh cos x sin + 0 0 0 
after: nan 

before: exp / tanh ~ / ~ tanh cos x sin + 0 0 0 
after: nan 

before: exp / sech ~ / ~ tanh cos x sin + 0 0 0 
after: nan 

before: exp * tanh ~ / ~ tanh cos x sin + 0 0 0 
after: 1 

before: exp * sech ~ / ~ tanh cos x sin + 0 0 0 
after: 1 

before: exp * sech ~ / ~ tanh cos x sin / 0 0 0 
after: 1 

before: cos + tanh ~ / ~ tanh cos x sin / x 0 0 
after: cos tanh ~ / ~ tanh cos x nan 

before: tanh / sin ~ / ~ tanh cos x sin ^ 0 cos x 0 
after: nan 

before: exp / sech ~ / ~ tanh cos x sin ^ 0 + x x 0 
after: nan 

before: + nan cos sin ^ x tanh 2 
after: nan 

before: - cos sin ^ x tanh 2 nan 
after: nan 

before: - nan cos sin ^ x tanh 2 
after: nan 

before: + cos sin ^ x tanh 2 nan 
after: nan 

before: * nan cos sin ^ x tanh 2 
after: nan 

before: * cos sin ^ x tanh 2 nan 
after: nan 

before: * nan tanh sech + x tanh - 2 ^ x 2 
after: nan 

before: * sin asin / x tanh sech acos arccos + apple 2 nan 
after: nan 

before: / nan asin acos / x tanh 2 
after: nan 

before: / acos tanh - y tanh 2 nan 
after: nan 
'''
before='''before: ^ * 1 + tanh x x 0 
after: 1 

before: ^ 0 * tanh cos x 1 
after: nan 

before: sin arcsin ^ 0 * y y 
after: nan 

before: ^ sin arcsin ^ x * y y 1 
after: ^ x * y y 

before: ^ + x - 0 + x + x x 1 
after: + x ~ + x + x x 

before: ^ 1 + * x x / sin arcsin / ~ * y y x sin arcsin / ~ * y y x 
after: 1 

before: + ^ 1 sin arcsin ^ x * y y 0 
after: 1 

before: cos sin arcsin ^ 0 * y y 
after: nan 

before: cos ^ 0 * tanh cos x 1 
after: nan 

before: cos ~ ^ 0 * tanh cos x 1 
after: nan 

before: sin cos ~ arcsin ^ 0 * y y 
after: sin cos arcsin ^ 0 * y y 

before: sin sin arcsin ^ 0 * y y 
after: nan 

before: sin ^ sin sin arcsin ^ 0 * y y 1 
after: nan 

before: tanh sin sin arcsin ^ 0 * y y 
after: nan 

before: tanh ^ sin sin arcsin ^ 0 * y y 1 
after: nan 

before: sech tanh ^ sin sin arcsin ^ 0 * y y 1 
after: nan 

before: sech sin sin arcsin ^ 0 * y y 
after: nan 

before: tanh / tanh cos x 0 
after: nan 

before: tanh / sech cos + z x 0 
after: nan 

before: tanh / ~ sech cos + z x 0 
after: nan 

before: tanh / ~ tanh cos x 0 
after: nan 

before: tanh ~ inf 
after: -1 

before: tanh - 0 inf 
after: -1 

before: sech / ~ sech cos + z x 0 
after: nan 

before: sech / ~ tanh cos x sin + 0 0 
after: nan 

before: sech ~ / ~ sech cos + z x 0 
after: nan 

before: sech ~ / ~ tanh cos x sin + 0 0 
after: nan 

before: sech + ~ * 0 tanh tanh x x 
after: sech x 

before: / - cos x cos x 0 
after: nan 

before: / - sech cos x sech cos x 0 
after: nan 

before: / - + z nan - nan x 0 
after: nan 

before: tanh / ~ - x nan 0 
after: nan 

before: / sech ~ / ~ tanh cos x sin + 0 0 0 
after: nan 

before: ~ / sech ~ / ~ tanh cos x sin + 0 0 0 
after: nan 

before: ~ / tanh ~ / ~ tanh cos x sin + 0 0 0 
after: nan 

before: exp / tanh ~ / ~ tanh cos x sin + 0 0 0 
after: nan 

before: exp / sech ~ / ~ tanh cos x sin + 0 0 0 
after: nan 

before: exp * tanh ~ / ~ tanh cos x sin + 0 0 0 
after: 1 

before: exp * sech ~ / ~ tanh cos x sin + 0 0 0 
after: 1 

before: exp * sech ~ / ~ tanh cos x sin / 0 0 0 
after: 1 

before: cos + tanh ~ / ~ tanh cos x sin / x 0 0 
after: cos tanh ~ / ~ tanh cos x nan 

before: tanh / sin ~ / ~ tanh cos x sin ^ 0 cos x 0 
after: nan 

before: exp / sech ~ / ~ tanh cos x sin ^ 0 + x x 0 
after: nan 

before: + nan cos sin ^ x tanh 2 
after: nan 

before: - cos sin ^ x tanh 2 nan 
after: nan 

before: - nan cos sin ^ x tanh 2 
after: nan 

before: + cos sin ^ x tanh 2 nan 
after: nan 

before: * nan cos sin ^ x tanh 2 
after: nan 

before: * cos sin ^ x tanh 2 nan 
after: nan 

before: * nan tanh sech + x tanh - 2 ^ x 2 
after: nan 

before: * sin asin / x tanh sech acos arccos + apple 2 nan 
after: nan 

before: / nan asin acos / x tanh 2 
after: nan 

before: / acos tanh - y tanh 2 nan 
after: nan 
'''
print(before==after)
