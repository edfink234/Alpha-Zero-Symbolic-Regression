new = '''before: * * x x 1 
after: * * x x 1 

before: * + sin x x 1 
after: * + sin x x 1 

before: + ~ tanh * x 1 * 1 1 
after: + ~ tanh x 1 

before: + - sin x sin x sin x 
after: + - sin x sin x sin x 

before: / x 1 
after: x 

before: / * x x 1 
after: / * x x 1 

before: / * x cos x 1 
after: / * x cos x 1 

before: / 0 * x x 
after: / 0 * x x 

before: / 0 * x cos x 
after: / 0 * x cos x 

before: / 0 * sin x sech x 
after: / 0 * sin x sech x 

before: / 1 * x x 
after: / 1 * x x 

before: / 1 cos x 
after: / 1 cos x 

before: / 1 * cos x sin x 
after: / 1 * cos x sin x 

before: + x sin ~ ~ x 
after: + x sin x 

before: - tanh ~ ~ x x 
after: - tanh x x 

before: + ^ 0 x x 
after: x 

before: - ^ 0 x x 
after: ^ x x 

before: - cos x ^ 0 x 
after: - cos x 0 

before: + x ^ x 0 
after: + x 1 

before: - ^ x 0 x 
after: - 1 x 

before: - cos x ^ x 0 
after: - cos x 1 

before: + x ^ 1 x 
after: + x 1 

before: - ^ 1 x x 
after: - 1 x 

before: - cos x ^ 1 x 
after: - cos x 1 

before: + x ^ x 1 
after: + x x 

before: - ^ x 1 x 
after: 0 

before: - cos x ^ x 1 
after: - cos x x 

before: ln * 1 exp x 
after: ln * 1 exp x 

before: - x ln * 1 exp x 
after: - x ln * 1 exp x 

before: cos - x ln * 1 exp x 
after: cos - x ln * 1 exp x 

before: ln exp * y y 
after: * y y 

before: * exp * x x ln y 
after: * exp * x x ln y 

before: + exp exp - y y sin x 
after: + 2.718282 sin x 

before: / sin * x x y 
after: / sin * x x y 

before: / sin cos x y 
after: / sin cos x y 

before: cos sqrt - x x 
after: 1 

before: sin tanh sqrt - * x x * x x 
after: sin tanh sqrt - * x x * x x 

before: sqrt sqrt - * x cos x * x cos x 
after: sqrt sqrt - * x cos x * x cos x 

before: cos arcsin - x x 
after: 1 

before: sin tanh asin - ^ x x ^ x x 
after: sin tanh asin - ^ x x ^ x x 

before: asin arcsin - * x sin x * x sin x 
after: asin arcsin - * x sin x * x sin x 

before: exp acos - tanh x tanh x 
after: exp acos - tanh x tanh x 

before: sech sech arccos - / x x / x x 
after: 0.925521 

before: acos arccos - - x sech x - x sech x 
after: acos arccos - - x sech x - x sech x 

before: acos tanh - * x exp x * x exp x 
after: acos tanh - * x exp x * x exp x 

before: asin sech - * x exp x * x exp x 
after: asin sech - * x exp x * x exp x 

before: acos sech - - x sech x - x sech x 
after: acos sech - - x sech x - x sech x 

before: + ~ * 0 tanh tanh x x 
after: + ~ * 0 tanh tanh x x 

before: * * y 1 * x2 1 
after: * y x2 

before: * + 0 y + 0 x2 
after: * y x2 

before: * + x 0 + 0 y 
after: * x y 

before: / + x3 0 + 0 y 
after: / x3 y 

before: / / 0 x3 / 1 y 
after: / 0 / 1 y 

before: / / 0 w / y 1 
after: 0 

before: cos acos * y y 
after: * y y 

before: ln exp * cos arccos y y 
after: * y y 

before: arccos cos * y y 
after: * y y 

before: ln exp * arccos cos y y 
after: * y y 

before: sin arcsin * y y 
after: * y y 

before: ln exp * sin asin y y 
after: * y y 

before: arcsin sin * y y 
after: * y y 

before: ln exp * asin sin y y 
after: * y y 
'''
old = '''before: * * x x 1 
after: * * x x 1 

before: * + sin x x 1 
after: * + sin x x 1 

before: + ~ tanh * x 1 * 1 1 
after: + ~ tanh x 1 

before: + - sin x sin x sin x 
after: + - sin x sin x sin x 

before: / x 1 
after: x 

before: / * x x 1 
after: / * x x 1 

before: / * x cos x 1 
after: / * x cos x 1 

before: / 0 * x x 
after: / 0 * x x 

before: / 0 * x cos x 
after: / 0 * x cos x 

before: / 0 * sin x sech x 
after: / 0 * sin x sech x 

before: / 1 * x x 
after: / 1 * x x 

before: / 1 cos x 
after: / 1 cos x 

before: / 1 * cos x sin x 
after: / 1 * cos x sin x 

before: + x sin ~ ~ x 
after: + x sin x 

before: - tanh ~ ~ x x 
after: - tanh x x 

before: + ^ 0 x x 
after: x 

before: - ^ 0 x x 
after: ^ x x 

before: - cos x ^ 0 x 
after: - cos x 0 

before: + x ^ x 0 
after: + x 1 

before: - ^ x 0 x 
after: - 1 x 

before: - cos x ^ x 0 
after: - cos x 1 

before: + x ^ 1 x 
after: + x 1 

before: - ^ 1 x x 
after: - 1 x 

before: - cos x ^ 1 x 
after: - cos x 1 

before: + x ^ x 1 
after: + x x 

before: - ^ x 1 x 
after: 0 

before: - cos x ^ x 1 
after: - cos x x 

before: ln * 1 exp x 
after: ln * 1 exp x 

before: - x ln * 1 exp x 
after: - x ln * 1 exp x 

before: cos - x ln * 1 exp x 
after: cos - x ln * 1 exp x 

before: ln exp * y y 
after: * y y 

before: * exp * x x ln y 
after: * exp * x x ln y 

before: + exp exp - y y sin x 
after: + 2.718282 sin x 

before: / sin * x x y 
after: / sin * x x y 

before: / sin cos x y 
after: / sin cos x y 

before: cos sqrt - x x 
after: 1 

before: sin tanh sqrt - * x x * x x 
after: sin tanh sqrt - * x x * x x 

before: sqrt sqrt - * x cos x * x cos x 
after: sqrt sqrt - * x cos x * x cos x 

before: cos arcsin - x x 
after: 1 

before: sin tanh asin - ^ x x ^ x x 
after: sin tanh asin - ^ x x ^ x x 

before: asin arcsin - * x sin x * x sin x 
after: asin arcsin - * x sin x * x sin x 

before: exp acos - tanh x tanh x 
after: exp acos - tanh x tanh x 

before: sech sech arccos - / x x / x x 
after: 0.925521 

before: acos arccos - - x sech x - x sech x 
after: acos arccos - - x sech x - x sech x 

before: acos tanh - * x exp x * x exp x 
after: acos tanh - * x exp x * x exp x 

before: asin sech - * x exp x * x exp x 
after: asin sech - * x exp x * x exp x 

before: acos sech - - x sech x - x sech x 
after: acos sech - - x sech x - x sech x 

before: + ~ * 0 tanh tanh x x 
after: + ~ * 0 tanh tanh x x 

before: * * y 1 * x2 1 
after: * y x2 

before: * + 0 y + 0 x2 
after: * y x2 

before: * + x 0 + 0 y 
after: * x y 

before: / + x3 0 + 0 y 
after: / x3 y 

before: / / 0 x3 / 1 y 
after: / 0 / 1 y 

before: / / 0 w / y 1 
after: 0 

before: cos acos * y y 
after: * y y 

before: ln exp * cos arccos y y 
after: * y y 

before: arccos cos * y y 
after: * y y 

before: ln exp * arccos cos y y 
after: * y y 

before: sin arcsin * y y 
after: * y y 

before: ln exp * sin asin y y 
after: * y y 

before: arcsin sin * y y 
after: * y y 

before: ln exp * asin sin y y 
after: * y y 
'''
print(old==new)
