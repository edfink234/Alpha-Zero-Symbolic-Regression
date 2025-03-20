before='''before: - - - x1 x1 0 + x1 x1 
after: ~ + x1 x1 

before: - - - 2.33 1.222 0 x1 
after: - 1.108000 x1 

before: + - 0 x - 0 - 0 y 
after: + ~ x y 

before: + - x 0 - 0 - y 0 
after: + x ~ y 

before: cos + - 3 0 - 0 - 4 0 
after: 0.540302 

before: + + - * ^ exp log 20.000000 / x1 - ~ 0 exp x0 * ln exp log 20.000000 / - - ~ 0 exp x0 * x1 ~ 0 * - ~ 0 exp x0 - ~ 0 exp x0 * -0.214359 * ^ exp log 20.000000 / x1 - ~ 0 exp x0 * ln exp log 20.000000 / ~ * x1 - ~ 0 exp x0 * - ~ 0 exp x0 - ~ 0 exp x0 / * 0.001370 ^ exp log 20.000000 / x1 - ~ 0 exp x0 + 1.244282 ^ exp log 20.000000 / x1 - ~ 0 exp x0 * * 1.238819 ^ exp log 20.000000 / x1 - ~ 0 exp x0 sech exp * 0.805109 + x0 x1 
after: + + - * ^ 20.000000 / x1 ~ exp x0 * 2.995732 / ~ exp x0 * ~ exp x0 ~ exp x0 * -0.214359 * ^ 20.000000 / x1 ~ exp x0 * 2.995732 / ~ * x1 ~ exp x0 * ~ exp x0 ~ exp x0 / * 0.001370 ^ 20.000000 / x1 ~ exp x0 + 1.244282 ^ 20.000000 / x1 ~ exp x0 * * 1.238819 ^ 20.000000 / x1 ~ exp x0 sech exp * 0.805109 + x0 x1 

before: + x x 
after: + x x 

before: + - x x x 
after: x 

before: + - - x x x y 
after: + ~ x y 

before: + cos / * y y x y 
after: + cos / * y y x y 

before: + cos * * y x y y 
after: + cos * * y x y y 

before: + * x x y 
after: + * x x y 

before: + + x x y 
after: + + x x y 

before: + + cos x x y 
after: + + cos x x y 

before: - y + cos x x 
after: - y + cos x x 

before: - y x 
after: - y x 

before: * x cos cos - y x 
after: * x cos cos - y x 

before: + x / x sin - y x 
after: + x / x sin - y x 

before: / x / x * y cos sin y 
after: / x / x * y cos sin y 

before: / sin ~ ~ x y 
after: / sin x y 

before: sqrt x 
after: sqrt x 

before: * sqrt x y 
after: * sqrt x y 

before: * ln x y 
after: * ln x y 

before: * ln ~ x x 
after: * ln ~ x x 

before: * ln sqrt x y 
after: * ln sqrt x y 

before: asin * x x 
after: asin * x x 

before: arcsin * ln x y 
after: arcsin * ln x y 

before: arcsin * ln x y 
after: arcsin * ln x y 

before: arcsin / acos x y 
after: arcsin / acos x y 

before: + arcsin * ln x y acos y 
after: + arcsin * ln x y acos y 

before: acos * acos acos x ~ x 
after: acos * acos acos x ~ x 

before: / exp x exp cos x 
after: / exp x exp cos x 

before: + exp ~ x * * x y x 
after: + exp ~ x * * x y x 

before: arccos * exp arcsin y ~ x 
after: arccos * exp arcsin y ~ x 

before: ^ x y 
after: ^ x y 

before: * ^ cos x cos y x 
after: * ^ cos x cos y x 

before: * ^ cos x cos y x 
after: * ^ cos x cos y x 

before: * ^ ^ x x x y 
after: * ^ ^ x x x y 

before: * ^ ^ x x x y 
after: * ^ ^ x x x y 

before: * ^ tanh sech x x y 
after: * ^ tanh sech x x y 

before: * x ^ tanh / x y sin x 
after: * x ^ tanh / x y sin x 

before: sech sin sin ^ sech sin x * x y 
after: sech sin sin ^ sech sin x * x y 

before: sin ~ sech / arccos ln x * x y 
after: sin ~ sech / arccos ln x * x y 

before: * 0 x 
after: 0 

before: - * 0 x + x sin x 
after: ~ + x sin x 

before: + ~ * 0 x tanh x 
after: tanh x 

before: * 1 x 
after: x 

before: - * 1 x + x sin x 
after: - x + x sin x 

before: + ~ * 1 x tanh x 
after: + ~ x tanh x 

before: * x 0 
after: 0 

before: - * x 0 + x sin x 
after: ~ + x sin x 

before: + ~ * x 0 tanh x 
after: tanh x 

before: * * x x 1 
after: * x x 

before: * + sin x x 1 
after: + sin x x 

before: + ~ tanh * x 1 * 1 1 
after: + ~ tanh x 1 

before: + - sin x sin x sin x 
after: sin x 

before: / x 1 
after: x 

before: / * x x 1 
after: * x x 

before: / * x cos x 1 
after: * x cos x 

before: / 0 * x x 
after: 0 

before: / 0 * x cos x 
after: 0 

before: / 0 * sin x sech x 
after: 0 

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
after: ~ x 

before: - cos x ^ 0 x 
after: cos x 

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
after: 0 

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

before: + 0 + x + 0 + x + x x 
after: + x + x + x x 

before: - 0 + x - 0 + x + x x 
after: ~ + x ~ + x + x x 

before: + tanh cos x ^ 0 x 
after: tanh cos x 

before: - tanh cos x tanh cos x 
after: 0 

before: * tanh cos x 0 
after: 0 

before: * + tanh cos x 0 0 
after: 0 

before: * 0 tanh tanh x 
after: 0 

before: + 0 * 0 cos tanh x 
after: 0 

before: * 1 * x + x x 
after: * x + x x 

before: + 0 * 1 + tanh x x 
after: + tanh x x 

before: / tanh cos x 0 
after: inf 

before: sin arcsin / ~ * y y 0 
after: -inf 

before: / sin arcsin / ~ * y y x sin arcsin / ~ * y y x 
after: 1 

before: + * x x / sin arcsin / ~ * y y x sin arcsin / ~ * y y x 
after: + * x x 1 

before: ^ sin arcsin / ~ * y y 0 0 
after: 1 

before: ^ * 1 + tanh x x 0 
after: 1 
'''
after='''before: - - - x1 x1 0 + x1 x1 
after: ~ + x1 x1 

before: - - - 2.33 1.222 0 x1 
after: - 1.108000 x1 

before: + - 0 x - 0 - 0 y 
after: + ~ x y 

before: + - x 0 - 0 - y 0 
after: + x ~ y 

before: cos + - 3 0 - 0 - 4 0 
after: 0.540302 

before: + + - * ^ exp log 20.000000 / x1 - ~ 0 exp x0 * ln exp log 20.000000 / - - ~ 0 exp x0 * x1 ~ 0 * - ~ 0 exp x0 - ~ 0 exp x0 * -0.214359 * ^ exp log 20.000000 / x1 - ~ 0 exp x0 * ln exp log 20.000000 / ~ * x1 - ~ 0 exp x0 * - ~ 0 exp x0 - ~ 0 exp x0 / * 0.001370 ^ exp log 20.000000 / x1 - ~ 0 exp x0 + 1.244282 ^ exp log 20.000000 / x1 - ~ 0 exp x0 * * 1.238819 ^ exp log 20.000000 / x1 - ~ 0 exp x0 sech exp * 0.805109 + x0 x1 
after: + + - * ^ 20.000000 / x1 ~ exp x0 * 2.995732 / ~ exp x0 * ~ exp x0 ~ exp x0 * -0.214359 * ^ 20.000000 / x1 ~ exp x0 * 2.995732 / ~ * x1 ~ exp x0 * ~ exp x0 ~ exp x0 / * 0.001370 ^ 20.000000 / x1 ~ exp x0 + 1.244282 ^ 20.000000 / x1 ~ exp x0 * * 1.238819 ^ 20.000000 / x1 ~ exp x0 sech exp * 0.805109 + x0 x1 

before: + x x 
after: + x x 

before: + - x x x 
after: x 

before: + - - x x x y 
after: + ~ x y 

before: + cos / * y y x y 
after: + cos / * y y x y 

before: + cos * * y x y y 
after: + cos * * y x y y 

before: + * x x y 
after: + * x x y 

before: + + x x y 
after: + + x x y 

before: + + cos x x y 
after: + + cos x x y 

before: - y + cos x x 
after: - y + cos x x 

before: - y x 
after: - y x 

before: * x cos cos - y x 
after: * x cos cos - y x 

before: + x / x sin - y x 
after: + x / x sin - y x 

before: / x / x * y cos sin y 
after: / x / x * y cos sin y 

before: / sin ~ ~ x y 
after: / sin x y 

before: sqrt x 
after: sqrt x 

before: * sqrt x y 
after: * sqrt x y 

before: * ln x y 
after: * ln x y 

before: * ln ~ x x 
after: * ln ~ x x 

before: * ln sqrt x y 
after: * ln sqrt x y 

before: asin * x x 
after: asin * x x 

before: arcsin * ln x y 
after: arcsin * ln x y 

before: arcsin * ln x y 
after: arcsin * ln x y 

before: arcsin / acos x y 
after: arcsin / acos x y 

before: + arcsin * ln x y acos y 
after: + arcsin * ln x y acos y 

before: acos * acos acos x ~ x 
after: acos * acos acos x ~ x 

before: / exp x exp cos x 
after: / exp x exp cos x 

before: + exp ~ x * * x y x 
after: + exp ~ x * * x y x 

before: arccos * exp arcsin y ~ x 
after: arccos * exp arcsin y ~ x 

before: ^ x y 
after: ^ x y 

before: * ^ cos x cos y x 
after: * ^ cos x cos y x 

before: * ^ cos x cos y x 
after: * ^ cos x cos y x 

before: * ^ ^ x x x y 
after: * ^ ^ x x x y 

before: * ^ ^ x x x y 
after: * ^ ^ x x x y 

before: * ^ tanh sech x x y 
after: * ^ tanh sech x x y 

before: * x ^ tanh / x y sin x 
after: * x ^ tanh / x y sin x 

before: sech sin sin ^ sech sin x * x y 
after: sech sin sin ^ sech sin x * x y 

before: sin ~ sech / arccos ln x * x y 
after: sin ~ sech / arccos ln x * x y 

before: * 0 x 
after: 0 

before: - * 0 x + x sin x 
after: ~ + x sin x 

before: + ~ * 0 x tanh x 
after: tanh x 

before: * 1 x 
after: x 

before: - * 1 x + x sin x 
after: - x + x sin x 

before: + ~ * 1 x tanh x 
after: + ~ x tanh x 

before: * x 0 
after: 0 

before: - * x 0 + x sin x 
after: ~ + x sin x 

before: + ~ * x 0 tanh x 
after: tanh x 

before: * * x x 1 
after: * x x 

before: * + sin x x 1 
after: + sin x x 

before: + ~ tanh * x 1 * 1 1 
after: + ~ tanh x 1 

before: + - sin x sin x sin x 
after: sin x 

before: / x 1 
after: x 

before: / * x x 1 
after: * x x 

before: / * x cos x 1 
after: * x cos x 

before: / 0 * x x 
after: 0 

before: / 0 * x cos x 
after: 0 

before: / 0 * sin x sech x 
after: 0 

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
after: ~ x 

before: - cos x ^ 0 x 
after: cos x 

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
after: 0 

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

before: + 0 + x + 0 + x + x x 
after: + x + x + x x 

before: - 0 + x - 0 + x + x x 
after: ~ + x ~ + x + x x 

before: + tanh cos x ^ 0 x 
after: tanh cos x 

before: - tanh cos x tanh cos x 
after: 0 

before: * tanh cos x 0 
after: 0 

before: * + tanh cos x 0 0 
after: 0 

before: * 0 tanh tanh x 
after: 0 

before: + 0 * 0 cos tanh x 
after: 0 

before: * 1 * x + x x 
after: * x + x x 

before: + 0 * 1 + tanh x x 
after: + tanh x x 

before: / tanh cos x 0 
after: inf 

before: sin arcsin / ~ * y y 0 
after: -inf 

before: / sin arcsin / ~ * y y x sin arcsin / ~ * y y x 
after: 1 

before: + * x x / sin arcsin / ~ * y y x sin arcsin / ~ * y y x 
after: + * x x 1 

before: ^ sin arcsin / ~ * y y 0 0 
after: 1 

before: ^ * 1 + tanh x x 0 
after: 1 
'''
print(before==after)
