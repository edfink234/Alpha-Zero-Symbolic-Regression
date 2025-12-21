after='''before: - - - x1 x1 0 + x1 x1 
after: ~ + x1 x1 

before: - - - 2.33 1.222 0 x1 
after: - 1.108 x1 

before: + - 0 x - 0 - 0 y 
after: + ~ x y 

before: + - x 0 - 0 - y 0 
after: + x ~ y 

before: cos + - 3 0 - 0 - 4 0 
after: 0.5403023058681398 

before: + + - * ^ exp log 20.000000 / x1 - ~ 0 exp x0 * ln exp log 20.000000 / - - ~ 0 exp x0 * x1 ~ 0 * - ~ 0 exp x0 - ~ 0 exp x0 * -0.214359 * ^ exp log 20.000000 / x1 - ~ 0 exp x0 * ln exp log 20.000000 / ~ * x1 - ~ 0 exp x0 * - ~ 0 exp x0 - ~ 0 exp x0 / * 0.001370 ^ exp log 20.000000 / x1 - ~ 0 exp x0 + 1.244282 ^ exp log 20.000000 / x1 - ~ 0 exp x0 * * 1.238819 ^ exp log 20.000000 / x1 - ~ 0 exp x0 sech exp * 0.805109 + x0 x1 
after: + + - * ^ 20.000000 / x1 ~ exp x0 * 2.995732273553991 / ~ exp x0 * ~ exp x0 ~ exp x0 * -0.214359 * ^ 20.000000 / x1 ~ exp x0 * 2.995732273553991 / ~ * x1 ~ exp x0 * ~ exp x0 ~ exp x0 / * 0.001370 ^ 20.000000 / x1 ~ exp x0 + 1.244282 ^ 20.000000 / x1 ~ exp x0 * * 1.238819 ^ 20.000000 / x1 ~ exp x0 sech exp * 0.805109 + x0 x1 

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
after: nan 

before: - ^ 0 x x 
after: nan 

before: - cos x ^ 0 x 
after: nan 

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
after: x 

before: - x ln * 1 exp x 
after: 0 

before: cos - x ln * 1 exp x 
after: 1 

before: ln exp * y y 
after: * y y 

before: * exp * x x ln y 
after: * exp * x x ln y 

before: + exp exp - y y sin x 
after: + 2.718281828459045 sin x 

before: / sin * x x y 
after: / sin * x x y 

before: / sin cos x y 
after: / sin cos x y 

before: cos sqrt - x x 
after: 1 

before: sin tanh sqrt - * x x * x x 
after: 0 

before: sqrt sqrt - * x cos x * x cos x 
after: 0 

before: cos arcsin - x x 
after: 1 

before: sin tanh asin - ^ x x ^ x x 
after: 0 

before: asin arcsin - * x sin x * x sin x 
after: 0 

before: exp acos - tanh x tanh x 
after: 4.810477380965351 

before: sech sech arccos - / x x / x x 
after: 0.9255209913302057 

before: acos arccos - - x sech x - x sech x 
after: nan 

before: acos tanh - * x exp x * x exp x 
after: 1.5707963267948966 

before: asin sech - * x exp x * x exp x 
after: 1.5707963267948966 

before: acos sech - - x sech x - x sech x 
after: 0 

before: + ~ * 0 tanh tanh x x 
after: x 

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
after: nan 

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
after: nan 

before: sin arcsin / ~ * y y 0 
after: nan 

before: / sin arcsin / ~ * y y x sin arcsin / ~ * y y x 
after: 1 

before: + * x x / sin arcsin / ~ * y y x sin arcsin / ~ * y y x 
after: + * x x 1 

before: ^ sin arcsin / ~ * y y 0 0 
after: nan 

before: ^ * 1 + tanh x x 0 
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
after: nan 

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
after: nan 

before: exp * sech ~ / ~ tanh cos x sin + 0 0 0 
after: nan 

before: exp * sech ~ / ~ tanh cos x sin / 0 0 0 
after: nan 

before: cos + tanh ~ / ~ tanh cos x sin / x 0 0 
after: nan 

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

before: ^ nan sech + 1 + x tanh - 2 ^ x 2 
after: nan 

before: ^ sech asin - apple tanh sech acos arccos + apple 2 nan 
after: nan 

before: cos ^ sech asin - apple tanh sech acos arccos + apple 2 nan 
after: nan 

before: cos ^ nan sech + 1 + x tanh - 2 ^ x 2 
after: nan 

before: sin ^ sech acos - apple tanh sech acos arccos + apple 2 nan 
after: nan 

before: sin ^ nan tanh + 1 + x tanh - 2 ^ x 2 
after: nan 

before: tanh * sech acos - apple tanh sech acos arccos + apple 2 nan 
after: nan 

before: tanh * nan tanh + 1 + x tanh - 3 ^ x 2 
after: nan 

before: sech + sech acos - orange tanh sech acos arccos + apple 2 nan 
after: nan 

before: sech - nan tanh + 3 + x tanh - 3 cos ^ x 2 
after: nan 

before: ~ sech + sech acos - orange tanh sech acos arccos + apple 2 nan 
after: nan 

before: ~ sech - nan tanh + 3 + x tanh - 3 sin ^ x 2 
after: nan 

before: exp ~ sech + sech acos - orange tanh sin arccos arccos - apple 2 nan 
after: nan 

before: exp ~ sech - nan arcsin - 30 + x tanh - 3 sin ^ x 2 
after: nan 

before: + + - 9736 x7 / -100.051731 ^ -0.000000 x15 * + x20 * 1075.000000 x5 -17064.107062 
after: nan 

before: + x -0.000 
after: x 

before: - x -0.000 
after: x 

before: + -0.000 sin x 
after: sin x 

before: - -0.00000 cos x 
after: ~ cos x 

before: * x -0.000 
after: 0 

before: * sin + x x 0.000 
after: 0 

before: * -0.000 ^ 2 x 
after: 0 

before: * 0.000000e0 sin + x x 
after: 0 

before: / -0.000 * 0.0000 x 
after: nan 

before: / 0.000000e0 tanh * 0.e0 x 
after: nan 

before: / apple sin * 0.0000e0 x 
after: nan 

before: / watermelon sin tanh * 0.e0 x 
after: nan 

before: / -0.000e0 * x + x x 
after: 0 

before: / 0.000000e0 tanh / sin x 0 
after: nan 

before: ^ cos x sin * 0.0000e0 x 
after: 1 

before: ^ + x x1 sin tanh * 0.e0 x 
after: 1 

before: ^ -0.0000e0 + x1 + x x 
after: nan 

before: ^ 0.00000 tanh / sech x22 x0 
after: nan 

before: ^ sech x + 1.000e0 * 0.0000e0 x 
after: sech x 

before: ^ + x x1 - sin tanh * 0.e0 x ~ 1.000000 
after: + x x1 

before: cos * 0.00000 ^ sech x + 1.000e0 * 0.0000e0 x 
after: 1 

before: cos * ^ + x x1 - sin tanh * 0.e0 x ~ 1.000000 0.000e0 
after: 1 

before: sin * 0.00000 ^ sech x + 1.000e0 * 0.0000e0 x 
after: 0 

before: sin * ^ + x x1 - sin tanh * 0.e0 x ~ 1.000000 0.000e0 
after: 0 

before: tanh sin * 0.00000 ^ sech x + 1.000e0 * 0.0000e0 x 
after: 0 

before: tanh sin sin * ^ + x x1 - sin tanh * 0.e0 x ~ 1.000000 0.000e0 
after: 0 

before: sech tanh sin * 0.00000 ^ sech x33 + 1.000e0 * 0.0000e0 x 
after: 1 

before: sech tanh sin sin * ^ + x x2 - sin tanh * 0.e0 x ~ 1.000000 0.000e0 
after: 1 

before: ~ tanh sin * 0.00000 ^ sech x + 1.000e0 * 0.0000e0 x 
after: 0 

before: ~ tanh sin tanh sin sin * ^ + x x1 - sin tanh * 0.e0 x ~ 1.000000 0.000e0 
after: 0 

before: exp ~ tanh sin * 0.00000 ^ sech x345234234 + 1.000e0 * 0.0000e0 x 
after: 1 

before: exp ~ tanh sin tanh sin sin * ^ + x111 x1 - sin tanh * 0.e0 x ~ 1.000000 0.000e0 
after: 1 

before: - 0.0000 x 
after: ~ x 

before: - -0.0000e0 x 
after: ~ x 

before: - x 0.000 
after: x 

before: - x12 0.000e0 
after: x12 

before: * 0.0000 x 
after: 0 

before: * -0.0000e0 x 
after: 0 

before: * x 0.00000 
after: 0 

before: * x12 0.000e0 
after: 0 

before: * 1.0000 x 
after: x 

before: * 1.0000e0 x 
after: x 

before: * x 1.00000 
after: x 

before: * x122 1.00000e0 
after: x122 

before: + 0.0000 x 
after: x 

before: + -0.0000e0 x2 
after: x2 

before: + x 0.000000 
after: x 

before: + x112 0.00000e0 
after: x112 

before: / 0.0000e0 x 
after: 0 

before: / -0 x2 
after: 0 

before: / x 100000.0e-5 
after: x 

before: / x112 1.00000 
after: x112 

before: ^ x 0.000000 
after: 1 

before: ^ x112 0.00000e0 
after: 1 

before: ^ 0.0000e0 x 
after: nan 

before: ^ -0.00e0 x4 
after: nan 

before: ^ x 10000.0e-4 
after: x 

before: ^ x133 1.00000e0 
after: x133 

before: ^ 1000.0000e-3 x 
after: 1 

before: - ^ x 0.000 1.00000 
after: 0 

before: - ^ + y x 0.000 1.00000 
after: 0 

before: tanh + - ^ x 0.000 1.00000 ~ inf 
after: -1 

before: tanh + - ^ + y x 0.000 1.00000 ~ inf 
after: -1 

before: sech ~ sin tanh sqrt - * x x * x x 
after: 1 

before: sech ~ sqrt sqrt - * x cos x * x cos x 
after: 1 

before: cos + - ^ x 0.000 1.00000 ~ inf 
after: nan 

before: cos + - ^ + y x 0.000 1.00000 ~ inf 
after: nan 

before: cos + - ^ x 0.000 1.00000 exp inf 
after: nan 

before: cos + - ^ + y x 0.000 1.00000 exp inf 
after: nan 

before: sin + - ^ x 0.000 1.00000 ~ inf 
after: nan 

before: sin + - ^ + y x 0.000 1.00000 ~ inf 
after: nan 

before: sin + - ^ x 0.000 1.00000 exp exp inf 
after: nan 

before: sin + - ^ + y x 0.000 1.00000 exp exp inf 
after: nan 

before: * tanh + - ^ x 0.000 1.00000 ~ inf x 
after: ~ x 

before: * tanh + - ^ + y x 0.000 1.00000 ~ inf x 
after: ~ x 

before: * x + tanh + - ^ x 0.000 1.00000 ~ inf 0.00000 
after: ~ x 

before: * x + tanh + - ^ + y x 0.000 1.00000 ~ inf - 1 1 
after: ~ x 

before: sech / x + tanh + - ^ x 0.000 1.00000 ~ inf 0.00000 
after: sech x 

before: sech / x + tanh + - ^ + y x 0.000 1.00000 ~ inf - 1 1 
after: sech x 

before: ln ~ - cos + x y cos + x y 
after: -inf 

before: ln sin - sin sin + x y sin sin + x y 
after: -inf 

before: ln ^ ~ - cos + x y cos + x y banana 
after: nan 

before: ln / + x water sin - sin sin + x y sin sin + x y 
after: nan 

before: ln + inf ~ - cos + x y cos + x y 
after: inf 

before: ln + sin - sin sin + x y sin sin + x y exp inf 
after: inf 

before: asin tanh - cos + x y cos + x y 
after: 0 

before: asin sin sin - sin sin + x y sin sin + x y 
after: 0 

before: asin + nan tanh - cos + x y cos + x y 
after: nan 

before: asin / sin sin - sin sin + x y sin sin + x y sin nan 
after: nan 

before: asin exp + inf tanh - cos + x y cos + x y 
after: nan 

before: asin + sin sin - sin sin + x y sin sin + x y exp exp exp inf 
after: nan 

before: arccos + 1.0000 tanh - cos + x y cos + x y 
after: 0 

before: acos + sin sin - sin sin + x y sin sin + x y exp 0.00000e0 
after: 0 

before: arccos / + 1.0000 tanh - cos + x y cos + x y cos nan 
after: nan 

before: acos * + sin sin - sin sin + x y sin sin + x y exp 0.00000e0 sin sech / x .000 
after: nan 

before: arccos * exp inf + 1.0000 tanh - cos + x y cos + x y 
after: nan 

before: acos * + sin sin - sin sin + x y sin sin + x y exp 0.00000e0 ln exp ln inf 
after: nan 

before: sqrt * 1.0000 arccos + 1.0000 tanh - cos + x y cos + x y 
after: 0 

before: sqrt * 1.09214234 acos + sin sin - sin sin + x y sin sin + x y exp 0.00000e0 
after: 0 

before: sqrt * nan arccos + 1.0000 tanh - cos + x y cos + x y 
after: nan 

before: sqrt * + nan sech 1.09214234 acos + sin sin - sin sin + x y sin sin + x y exp 0.00000e0 
after: nan 

before: sqrt + inf * 1.0000 arccos + 1.0000 tanh - cos + x y cos + x y 
after: inf 

before: sqrt + exp inf * 1.09214234 acos + sin sin - sin sin + x y sin sin + x y exp 0.00000e0 
after: inf 

before: sqrt + * 1.0000 arccos + 1.0000 tanh - cos + x y cos + x y 1.0000e0 
after: 1 

before: sqrt - * 1.09214234 acos + sin sin - sin sin + x y sin sin + x y exp 0.00000e0 -1.000 
after: 1 

before: cos ~ + inf * 1.0000 arccos + 1.0000 tanh - cos + x y cos + x y 
after: nan 

before: cos ~ + exp inf * 1.09214234 acos + sin sin - sin sin + x y sin sin + x y exp 0.00000e0 
after: nan 

before: cos ~ exp + inf tanh - cos + x y cos + x y 
after: nan 

before: cos ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 
after: nan 

before: sin * 1.000 ~ + inf * 1.0000 arccos + 1.0000 tanh - cos + x y cos + x y 
after: nan 

before: sin / ~ + exp inf * 1.09214234 acos + sin sin - sin sin + x y sin sin + x y exp 0.00000e0 1.0000e0 
after: nan 

before: sin * ~ exp + inf tanh - cos + x y cos + x y 1 
after: nan 

before: sin + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 
after: nan 

before: ~ * sqrt 1.000 * 1.000 ~ + inf * 1.0000 arccos + 1.0000 tanh - cos + x y cos + x y 
after: inf 

before: ~ * / ~ + exp inf * 1.09214234 acos + sin sin - sin sin + x y sin sin + x y exp 0.00000e0 1.0000e0 1.000e0 
after: inf 

before: ~ * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 
after: inf 

before: ~ / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 
after: inf 

before: exp ~ ~ * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 
after: 0 

before: exp ~ ~ ~ ~ / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 
after: 0 

before: exp * 1.00e0 * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 
after: 0 

before: exp * / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 100.0e-2 
after: 0 

before: ln + sin 0.000 ~ ~ * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 
after: nan 

before: log ~ ~ ~ ~ ~ ~ / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 
after: nan 

before: ln + * 1.00e0 * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 0.00e0 
after: nan 

before: ln + * / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 100.0e-2 sin 0.0000 
after: nan 

before: asin + sin 0.0000 + sin 0.000 ~ ~ * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 
after: nan 

before: arcsin * 1.0000 ~ ~ ~ ~ ~ ~ / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 
after: nan 

before: arcsin * + * 1.00e0 * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 0.00e0 exp 0.000000 
after: nan 

before: asin / + * / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 100.0e-2 sin 0.0000 exp sin 0.000e0 
after: nan 

before: arccos + 0 + sin 0.0000 + sin 0.000 ~ ~ * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 
after: nan 

before: acos * 1.000 ~ ~ ~ ~ ~ ~ / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 
after: nan 

before: acos * + * 1.00e0 * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 0.00e0 exp 0.00000000e0 
after: nan 

before: arccos * 1 / + * / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 100.0e-2 sin 0.0000 exp sin 0.000e0 
after: nan 

before: sqrt + 0 + sin 0.0000 + sin 0.000 ~ ~ * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00e0 
after: nan 

before: sqrt * 1000.e-3 ~ ~ ~ ~ ~ ~ / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 
after: nan 

before: sqrt * + * 1.00e0 * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 0.00e0 exp sin 0.00000000e0 
after: nan 

before: sqrt * 1.000e0 / + * / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 100.0e-2 sin 0.0000 exp sin 0.000e0 
after: nan 

before: sqrt - exp ~ ~ * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 1.00000 
after: nan 

before: sqrt + -1.000 exp ~ ~ ~ ~ / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 
after: nan 

before: sqrt + * -1 1.000e0 exp * 1.00e0 * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 
after: nan 

before: sqrt + exp * / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 100.0e-2 -100e-2 
after: nan 

before: ~ * -1.000e0 + * -1 1.000e0 exp * 1.00e0 * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 
after: -1.000e0 

before: ~ * + exp * / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 100.0e-2 -100e-2 -100e-2 
after: -100e-2 

before: ~ + sin 0.000e0 - exp ~ ~ * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 1.00000 
after: 1.00000 

before: ~ + -1.000 exp ~ ~ ~ ~ / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 
after: 1 

before: / * sin cos + x y ~ + sin 0.000e0 - exp ~ ~ * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 1.00000 0.000e0 
after: nan 

before: / * ~ + -1.000 exp ~ ~ ~ ~ / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 tanh tanh x sin 0.00000 
after: nan 

before: * 1e500 + 3.444 cos 1.22 
after: inf 

before: * -2e390 + + 3.444 cos 1.22 + 3.444 sin 1.22 
after: -inf 
'''
before='''before: - - - x1 x1 0 + x1 x1 
after: ~ + x1 x1 

before: - - - 2.33 1.222 0 x1 
after: - 1.108 x1 

before: + - 0 x - 0 - 0 y 
after: + ~ x y 

before: + - x 0 - 0 - y 0 
after: + x ~ y 

before: cos + - 3 0 - 0 - 4 0 
after: 0.5403023058681398 

before: + + - * ^ exp log 20.000000 / x1 - ~ 0 exp x0 * ln exp log 20.000000 / - - ~ 0 exp x0 * x1 ~ 0 * - ~ 0 exp x0 - ~ 0 exp x0 * -0.214359 * ^ exp log 20.000000 / x1 - ~ 0 exp x0 * ln exp log 20.000000 / ~ * x1 - ~ 0 exp x0 * - ~ 0 exp x0 - ~ 0 exp x0 / * 0.001370 ^ exp log 20.000000 / x1 - ~ 0 exp x0 + 1.244282 ^ exp log 20.000000 / x1 - ~ 0 exp x0 * * 1.238819 ^ exp log 20.000000 / x1 - ~ 0 exp x0 sech exp * 0.805109 + x0 x1 
after: + + - * ^ 20.000000 / x1 ~ exp x0 * 2.995732273553991 / ~ exp x0 * ~ exp x0 ~ exp x0 * -0.214359 * ^ 20.000000 / x1 ~ exp x0 * 2.995732273553991 / ~ * x1 ~ exp x0 * ~ exp x0 ~ exp x0 / * 0.001370 ^ 20.000000 / x1 ~ exp x0 + 1.244282 ^ 20.000000 / x1 ~ exp x0 * * 1.238819 ^ 20.000000 / x1 ~ exp x0 sech exp * 0.805109 + x0 x1 

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
after: nan 

before: - ^ 0 x x 
after: nan 

before: - cos x ^ 0 x 
after: nan 

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
after: x 

before: - x ln * 1 exp x 
after: 0 

before: cos - x ln * 1 exp x 
after: 1 

before: ln exp * y y 
after: * y y 

before: * exp * x x ln y 
after: * exp * x x ln y 

before: + exp exp - y y sin x 
after: + 2.718281828459045 sin x 

before: / sin * x x y 
after: / sin * x x y 

before: / sin cos x y 
after: / sin cos x y 

before: cos sqrt - x x 
after: 1 

before: sin tanh sqrt - * x x * x x 
after: 0 

before: sqrt sqrt - * x cos x * x cos x 
after: 0 

before: cos arcsin - x x 
after: 1 

before: sin tanh asin - ^ x x ^ x x 
after: 0 

before: asin arcsin - * x sin x * x sin x 
after: 0 

before: exp acos - tanh x tanh x 
after: 4.810477380965351 

before: sech sech arccos - / x x / x x 
after: 0.9255209913302057 

before: acos arccos - - x sech x - x sech x 
after: nan 

before: acos tanh - * x exp x * x exp x 
after: 1.5707963267948966 

before: asin sech - * x exp x * x exp x 
after: 1.5707963267948966 

before: acos sech - - x sech x - x sech x 
after: 0 

before: + ~ * 0 tanh tanh x x 
after: x 

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
after: nan 

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
after: nan 

before: sin arcsin / ~ * y y 0 
after: nan 

before: / sin arcsin / ~ * y y x sin arcsin / ~ * y y x 
after: 1 

before: + * x x / sin arcsin / ~ * y y x sin arcsin / ~ * y y x 
after: + * x x 1 

before: ^ sin arcsin / ~ * y y 0 0 
after: nan 

before: ^ * 1 + tanh x x 0 
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
after: nan 

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
after: nan 

before: exp * sech ~ / ~ tanh cos x sin + 0 0 0 
after: nan 

before: exp * sech ~ / ~ tanh cos x sin / 0 0 0 
after: nan 

before: cos + tanh ~ / ~ tanh cos x sin / x 0 0 
after: nan 

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

before: ^ nan sech + 1 + x tanh - 2 ^ x 2 
after: nan 

before: ^ sech asin - apple tanh sech acos arccos + apple 2 nan 
after: nan 

before: cos ^ sech asin - apple tanh sech acos arccos + apple 2 nan 
after: nan 

before: cos ^ nan sech + 1 + x tanh - 2 ^ x 2 
after: nan 

before: sin ^ sech acos - apple tanh sech acos arccos + apple 2 nan 
after: nan 

before: sin ^ nan tanh + 1 + x tanh - 2 ^ x 2 
after: nan 

before: tanh * sech acos - apple tanh sech acos arccos + apple 2 nan 
after: nan 

before: tanh * nan tanh + 1 + x tanh - 3 ^ x 2 
after: nan 

before: sech + sech acos - orange tanh sech acos arccos + apple 2 nan 
after: nan 

before: sech - nan tanh + 3 + x tanh - 3 cos ^ x 2 
after: nan 

before: ~ sech + sech acos - orange tanh sech acos arccos + apple 2 nan 
after: nan 

before: ~ sech - nan tanh + 3 + x tanh - 3 sin ^ x 2 
after: nan 

before: exp ~ sech + sech acos - orange tanh sin arccos arccos - apple 2 nan 
after: nan 

before: exp ~ sech - nan arcsin - 30 + x tanh - 3 sin ^ x 2 
after: nan 

before: + + - 9736 x7 / -100.051731 ^ -0.000000 x15 * + x20 * 1075.000000 x5 -17064.107062 
after: nan 

before: + x -0.000 
after: x 

before: - x -0.000 
after: x 

before: + -0.000 sin x 
after: sin x 

before: - -0.00000 cos x 
after: ~ cos x 

before: * x -0.000 
after: 0 

before: * sin + x x 0.000 
after: 0 

before: * -0.000 ^ 2 x 
after: 0 

before: * 0.000000e0 sin + x x 
after: 0 

before: / -0.000 * 0.0000 x 
after: nan 

before: / 0.000000e0 tanh * 0.e0 x 
after: nan 

before: / apple sin * 0.0000e0 x 
after: nan 

before: / watermelon sin tanh * 0.e0 x 
after: nan 

before: / -0.000e0 * x + x x 
after: 0 

before: / 0.000000e0 tanh / sin x 0 
after: nan 

before: ^ cos x sin * 0.0000e0 x 
after: 1 

before: ^ + x x1 sin tanh * 0.e0 x 
after: 1 

before: ^ -0.0000e0 + x1 + x x 
after: nan 

before: ^ 0.00000 tanh / sech x22 x0 
after: nan 

before: ^ sech x + 1.000e0 * 0.0000e0 x 
after: sech x 

before: ^ + x x1 - sin tanh * 0.e0 x ~ 1.000000 
after: + x x1 

before: cos * 0.00000 ^ sech x + 1.000e0 * 0.0000e0 x 
after: 1 

before: cos * ^ + x x1 - sin tanh * 0.e0 x ~ 1.000000 0.000e0 
after: 1 

before: sin * 0.00000 ^ sech x + 1.000e0 * 0.0000e0 x 
after: 0 

before: sin * ^ + x x1 - sin tanh * 0.e0 x ~ 1.000000 0.000e0 
after: 0 

before: tanh sin * 0.00000 ^ sech x + 1.000e0 * 0.0000e0 x 
after: 0 

before: tanh sin sin * ^ + x x1 - sin tanh * 0.e0 x ~ 1.000000 0.000e0 
after: 0 

before: sech tanh sin * 0.00000 ^ sech x33 + 1.000e0 * 0.0000e0 x 
after: 1 

before: sech tanh sin sin * ^ + x x2 - sin tanh * 0.e0 x ~ 1.000000 0.000e0 
after: 1 

before: ~ tanh sin * 0.00000 ^ sech x + 1.000e0 * 0.0000e0 x 
after: 0 

before: ~ tanh sin tanh sin sin * ^ + x x1 - sin tanh * 0.e0 x ~ 1.000000 0.000e0 
after: 0 

before: exp ~ tanh sin * 0.00000 ^ sech x345234234 + 1.000e0 * 0.0000e0 x 
after: 1 

before: exp ~ tanh sin tanh sin sin * ^ + x111 x1 - sin tanh * 0.e0 x ~ 1.000000 0.000e0 
after: 1 

before: - 0.0000 x 
after: ~ x 

before: - -0.0000e0 x 
after: ~ x 

before: - x 0.000 
after: x 

before: - x12 0.000e0 
after: x12 

before: * 0.0000 x 
after: 0 

before: * -0.0000e0 x 
after: 0 

before: * x 0.00000 
after: 0 

before: * x12 0.000e0 
after: 0 

before: * 1.0000 x 
after: x 

before: * 1.0000e0 x 
after: x 

before: * x 1.00000 
after: x 

before: * x122 1.00000e0 
after: x122 

before: + 0.0000 x 
after: x 

before: + -0.0000e0 x2 
after: x2 

before: + x 0.000000 
after: x 

before: + x112 0.00000e0 
after: x112 

before: / 0.0000e0 x 
after: 0 

before: / -0 x2 
after: 0 

before: / x 100000.0e-5 
after: x 

before: / x112 1.00000 
after: x112 

before: ^ x 0.000000 
after: 1 

before: ^ x112 0.00000e0 
after: 1 

before: ^ 0.0000e0 x 
after: nan 

before: ^ -0.00e0 x4 
after: nan 

before: ^ x 10000.0e-4 
after: x 

before: ^ x133 1.00000e0 
after: x133 

before: ^ 1000.0000e-3 x 
after: 1 

before: - ^ x 0.000 1.00000 
after: 0 

before: - ^ + y x 0.000 1.00000 
after: 0 

before: tanh + - ^ x 0.000 1.00000 ~ inf 
after: -1 

before: tanh + - ^ + y x 0.000 1.00000 ~ inf 
after: -1 

before: sech ~ sin tanh sqrt - * x x * x x 
after: 1 

before: sech ~ sqrt sqrt - * x cos x * x cos x 
after: 1 

before: cos + - ^ x 0.000 1.00000 ~ inf 
after: nan 

before: cos + - ^ + y x 0.000 1.00000 ~ inf 
after: nan 

before: cos + - ^ x 0.000 1.00000 exp inf 
after: nan 

before: cos + - ^ + y x 0.000 1.00000 exp inf 
after: nan 

before: sin + - ^ x 0.000 1.00000 ~ inf 
after: nan 

before: sin + - ^ + y x 0.000 1.00000 ~ inf 
after: nan 

before: sin + - ^ x 0.000 1.00000 exp exp inf 
after: nan 

before: sin + - ^ + y x 0.000 1.00000 exp exp inf 
after: nan 

before: * tanh + - ^ x 0.000 1.00000 ~ inf x 
after: ~ x 

before: * tanh + - ^ + y x 0.000 1.00000 ~ inf x 
after: ~ x 

before: * x + tanh + - ^ x 0.000 1.00000 ~ inf 0.00000 
after: ~ x 

before: * x + tanh + - ^ + y x 0.000 1.00000 ~ inf - 1 1 
after: ~ x 

before: sech / x + tanh + - ^ x 0.000 1.00000 ~ inf 0.00000 
after: sech x 

before: sech / x + tanh + - ^ + y x 0.000 1.00000 ~ inf - 1 1 
after: sech x 

before: ln ~ - cos + x y cos + x y 
after: -inf 

before: ln sin - sin sin + x y sin sin + x y 
after: -inf 

before: ln ^ ~ - cos + x y cos + x y banana 
after: nan 

before: ln / + x water sin - sin sin + x y sin sin + x y 
after: nan 

before: ln + inf ~ - cos + x y cos + x y 
after: inf 

before: ln + sin - sin sin + x y sin sin + x y exp inf 
after: inf 

before: asin tanh - cos + x y cos + x y 
after: 0 

before: asin sin sin - sin sin + x y sin sin + x y 
after: 0 

before: asin + nan tanh - cos + x y cos + x y 
after: nan 

before: asin / sin sin - sin sin + x y sin sin + x y sin nan 
after: nan 

before: asin exp + inf tanh - cos + x y cos + x y 
after: nan 

before: asin + sin sin - sin sin + x y sin sin + x y exp exp exp inf 
after: nan 

before: arccos + 1.0000 tanh - cos + x y cos + x y 
after: 0 

before: acos + sin sin - sin sin + x y sin sin + x y exp 0.00000e0 
after: 0 

before: arccos / + 1.0000 tanh - cos + x y cos + x y cos nan 
after: nan 

before: acos * + sin sin - sin sin + x y sin sin + x y exp 0.00000e0 sin sech / x .000 
after: nan 

before: arccos * exp inf + 1.0000 tanh - cos + x y cos + x y 
after: nan 

before: acos * + sin sin - sin sin + x y sin sin + x y exp 0.00000e0 ln exp ln inf 
after: nan 

before: sqrt * 1.0000 arccos + 1.0000 tanh - cos + x y cos + x y 
after: 0 

before: sqrt * 1.09214234 acos + sin sin - sin sin + x y sin sin + x y exp 0.00000e0 
after: 0 

before: sqrt * nan arccos + 1.0000 tanh - cos + x y cos + x y 
after: nan 

before: sqrt * + nan sech 1.09214234 acos + sin sin - sin sin + x y sin sin + x y exp 0.00000e0 
after: nan 

before: sqrt + inf * 1.0000 arccos + 1.0000 tanh - cos + x y cos + x y 
after: inf 

before: sqrt + exp inf * 1.09214234 acos + sin sin - sin sin + x y sin sin + x y exp 0.00000e0 
after: inf 

before: sqrt + * 1.0000 arccos + 1.0000 tanh - cos + x y cos + x y 1.0000e0 
after: 1 

before: sqrt - * 1.09214234 acos + sin sin - sin sin + x y sin sin + x y exp 0.00000e0 -1.000 
after: 1 

before: cos ~ + inf * 1.0000 arccos + 1.0000 tanh - cos + x y cos + x y 
after: nan 

before: cos ~ + exp inf * 1.09214234 acos + sin sin - sin sin + x y sin sin + x y exp 0.00000e0 
after: nan 

before: cos ~ exp + inf tanh - cos + x y cos + x y 
after: nan 

before: cos ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 
after: nan 

before: sin * 1.000 ~ + inf * 1.0000 arccos + 1.0000 tanh - cos + x y cos + x y 
after: nan 

before: sin / ~ + exp inf * 1.09214234 acos + sin sin - sin sin + x y sin sin + x y exp 0.00000e0 1.0000e0 
after: nan 

before: sin * ~ exp + inf tanh - cos + x y cos + x y 1 
after: nan 

before: sin + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 
after: nan 

before: ~ * sqrt 1.000 * 1.000 ~ + inf * 1.0000 arccos + 1.0000 tanh - cos + x y cos + x y 
after: inf 

before: ~ * / ~ + exp inf * 1.09214234 acos + sin sin - sin sin + x y sin sin + x y exp 0.00000e0 1.0000e0 1.000e0 
after: inf 

before: ~ * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 
after: inf 

before: ~ / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 
after: inf 

before: exp ~ ~ * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 
after: 0 

before: exp ~ ~ ~ ~ / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 
after: 0 

before: exp * 1.00e0 * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 
after: 0 

before: exp * / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 100.0e-2 
after: 0 

before: ln + sin 0.000 ~ ~ * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 
after: nan 

before: log ~ ~ ~ ~ ~ ~ / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 
after: nan 

before: ln + * 1.00e0 * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 0.00e0 
after: nan 

before: ln + * / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 100.0e-2 sin 0.0000 
after: nan 

before: asin + sin 0.0000 + sin 0.000 ~ ~ * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 
after: nan 

before: arcsin * 1.0000 ~ ~ ~ ~ ~ ~ / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 
after: nan 

before: arcsin * + * 1.00e0 * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 0.00e0 exp 0.000000 
after: nan 

before: asin / + * / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 100.0e-2 sin 0.0000 exp sin 0.000e0 
after: nan 

before: arccos + 0 + sin 0.0000 + sin 0.000 ~ ~ * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 
after: nan 

before: acos * 1.000 ~ ~ ~ ~ ~ ~ / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 
after: nan 

before: acos * + * 1.00e0 * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 0.00e0 exp 0.00000000e0 
after: nan 

before: arccos * 1 / + * / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 100.0e-2 sin 0.0000 exp sin 0.000e0 
after: nan 

before: sqrt + 0 + sin 0.0000 + sin 0.000 ~ ~ * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00e0 
after: nan 

before: sqrt * 1000.e-3 ~ ~ ~ ~ ~ ~ / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 
after: nan 

before: sqrt * + * 1.00e0 * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 0.00e0 exp sin 0.00000000e0 
after: nan 

before: sqrt * 1.000e0 / + * / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 100.0e-2 sin 0.0000 exp sin 0.000e0 
after: nan 

before: sqrt - exp ~ ~ * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 1.00000 
after: nan 

before: sqrt + -1.000 exp ~ ~ ~ ~ / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 
after: nan 

before: sqrt + * -1 1.000e0 exp * 1.00e0 * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 
after: nan 

before: sqrt + exp * / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 100.0e-2 -100e-2 
after: nan 

before: ~ * -1.000e0 + * -1 1.000e0 exp * 1.00e0 * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 
after: -1.000e0 

before: ~ * + exp * / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 100.0e-2 -100e-2 -100e-2 
after: -100e-2 

before: ~ + sin 0.000e0 - exp ~ ~ * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 1.00000 
after: 1.00000 

before: ~ + -1.000 exp ~ ~ ~ ~ / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 
after: 1 

before: / * sin cos + x y ~ + sin 0.000e0 - exp ~ ~ * * ~ exp + inf tanh - cos + x y cos + x y 1 + 1.000 0.00 1.00000 0.000e0 
after: nan 

before: / * ~ + -1.000 exp ~ ~ ~ ~ / + ~ + sin sin - sin sin + x y sin sin + x y exp exp exp inf 0.0000e0 1.0000 tanh tanh x sin 0.00000 
after: nan 

before: * 1e500 + 3.444 cos 1.22 
after: inf 

before: * -2e390 + + 3.444 cos 1.22 + 3.444 sin 1.22 
after: -inf 
'''
print(before==after)
