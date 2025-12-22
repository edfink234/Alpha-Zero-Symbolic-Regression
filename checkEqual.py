after='''before: x1 x1 - 0 - x1 x1 + - 
after: x1 x1 + ~ 

before: 2.33 1.222 - 0 - x1 - 
after: 1.108 x1 - 

before: 0 x - 0 0 y - - + 
after: x ~ y + 

before: x 0 - 0 y 0 - - + 
after: x y ~ + 

before: 3 0 - 0 4 0 - - + cos 
after: 0.5403023058681398 

before: x0 cos x0 x0 sin ~ * - x0 cos x0 cos * / x0 x0 cos / sech x0 x0 cos / sech * * 1 x0 x0 cos / tanh x0 x0 cos / tanh * - sqrt / ~ x0 x0 cos / tanh acos sin ~ * 
after: x0 cos x0 x0 sin ~ * - x0 cos x0 cos * / x0 x0 cos / sech x0 x0 cos / sech * * 1 x0 x0 cos / tanh x0 x0 cos / tanh * - sqrt / ~ x0 x0 cos / tanh acos sin ~ * 

before: x x + 
after: x x + 

before: x x x - + 
after: x 

before: x x - x - y + 
after: x ~ y + 

before: y y x / * cos y + 
after: y y x / * cos y + 

before: y y x * * cos y + 
after: y y x * * cos y + 

before: y x x * + 
after: y x x * + 

before: y x x + + 
after: y x x + + 

before: y x cos x + + 
after: y x cos x + + 

before: y x cos x + - 
after: y x cos x + - 

before: y x - 
after: y x - 

before: x y x - cos cos * 
after: x y x - cos cos * 

before: x x y x - sin / + 
after: x x y x - sin / + 

before: x x y y sin cos * / / 
after: x x y y sin cos * / / 

before: x ~ ~ sin y / 
after: x sin y / 

before: x sqrt 
after: x sqrt 

before: x sqrt y * 
after: x sqrt y * 

before: x ln y * 
after: x ln y * 

before: x ~ ln x * 
after: x ~ ln x * 

before: x sqrt ln y * 
after: x sqrt ln y * 

before: x x * asin 
after: x x * asin 

before: x ln y * asin 
after: x ln y * asin 

before: x ln y * asin 
after: x ln y * asin 

before: x acos y / asin 
after: x acos y / asin 

before: x ln y * asin y acos + 
after: x ln y * asin y acos + 

before: x acos acos x ~ * acos 
after: x acos acos x ~ * acos 

before: x exp x cos exp / 
after: x exp x cos exp / 

before: x ~ exp x x y * * + 
after: x ~ exp x x y * * + 

before: y arcsin exp x ~ * acos 
after: y arcsin exp x ~ * acos 

before: x y ^ 
after: x y ^ 

before: x cos y cos ^ x * 
after: x cos y cos ^ x * 

before: x cos y cos ^ x * 
after: x cos y cos ^ x * 

before: x x ^ x ^ y * 
after: x x ^ x ^ y * 

before: x x ^ x ^ y * 
after: x x ^ x ^ y * 

before: x sech tanh x ^ y * 
after: x sech tanh x ^ y * 

before: x y / tanh x sin ^ x * 
after: x y / tanh x sin ^ x * 

before: x sin sech x y * ^ sin sin sech 
after: x sin sech x y * ^ sin sin sech 

before: x ln arccos x y * / sech ~ sin 
after: x ln arccos x y * / sech ~ sin 

before: 0 x * 
after: 0 

before: 0 x * x x sin + - 
after: x x sin + ~ 

before: 0 x * ~ x tanh + 
after: x tanh 

before: 1 x * 
after: x 

before: 1 x * x x sin + - 
after: x x x sin + - 

before: 1 x * ~ x tanh + 
after: x ~ x tanh + 

before: x 0 * 
after: 0 

before: x 0 * x x sin + - 
after: x x sin + ~ 

before: x 0 * ~ x tanh + 
after: x tanh 

before: x x * 1 * 
after: x x * 

before: x x sin + 1 * 
after: x x sin + 

before: x 1 * tanh ~ 1 * 1 + 
after: x tanh ~ 1 + 

before: x sin x sin - x sin + 
after: x sin 

before: x 1 / 
after: x 

before: x x * 1 / 
after: x x * 

before: x x cos * 1 / 
after: x x cos * 

before: 0 x x * / 
after: 0 

before: 0 x x cos * / 
after: 0 

before: 0 x sin x sech * / 
after: 0 

before: 1 x x * / 
after: 1 x x * / 

before: 1 x cos / 
after: 1 x cos / 

before: 1 x cos x sin * / 
after: 1 x cos x sin * / 

before: x x ~ ~ sin + 
after: x x sin + 

before: x ~ ~ tanh x - 
after: x tanh x - 

before: x 0 x ^ + 
after: nan 

before: 0 x ^ x - 
after: nan 

before: x cos 0 x ^ - 
after: nan 

before: x x 0 ^ + 
after: x 1 + 

before: x 0 ^ x - 
after: 1 x - 

before: x cos x 0 ^ - 
after: x cos 1 - 

before: x 1 x ^ + 
after: x 1 + 

before: 1 x ^ x - 
after: 1 x - 

before: x cos 1 x ^ - 
after: x cos 1 - 

before: x x 1 ^ + 
after: x x + 

before: x 1 ^ x - 
after: 0 

before: x cos x 1 ^ - 
after: x cos x - 

before: 1 x exp * ln 
after: x 

before: x 1 x exp * ln - 
after: 0 

before: x 1 x exp * ln - cos 
after: 1 

before: y y * exp ln 
after: y y * 

before: x x * exp y ln * 
after: x x * exp y ln * 

before: y y - exp exp x sin + 
after: 2.718281828459045 x sin + 

before: x x * sin y / 
after: x x * sin y / 

before: x cos sin y / 
after: x cos sin y / 

before: x x - sqrt cos 
after: 1 

before: x x * x x * - sqrt tanh sin 
after: 0 

before: x x cos * x x cos * - sqrt sqrt 
after: 0 

before: x x - arcsin cos 
after: 1 

before: x x ^ x x ^ - asin tanh sin 
after: 0 

before: x x sin * x x sin * - arcsin asin 
after: 0 

before: x tanh x tanh - acos exp 
after: 4.810477380965351 

before: x x / x x / - arccos sech sech 
after: 0.9255209913302057 

before: x x sech - x x sech - - arccos acos 
after: nan 

before: x x exp * x x exp * - tanh acos 
after: 1.5707963267948966 

before: x x exp * x x exp * - sech asin 
after: 1.5707963267948966 

before: x x sech - x x sech - - sech acos 
after: 0 

before: x0 x0 cos / tanh acos cos 
after: x0 x0 cos / tanh 

before: 0 x tanh tanh * ~ x + 
after: x 

before: y 1 * x2 1 * * 
after: y x2 * 

before: 0 y + 0 x2 + * 
after: y x2 * 

before: x 0 + 0 y + * 
after: x y * 

before: x3 0 + 0 y + / 
after: x3 y / 

before: 0 x3 / 1 y / / 
after: 0 

before: 0 w / y 1 / / 
after: 0 

before: 1 w / y 1 / / exp ln ln exp ln ln exp 
after: 1 w / y / ln 

before: 1 w / y 1 / / asin sin sin asin sin sin arcsin 
after: 1 w / y / sin 

before: 1 w / y 1 / / arccos cos cos acos cos cos acos 
after: 1 w / y / cos 

before: 1 w / y 1 / / sin asin asin sin asin asin sin 
after: 1 w / y / asin 

before: 1 w / y 1 / / cos acos acos cos acos arccos cos 
after: 1 w / y / acos 

before: 0 x 0 x x x + + + + + 
after: x x x x + + + 

before: x x + cos cos sin tanh 0 - 
after: x x + cos cos sin tanh 

before: x x + cos cos sin tanh x x + cos cos sin tanh - 
after: 0 

before: 1 w / y 1 / / cos acos acos cos acos arccos cos 0 * 
after: 0 

before: x x ^ x x ^ - asin tanh sin x x - * 
after: 0 

before: 0 x x + sin * 
after: 0 

before: 0 y x x + tanh - * 
after: 0 

before: 1 x x + tanh x * * 
after: x x + tanh x * 

before: x 1 x x + asin x * * * 
after: x x x + asin x * * 

before: x 1 x x + asin x * * * ~ 0 / 
after: nan 

before: x x ^ x x ^ - asin tanh sin x x - * 0 / 
after: nan 

before: 1 x x + tanh x * * 1 x x + tanh x * * / 
after: 1 

before: x x * 1 x x + tanh x * * 1 x x + tanh x * * / * 
after: x x * 

before: x 1 x x + asin x * * * 0 ^ 
after: 1 

before: x x * 1 x x + tanh 0 ^ x * * 1 x x + tanh x * * / * 
after: x x * x x x + tanh x * / * 

before: 0 x x * 1 x x + tanh 0 ^ x * * 1 x x + tanh x * * / * ^ 
after: nan 

before: 0 x x ^ x x ^ - asin tanh sin x x - * 0 / ^ 
after: nan 

before: 0 x 0 x x x + + + + + 1 ^ 
after: x x x x + + + 

before: x x ^ x x ^ - asin tanh sin x x - * 0 / 1 ^ 
after: nan 

before: 1 x 1 x x + asin x * * * ^ 
after: 1 

before: 1 1 w / y 1 / / cos acos acos cos acos arccos cos ^ 
after: 1 

before: 0 x x ^ x x ^ - asin tanh sin x x - * 0 / ^ cos 
after: nan 

before: 0 x x * 1 x x + tanh 0 ^ x * * 1 x x + tanh x * * / * ^ cos 
after: nan 

before: 1 x x * 1 x x + tanh 0 ^ x * * 1 x x + tanh x * * / * ^ ~ cos 
after: 0.5403023058681398 

before: 1 x x ^ x x ^ - asin tanh ~ cos x x - * 0 / ^ cos 
after: nan 

before: 1 x x ^ x x ^ - asin tanh ~ cos x x - * 0 / ^ 1 - sin 
after: nan 

before: 0 x x * 1 x x + tanh 0 ^ x * * 1 x x + tanh x * * / * ^ cos 1 - sin 
after: nan 

before: 1 x x ^ x x ^ - asin tanh ~ cos x x - * 0 / ^ 1 - sin tanh 
after: nan 

before: 0 x x * 1 x x + tanh 0 ^ x * * 1 x x + tanh x * * / * ^ cos 1 - tanh 
after: nan 

before: 0 x x * 1 x x + tanh 0 ^ x * * 1 x x + tanh x * * / * ^ cos 1 - tanh sech 
after: nan 

before: 0 x x * 1 x x + tanh 0 ^ x * * 1 x x + tanh x * * / * ^ cos 1 - sech 
after: nan 

before: x x ^ x x ^ - asin tanh sin x x - * 0 / 1 ^ tanh 
after: nan 

before: x x + x x - - asin tanh sin x x - * 0 / tanh 
after: nan 

before: x x + x x - - asin tanh sin x x - * ~ 0 / tanh 
after: nan 

before: x x ^ x x ^ - asin tanh sin x x - * ~ 0 / 1 ^ tanh 
after: nan 

before: inf ~ tanh 
after: -1 

before: 0 inf - tanh 
after: -1 

before: x x ^ x x ^ - asin tanh sin x x - * 0 / 1 ^ sech 
after: nan 

before: x x + x x - - asin tanh sin x x - * 0 / tanh 0 / sech 
after: nan 

before: x x ^ x x ^ - asin tanh sin x x - * 0 / 1 ^ ~ sech 
after: nan 

before: x x + x x - - asin tanh sin x x - * 0 / tanh 0 / ~ sech 
after: nan 

before: nan x + x nan - - asin tanh sin x nan - * 0 / tanh 0 / ~ sech 
after: nan 

before: x nan ^ nan x ^ - asin tanh sin x nan - * 0 / 1 ^ ~ sech 
after: nan 

before: x nan ^ nan x ^ - asin tanh sin x nan - * 0 / 1 ^ ~ sech 
after: nan 

before: x1 10.000000 * 43.47847366333008 - 185.5887837532312 / 0.4000400020000667 x1 sin x0 sin * 1.5707963267948966 * - + 
after: x1 10.000000 * 43.47847366333008 - 185.5887837532312 / 0.4000400020000667 x1 sin x0 sin * 1.5707963267948966 * - + 

before: 1 x x * + 0 / ~ 
after: nan 

before: 1 x sin x * + 0 / ~ 
after: nan 

before: 1 x x * + 1 x x * + - exp 
after: 1 

before: 1 x sin x * + 1 x sin x * + - exp 
after: 1 

before: 1 x x * + 0 / exp 
after: nan 

before: 1 x sin x * + 0 / exp 
after: nan 

before: 1 x x * + 0 / ~ exp 
after: nan 

before: 1 x sin x * + 0 / ~ exp 
after: nan 

before: 1 x x * + 0 / ~ cos sin 
after: nan 

before: 1 x sin x * + 0 / ~ exp tanh 
after: nan 

before: 0 1 x x * + ^ ~ cos sin 
after: nan 

before: 0 1 x x tanh * + ^ ~ cos sin 
after: nan 

before: x 0 1 x x * + ^ ~ cos sin + 
after: nan 

before: x x * 0 1 x x tanh * + ^ ~ cos sin + 
after: nan 

before: z cos 0 1 x x * + ^ ~ cos sin - 
after: nan 

before: x x + 0 1 x x tanh * + ^ ~ cos sin - 
after: nan 

before: z sech tanh 0 1 x x * + ^ ~ cos sin * 
after: nan 

before: x x + sin 0 1 x x tanh * + ^ ~ cos sin * 
after: nan 

before: y tanh sin asin 0 1 x x * + ^ ~ cos sin / 
after: nan 

before: x x / sech arcsin 0 1 x x tanh * + ^ ~ cos sin / 
after: nan 

before: y sech sech sech 0 1 x x * + ^ ~ cos sin ^ 
after: nan 

before: x x ^ sech arcsin 0 1 x x tanh * + ^ ~ cos sin ^ 
after: nan 

before: y sech sech sech 0 1 x x * + ^ ~ cos sin ^ cos 
after: nan 

before: x x ^ sech arcsin cos tanh 0 1 x x tanh * + ^ ~ cos sin ^ cos 
after: nan 

before: y sech sech sech 0 1 0 - x x * + ^ ~ cos sin ^ sin 
after: nan 

before: x x ^ sech arcsin cos tanh 0 1 x x + x tanh * + ^ ~ cos sin ^ sin 
after: nan 

before: y y * sin sech sech sech 0 1 0 - x x * + ^ ~ cos sin ^ sin tanh 
after: nan 

before: 1 x x ^ ^ sech arcsin cos tanh 0 1 x x + x tanh * + ^ ~ cos sin ^ sin tanh 
after: nan 

before: y y 2 ^ * sin sech sech sech 0 1 0 - x x * + ^ ~ cos sin ^ sin sech 
after: nan 

before: 1 x x ^ ^ sech arcsin cos tanh 0 0 - 1 x x + x + x tanh * + ^ ~ cos sin ^ sin sech 
after: nan 

before: y y 2 1 + ^ * sin sech sech sech 0 1 0 - x x * + ^ ~ cos sin ^ sin exp 
after: nan 

before: 1 x x - ^ sech arcsin cos tanh 0 0 - 1 x x + x + x tanh * + ^ ~ cos sin ^ sin exp 
after: nan 

before: y y 2 1 x + + ^ * sin tanh sech sech 0 1 0 - x x * + ^ ~ cos sin ^ sin ~ 
after: nan 

before: 1 2 - x y / ^ sech arccos cos tanh 0 0 - 1 x x + x + x tanh * + ^ ~ cos sin ^ sin ~ 
after: nan 

before: x x + 0.00 + 
after: x x + 

before: x x + sin 0.00e0 - 
after: x x + sin 

before: 0.0000000 x x + + 
after: x x + 

before: 0.0000e0 x x + sin - 
after: x x + sin ~ 

before: x x + -0.000 * 
after: 0 

before: x x + sin tanh -0.00e0 * 
after: 0 

before: 0.0000000e0 x x + * 
after: 0 

before: -0.0000e0 x x + sin cos * 
after: 0 

before: x x * 1.000 * 
after: x x * 

before: x x + sin tanh 10000e-4 * 
after: x x + sin tanh 

before: 1.0000000e0 x x tanh + * 
after: x x tanh + 

before: 100000e-5 x x + sin asin * 
after: x x + 

before: 0.0000000e0 x x + * 0.00e0 / 
after: nan 

before: -0. sin -0.0000e0 x x + sin cos * / 
after: nan 

before: x x / 0.00000e0 x x + * / 
after: nan 

before: x asin arccos acos -0.0000e0 x x + sin cos * / 
after: nan 

before: 0.0000000e0 x x + * apple / 
after: 0 

before: -0.0000e0 x x + sin cos * appleOrange sin cos tanh / 
after: 0 

before: x x * sin 1.000 / 
after: x x * sin 

before: x yay + sin tanh 10000e-4 / 
after: x yay + sin tanh 

before: x x + -0.00000e0 x x + * ^ 
after: 1 

before: x asin arccos acos -0.00e0 x x + sech cos * ^ 
after: 1 

before: 0.0000000e0 x x + ^ 
after: nan 

before: -0.0000e0 x x1 + sin cos ^ 
after: nan 

before: x x + x x / ^ 
after: x x + 

before: x asin arccos acos x sin x sin / ^ 
after: x asin arccos acos 

before: 1 x x + ^ 
after: 1 

before: 1 -0.00e0 + x x1 + sin cos ^ 
after: 1 

before: 1.000e0 1 x x + ^ - cos 
after: 1 

before: 10e-1 1 -0.00e0 + x x1 + sin cos ^ - cos 
after: 1 

before: 1.0e0 1.000e0 1 x x + ^ - cos - sin 
after: 0 

before: 100.e-2 10e-1 1 -0.00e0 + x x1 + sin tanh ^ - cos - sin 
after: 0 

before: 1.0e0 1.000e0 1 x x + ^ - cos - sin tanh 
after: 0 

before: 100.e-2 10e-1 1 -0.00e0 + x x1 + sin tanh ^ - cos - sin sin tanh 
after: 0 

before: 1.0e0 1.000e0 1 x x + ^ - cos - sin tanh tanh sech 
after: 1 

before: 100.e-2 10e-1 1 -0.00e0 + x x1 + sin tanh ^ - cos - sin tanh tanh sech 
after: 1 

before: 1.0e0 1.000e0 1 x x + ^ - cos - sin tanh sin ~ 
after: 0 

before: 100.e-2 10e-1 1 -0.00e0 + x x1 + sin tanh ^ - cos - sin sin tanh tanh ~ 
after: 0 

before: 1.0e0 1.000e0 1 x x + ^ - cos - sin tanh sin ~ exp 
after: 1 

before: 100.e-2 10e-1 1 -0.00e0 + x x1 + sin tanh ^ - cos - sin sin tanh tanh ~ exp 
after: 1 

before: -0.00e0 xxxx - 
after: xxxx ~ 

before: 0.000e0 appleApple - 
after: appleApple ~ 

before: xxasdfasdfxx -0.000000e0 - 
after: xxasdfasdfxx 

before: appleApple123 0.000000e0 - 
after: appleApple123 

before: -0.00e0 xxxasdfx * 
after: 0 

before: 0.000000e0 appleApple24523 * 
after: 0 

before: xxasdfasdfxx -0.00000000e0 * 
after: 0 

before: appleApple123 -0.000000 * 
after: 0 

before: 100e-2 xxxasdfx * 
after: xxxasdfx 

before: 1000000e-6 appleApple24523 * 
after: appleApple24523 

before: xxasdfasdfxx 1.00000000 * 
after: xxasdfasdfxx 

before: appleApple123 100000000e-8 * 
after: appleApple123 

before: -0.00e0 xxxasdfx + 
after: xxxasdfx 

before: 0.00000e0 appleApple24523 + 
after: appleApple24523 

before: xxasdfasdfxx -0.00000000000e0 + 
after: xxasdfasdfxx 

before: appleApple1asdsq23 0.000000e-0 + 
after: appleApple1asdsq23 

before: -0.000 xxxasdfx / 
after: 0 

before: 0.00000000e-00 appleApple24523 / 
after: 0 

before: xxasdfasdfxx 1.00000000 / 
after: xxasdfasdfxx 

before: appleApple123 10000000e-07 / 
after: appleApple123 

before: xxasdfasdfxx -0.000000000e0 ^ 
after: 1 

before: appleApple1asdsq23 0.0000e-0 ^ 
after: 1 

before: -0.000000e00 xxxasdfx ^ 
after: nan 

before: 0.00000000e-0000 x23 ^ 
after: nan 

before: xxasdfasdfxx 1000000000e-09 ^ 
after: xxasdfasdfxx 

before: appleApple1asdsq23 1.0000e-0 ^ 
after: appleApple1asdsq23 

before: 1.000000 xxxasdfx ^ 
after: 1 

before: 1.0 x23 ^ 
after: 1 

before: 9736 x7 - -100.051731 -0.000000 x15 ^ / + x20 1075.000000 x5 * + -17064.107062 * + 
after: nan 

before: 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - 
after: 0.010000 x1 + sin x0 sin * ~ x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ + - 

before: 0.000 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - ^ 
after: nan 

before: sin tanh 0.000000e0 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - ^ 
after: nan 

before: 0.000 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * sin ~ 
after: 0 

before: sin tanh 0.000000e0 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * ~ 
after: 0 

before: inf 0.000 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * sin ~ + ~ 
after: -inf 

before: sin tanh 0.000000e0 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * ~ inf exp + ~ 
after: -inf 

before: x y + cos sin ~ sech 
after: x y + cos sin sech 

before: z apple sech + tanh ~ sech 
after: z apple sech + tanh sech 

before: x y + cos sin ~ sech 0.000 sin / 
after: nan 

before: z apple sech + tanh ~ sech 0 0.00e0 + tanh sin sin / 
after: nan 

before: inf 0.000 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * sin ~ + cos 
after: nan 

before: sin tanh 0.000000e0 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * ~ inf exp + exp ln log cos 
after: nan 

before: inf 0.000 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * sin ~ + exp ~ cos 
after: nan 

before: sin tanh 0.000000e0 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * ~ inf exp + exp ln log ~ ~ ~ cos 
after: nan 

before: inf 0.000 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * sin ~ + exp ~ ~ sin 
after: nan 

before: sin tanh 0.000000e0 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * ~ inf exp + exp ln log ~ ~ sin 
after: nan 

before: inf 0.000 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * sin ~ + exp ~ ~ ~ sin 
after: nan 

before: sin tanh 0.000000e0 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * ~ inf exp + exp ln log ~ sin 
after: nan 

before: x y + cos sin ~ sech 0 1 - * 
after: x y + cos sin sech ~ 

before: x y + cos cos sech sin ~ sech 0 1.0000e0 - * 
after: x y + cos cos sech sin sech ~ 

before: -1.000e0 x y + cos x y + sech + tanh * 
after: x y + cos x y + sech + tanh ~ 

before: -1000.0e-3 x x x + + sech * 
after: x x x + + sech ~ 

before: x y + cos sin ~ sech 00.000e0 1 - / 
after: x y + cos sin sech ~ 

before: x y + cos cos sech sin ~ sech 0000 1.0000e0 - / 
after: x y + cos cos sech sin sech ~ 

before: 0.000 sinx y + cos sin ~ sech 00.000e0 1 - / * log 
after: -inf 

before: 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / ln 
after: -inf 

before: 0.000e0 x y + cos sin ~ sech 00.000e0 1 - * ^ log 
after: nan 

before: nan 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / ^ ln 
after: nan 

before: inf exp x y + cos sin ~ sech 00.000e0 1 - * + log 
after: inf x y + cos sin sech ~ + log 

before: inf 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln 
after: inf 

before: 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin 
after: 0 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 
after: 0 

before: nan 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh + asin 
after: nan 

before: 0.000e0 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln ^ arcsin 
after: nan 

before: inf 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + asin 
after: nan 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 1.00 inf ln log * + arcsin 
after: nan 

before: 1.000e0 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos 
after: 0 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 1.00 1 * + arccos 
after: 0 

before: nan 1.000e0 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos * acos 
after: nan 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 1.00 nan * + arccos 
after: nan 

before: inf 1.000e0 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos + acos 
after: nan 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 1.00 inf ln * + arccos 
after: nan 

before: 100e-2 1.000e0 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt 
after: 1 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 
after: 1 

before: 100e-2 1.000e0 0.000 x ^ x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt 
after: nan 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0.00e0 * 0000 1.0000e0 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 
after: nan 

before: inf 100e-2 1.000e0 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt * sqrt 
after: inf 

before: 0 ln ~ 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt * sqrt 
after: inf 

before: 1 100e-2 1.000e0 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt 
after: 0 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt 
after: 0 

before: 1 100e-2 1.000e0 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln cos 
after: nan 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt 0 + log cos 
after: nan 

before: 1 100e-2 1.000e0 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln 1000.0e-4 * cos 
after: nan 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt inf - cos 
after: nan 

before: 1 1000e-3 1.000e0 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln sin 
after: nan 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt 0.00000e0 + log sin 
after: nan 

before: 1 100e-2 1.000e0 0.000 exp 0 * x y + cos sech sin ~ ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln 1000.0e-4 * sin 
after: nan 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 10000e-4 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt inf - sin 
after: nan 

before: 1 1000e-3 1.000e0 0.000e0 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln ~ 
after: inf 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt 0.000000000e0 + log ~ 
after: inf 

before: 1 100e-2 1.000e0 0.000 exp 0 * x y + cos sech sin ~ ~ ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln 1000.0e-4 * ~ 
after: inf 

before: 100e-2 0.000e0 x y + cos cos tanh sin ~ sech 0000 10000e-4 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt inf - ~ 
after: inf 

before: 1 1000e-3 1.000e0 0.000e0 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln ~ ~ ln 
after: nan 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 100e-2 0.0000e0 exp * + sqrt 0.000 exp - sqrt 0.000000000e0 + log ~ ~ ~ ~ log 
after: nan 

before: 1 100e-2 1.000e0 0.000 exp 0 * x y + cos sech sin ~ ~ ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln 1000.0e-4 * ~ ~ log 
after: nan 

before: 100e-2 0.000e0 x y + 1 * cos cos tanh sin ~ sech 0000 100000.e-4 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt inf - ~ ~ ln 
after: nan 

before: 1 1000e-3 1.000e0 0.000e0 exp ln exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln ~ ~ asin 
after: nan 

before: 1000e-3 0 + 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 100e-2 0.0000e0 exp * + sqrt 0.000 exp - sqrt 0.000000000e0 + log ~ ~ ~ ~ arcsin 
after: nan 

before: 1 0.000e0 + 100e-2 1.000e0 0.000 exp 0 * x y + cos sech sin ~ ~ ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln 1000.0e-4 * ~ ~ asin 
after: nan 

before: 100000.e-5 0. + 0.000e0 x y + 1 * cos cos tanh sin ~ sech 0000 100000.e-4 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt inf - ~ ~ arcsin 
after: nan 

before: 1 1000e-3 1.000e0 0.000e0 exp ln exp 0 * x y + cos ln exp sin ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln ~ ~ acos 
after: nan 

before: 1000e-3 0 + 0.000e0 x y + cos cos sech sin ~ ~ ~ sech 0000 1.0000e0 - / / + ln arcsin 100e-2 0.0000e0 exp * + sqrt 0.000 exp - sqrt 0.000000000e0 + log ~ ~ ~ ~ arccos 
after: nan 

before: 1 0.000e0 asin + 100e-2 1.000e0 0.000 exp 0 * x y + cos sech sin ~ ~ ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln 1000.0e-4 * ~ ~ ~ ~ acos 
after: nan 

before: 1000000.e-6 0. + 0.000e0 x 1.000e0 * y + 1 * cos cos tanh sin ~ sech 0000 100000.e-4 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt inf - ~ ~ arccos 
after: nan 

before: 1.000000 1000e-3 1.000e0 0 + 0.000e0 exp ln exp 0 * x y + cos ln exp sin ~ sech 00.000e0 1000e-3 - * * tanh asin + arccos + sqrt - sqrt ln ~ ~ sqrt 
after: nan 

before: 1000e-3 0 + 0.000e0 x yessss 1 * + cos cos sech sin ~ ~ ~ sech 0000 1.0000e0 - / / + ln arcsin 100e-2 0.0000e0 exp * + sqrt 0.000 exp - sqrt 0.000000000e0 + log ~ ~ ~ ~ sqrt 
after: nan 

before: 1 0.000e0 ~ ~ asin + 100e-2 1.000e0 0.000 exp 0 * x y + cos sech sin ~ ~ ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln 1000.0e-4 * ~ ~ ~ ~ sqrt 
after: nan 

before: 1000000.e-6 0. + 0.000e0 x ln 1.000e0 * y + 1 * cos cos tanh sin ~ sech 0000 100000.e-4 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt inf - ~ ~ sqrt 
after: nan 

before: 1000000.e-6 0. + 0.000e0 x ln 1.000e0 * y + 1 * cos cos tanh sin ~ sech 0000 100000.e-4 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt -1 - ~ 
after: -1 

before: 1000000.e-6 0. + 0.000e0 x ln 1.000e0 * y + 1 * cos cos tanh sin ~ sech 0000 100000.e-4 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt -100e-2 - ~ ~ ~ 
after: -100e-2 

before: 1000000.e-6 0. + 0.0000000e0 xasdfasdf ln 1.000e0 * y + 1 * cos cos tanh sin ~ sech 0000 100000.e-4 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt -1 - ~ ~ 
after: 1 

before: 1000000.e-6 0. + 0.000e0 x x + xxx + ln 1.000e0 * y + 1 * cos cos tanh sin ~ sech 0000 100000.e-4 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt -100e-2 - ~ ~ ~ ~ 
after: 1 

before: 0.7911530997475994 x0 x0 cos - sech sqrt 1.048576e+06 0 x0 10.000000 - ^ tanh 1.0001 / arccos log ~ 4 / cos cos ^ * ^ 
after: nan 

before: 1e500 3.444 1.22 cos + * 
after: inf 

before: -2e390 3.444 1.22 cos + 3.444 1.22 sin + + * 
after: -inf 
'''
before='''before: x1 x1 - 0 - x1 x1 + - 
after: x1 x1 + ~ 

before: 2.33 1.222 - 0 - x1 - 
after: 1.108 x1 - 

before: 0 x - 0 0 y - - + 
after: x ~ y + 

before: x 0 - 0 y 0 - - + 
after: x y ~ + 

before: 3 0 - 0 4 0 - - + cos 
after: 0.5403023058681398 

before: x0 cos x0 x0 sin ~ * - x0 cos x0 cos * / x0 x0 cos / sech x0 x0 cos / sech * * 1 x0 x0 cos / tanh x0 x0 cos / tanh * - sqrt / ~ x0 x0 cos / tanh acos sin ~ * 
after: x0 cos x0 x0 sin ~ * - x0 cos x0 cos * / x0 x0 cos / sech x0 x0 cos / sech * * 1 x0 x0 cos / tanh x0 x0 cos / tanh * - sqrt / ~ x0 x0 cos / tanh acos sin ~ * 

before: x x + 
after: x x + 

before: x x x - + 
after: x 

before: x x - x - y + 
after: x ~ y + 

before: y y x / * cos y + 
after: y y x / * cos y + 

before: y y x * * cos y + 
after: y y x * * cos y + 

before: y x x * + 
after: y x x * + 

before: y x x + + 
after: y x x + + 

before: y x cos x + + 
after: y x cos x + + 

before: y x cos x + - 
after: y x cos x + - 

before: y x - 
after: y x - 

before: x y x - cos cos * 
after: x y x - cos cos * 

before: x x y x - sin / + 
after: x x y x - sin / + 

before: x x y y sin cos * / / 
after: x x y y sin cos * / / 

before: x ~ ~ sin y / 
after: x sin y / 

before: x sqrt 
after: x sqrt 

before: x sqrt y * 
after: x sqrt y * 

before: x ln y * 
after: x ln y * 

before: x ~ ln x * 
after: x ~ ln x * 

before: x sqrt ln y * 
after: x sqrt ln y * 

before: x x * asin 
after: x x * asin 

before: x ln y * asin 
after: x ln y * asin 

before: x ln y * asin 
after: x ln y * asin 

before: x acos y / asin 
after: x acos y / asin 

before: x ln y * asin y acos + 
after: x ln y * asin y acos + 

before: x acos acos x ~ * acos 
after: x acos acos x ~ * acos 

before: x exp x cos exp / 
after: x exp x cos exp / 

before: x ~ exp x x y * * + 
after: x ~ exp x x y * * + 

before: y arcsin exp x ~ * acos 
after: y arcsin exp x ~ * acos 

before: x y ^ 
after: x y ^ 

before: x cos y cos ^ x * 
after: x cos y cos ^ x * 

before: x cos y cos ^ x * 
after: x cos y cos ^ x * 

before: x x ^ x ^ y * 
after: x x ^ x ^ y * 

before: x x ^ x ^ y * 
after: x x ^ x ^ y * 

before: x sech tanh x ^ y * 
after: x sech tanh x ^ y * 

before: x y / tanh x sin ^ x * 
after: x y / tanh x sin ^ x * 

before: x sin sech x y * ^ sin sin sech 
after: x sin sech x y * ^ sin sin sech 

before: x ln arccos x y * / sech ~ sin 
after: x ln arccos x y * / sech ~ sin 

before: 0 x * 
after: 0 

before: 0 x * x x sin + - 
after: x x sin + ~ 

before: 0 x * ~ x tanh + 
after: x tanh 

before: 1 x * 
after: x 

before: 1 x * x x sin + - 
after: x x x sin + - 

before: 1 x * ~ x tanh + 
after: x ~ x tanh + 

before: x 0 * 
after: 0 

before: x 0 * x x sin + - 
after: x x sin + ~ 

before: x 0 * ~ x tanh + 
after: x tanh 

before: x x * 1 * 
after: x x * 

before: x x sin + 1 * 
after: x x sin + 

before: x 1 * tanh ~ 1 * 1 + 
after: x tanh ~ 1 + 

before: x sin x sin - x sin + 
after: x sin 

before: x 1 / 
after: x 

before: x x * 1 / 
after: x x * 

before: x x cos * 1 / 
after: x x cos * 

before: 0 x x * / 
after: 0 

before: 0 x x cos * / 
after: 0 

before: 0 x sin x sech * / 
after: 0 

before: 1 x x * / 
after: 1 x x * / 

before: 1 x cos / 
after: 1 x cos / 

before: 1 x cos x sin * / 
after: 1 x cos x sin * / 

before: x x ~ ~ sin + 
after: x x sin + 

before: x ~ ~ tanh x - 
after: x tanh x - 

before: x 0 x ^ + 
after: nan 

before: 0 x ^ x - 
after: nan 

before: x cos 0 x ^ - 
after: nan 

before: x x 0 ^ + 
after: x 1 + 

before: x 0 ^ x - 
after: 1 x - 

before: x cos x 0 ^ - 
after: x cos 1 - 

before: x 1 x ^ + 
after: x 1 + 

before: 1 x ^ x - 
after: 1 x - 

before: x cos 1 x ^ - 
after: x cos 1 - 

before: x x 1 ^ + 
after: x x + 

before: x 1 ^ x - 
after: 0 

before: x cos x 1 ^ - 
after: x cos x - 

before: 1 x exp * ln 
after: x 

before: x 1 x exp * ln - 
after: 0 

before: x 1 x exp * ln - cos 
after: 1 

before: y y * exp ln 
after: y y * 

before: x x * exp y ln * 
after: x x * exp y ln * 

before: y y - exp exp x sin + 
after: 2.718281828459045 x sin + 

before: x x * sin y / 
after: x x * sin y / 

before: x cos sin y / 
after: x cos sin y / 

before: x x - sqrt cos 
after: 1 

before: x x * x x * - sqrt tanh sin 
after: 0 

before: x x cos * x x cos * - sqrt sqrt 
after: 0 

before: x x - arcsin cos 
after: 1 

before: x x ^ x x ^ - asin tanh sin 
after: 0 

before: x x sin * x x sin * - arcsin asin 
after: 0 

before: x tanh x tanh - acos exp 
after: 4.810477380965351 

before: x x / x x / - arccos sech sech 
after: 0.9255209913302057 

before: x x sech - x x sech - - arccos acos 
after: nan 

before: x x exp * x x exp * - tanh acos 
after: 1.5707963267948966 

before: x x exp * x x exp * - sech asin 
after: 1.5707963267948966 

before: x x sech - x x sech - - sech acos 
after: 0 

before: x0 x0 cos / tanh acos cos 
after: x0 x0 cos / tanh 

before: 0 x tanh tanh * ~ x + 
after: x 

before: y 1 * x2 1 * * 
after: y x2 * 

before: 0 y + 0 x2 + * 
after: y x2 * 

before: x 0 + 0 y + * 
after: x y * 

before: x3 0 + 0 y + / 
after: x3 y / 

before: 0 x3 / 1 y / / 
after: 0 

before: 0 w / y 1 / / 
after: 0 

before: 1 w / y 1 / / exp ln ln exp ln ln exp 
after: 1 w / y / ln 

before: 1 w / y 1 / / asin sin sin asin sin sin arcsin 
after: 1 w / y / sin 

before: 1 w / y 1 / / arccos cos cos acos cos cos acos 
after: 1 w / y / cos 

before: 1 w / y 1 / / sin asin asin sin asin asin sin 
after: 1 w / y / asin 

before: 1 w / y 1 / / cos acos acos cos acos arccos cos 
after: 1 w / y / acos 

before: 0 x 0 x x x + + + + + 
after: x x x x + + + 

before: x x + cos cos sin tanh 0 - 
after: x x + cos cos sin tanh 

before: x x + cos cos sin tanh x x + cos cos sin tanh - 
after: 0 

before: 1 w / y 1 / / cos acos acos cos acos arccos cos 0 * 
after: 0 

before: x x ^ x x ^ - asin tanh sin x x - * 
after: 0 

before: 0 x x + sin * 
after: 0 

before: 0 y x x + tanh - * 
after: 0 

before: 1 x x + tanh x * * 
after: x x + tanh x * 

before: x 1 x x + asin x * * * 
after: x x x + asin x * * 

before: x 1 x x + asin x * * * ~ 0 / 
after: nan 

before: x x ^ x x ^ - asin tanh sin x x - * 0 / 
after: nan 

before: 1 x x + tanh x * * 1 x x + tanh x * * / 
after: 1 

before: x x * 1 x x + tanh x * * 1 x x + tanh x * * / * 
after: x x * 

before: x 1 x x + asin x * * * 0 ^ 
after: 1 

before: x x * 1 x x + tanh 0 ^ x * * 1 x x + tanh x * * / * 
after: x x * x x x + tanh x * / * 

before: 0 x x * 1 x x + tanh 0 ^ x * * 1 x x + tanh x * * / * ^ 
after: nan 

before: 0 x x ^ x x ^ - asin tanh sin x x - * 0 / ^ 
after: nan 

before: 0 x 0 x x x + + + + + 1 ^ 
after: x x x x + + + 

before: x x ^ x x ^ - asin tanh sin x x - * 0 / 1 ^ 
after: nan 

before: 1 x 1 x x + asin x * * * ^ 
after: 1 

before: 1 1 w / y 1 / / cos acos acos cos acos arccos cos ^ 
after: 1 

before: 0 x x ^ x x ^ - asin tanh sin x x - * 0 / ^ cos 
after: nan 

before: 0 x x * 1 x x + tanh 0 ^ x * * 1 x x + tanh x * * / * ^ cos 
after: nan 

before: 1 x x * 1 x x + tanh 0 ^ x * * 1 x x + tanh x * * / * ^ ~ cos 
after: 0.5403023058681398 

before: 1 x x ^ x x ^ - asin tanh ~ cos x x - * 0 / ^ cos 
after: nan 

before: 1 x x ^ x x ^ - asin tanh ~ cos x x - * 0 / ^ 1 - sin 
after: nan 

before: 0 x x * 1 x x + tanh 0 ^ x * * 1 x x + tanh x * * / * ^ cos 1 - sin 
after: nan 

before: 1 x x ^ x x ^ - asin tanh ~ cos x x - * 0 / ^ 1 - sin tanh 
after: nan 

before: 0 x x * 1 x x + tanh 0 ^ x * * 1 x x + tanh x * * / * ^ cos 1 - tanh 
after: nan 

before: 0 x x * 1 x x + tanh 0 ^ x * * 1 x x + tanh x * * / * ^ cos 1 - tanh sech 
after: nan 

before: 0 x x * 1 x x + tanh 0 ^ x * * 1 x x + tanh x * * / * ^ cos 1 - sech 
after: nan 

before: x x ^ x x ^ - asin tanh sin x x - * 0 / 1 ^ tanh 
after: nan 

before: x x + x x - - asin tanh sin x x - * 0 / tanh 
after: nan 

before: x x + x x - - asin tanh sin x x - * ~ 0 / tanh 
after: nan 

before: x x ^ x x ^ - asin tanh sin x x - * ~ 0 / 1 ^ tanh 
after: nan 

before: inf ~ tanh 
after: -1 

before: 0 inf - tanh 
after: -1 

before: x x ^ x x ^ - asin tanh sin x x - * 0 / 1 ^ sech 
after: nan 

before: x x + x x - - asin tanh sin x x - * 0 / tanh 0 / sech 
after: nan 

before: x x ^ x x ^ - asin tanh sin x x - * 0 / 1 ^ ~ sech 
after: nan 

before: x x + x x - - asin tanh sin x x - * 0 / tanh 0 / ~ sech 
after: nan 

before: nan x + x nan - - asin tanh sin x nan - * 0 / tanh 0 / ~ sech 
after: nan 

before: x nan ^ nan x ^ - asin tanh sin x nan - * 0 / 1 ^ ~ sech 
after: nan 

before: x nan ^ nan x ^ - asin tanh sin x nan - * 0 / 1 ^ ~ sech 
after: nan 

before: x1 10.000000 * 43.47847366333008 - 185.5887837532312 / 0.4000400020000667 x1 sin x0 sin * 1.5707963267948966 * - + 
after: x1 10.000000 * 43.47847366333008 - 185.5887837532312 / 0.4000400020000667 x1 sin x0 sin * 1.5707963267948966 * - + 

before: 1 x x * + 0 / ~ 
after: nan 

before: 1 x sin x * + 0 / ~ 
after: nan 

before: 1 x x * + 1 x x * + - exp 
after: 1 

before: 1 x sin x * + 1 x sin x * + - exp 
after: 1 

before: 1 x x * + 0 / exp 
after: nan 

before: 1 x sin x * + 0 / exp 
after: nan 

before: 1 x x * + 0 / ~ exp 
after: nan 

before: 1 x sin x * + 0 / ~ exp 
after: nan 

before: 1 x x * + 0 / ~ cos sin 
after: nan 

before: 1 x sin x * + 0 / ~ exp tanh 
after: nan 

before: 0 1 x x * + ^ ~ cos sin 
after: nan 

before: 0 1 x x tanh * + ^ ~ cos sin 
after: nan 

before: x 0 1 x x * + ^ ~ cos sin + 
after: nan 

before: x x * 0 1 x x tanh * + ^ ~ cos sin + 
after: nan 

before: z cos 0 1 x x * + ^ ~ cos sin - 
after: nan 

before: x x + 0 1 x x tanh * + ^ ~ cos sin - 
after: nan 

before: z sech tanh 0 1 x x * + ^ ~ cos sin * 
after: nan 

before: x x + sin 0 1 x x tanh * + ^ ~ cos sin * 
after: nan 

before: y tanh sin asin 0 1 x x * + ^ ~ cos sin / 
after: nan 

before: x x / sech arcsin 0 1 x x tanh * + ^ ~ cos sin / 
after: nan 

before: y sech sech sech 0 1 x x * + ^ ~ cos sin ^ 
after: nan 

before: x x ^ sech arcsin 0 1 x x tanh * + ^ ~ cos sin ^ 
after: nan 

before: y sech sech sech 0 1 x x * + ^ ~ cos sin ^ cos 
after: nan 

before: x x ^ sech arcsin cos tanh 0 1 x x tanh * + ^ ~ cos sin ^ cos 
after: nan 

before: y sech sech sech 0 1 0 - x x * + ^ ~ cos sin ^ sin 
after: nan 

before: x x ^ sech arcsin cos tanh 0 1 x x + x tanh * + ^ ~ cos sin ^ sin 
after: nan 

before: y y * sin sech sech sech 0 1 0 - x x * + ^ ~ cos sin ^ sin tanh 
after: nan 

before: 1 x x ^ ^ sech arcsin cos tanh 0 1 x x + x tanh * + ^ ~ cos sin ^ sin tanh 
after: nan 

before: y y 2 ^ * sin sech sech sech 0 1 0 - x x * + ^ ~ cos sin ^ sin sech 
after: nan 

before: 1 x x ^ ^ sech arcsin cos tanh 0 0 - 1 x x + x + x tanh * + ^ ~ cos sin ^ sin sech 
after: nan 

before: y y 2 1 + ^ * sin sech sech sech 0 1 0 - x x * + ^ ~ cos sin ^ sin exp 
after: nan 

before: 1 x x - ^ sech arcsin cos tanh 0 0 - 1 x x + x + x tanh * + ^ ~ cos sin ^ sin exp 
after: nan 

before: y y 2 1 x + + ^ * sin tanh sech sech 0 1 0 - x x * + ^ ~ cos sin ^ sin ~ 
after: nan 

before: 1 2 - x y / ^ sech arccos cos tanh 0 0 - 1 x x + x + x tanh * + ^ ~ cos sin ^ sin ~ 
after: nan 

before: x x + 0.00 + 
after: x x + 

before: x x + sin 0.00e0 - 
after: x x + sin 

before: 0.0000000 x x + + 
after: x x + 

before: 0.0000e0 x x + sin - 
after: x x + sin ~ 

before: x x + -0.000 * 
after: 0 

before: x x + sin tanh -0.00e0 * 
after: 0 

before: 0.0000000e0 x x + * 
after: 0 

before: -0.0000e0 x x + sin cos * 
after: 0 

before: x x * 1.000 * 
after: x x * 

before: x x + sin tanh 10000e-4 * 
after: x x + sin tanh 

before: 1.0000000e0 x x tanh + * 
after: x x tanh + 

before: 100000e-5 x x + sin asin * 
after: x x + 

before: 0.0000000e0 x x + * 0.00e0 / 
after: nan 

before: -0. sin -0.0000e0 x x + sin cos * / 
after: nan 

before: x x / 0.00000e0 x x + * / 
after: nan 

before: x asin arccos acos -0.0000e0 x x + sin cos * / 
after: nan 

before: 0.0000000e0 x x + * apple / 
after: 0 

before: -0.0000e0 x x + sin cos * appleOrange sin cos tanh / 
after: 0 

before: x x * sin 1.000 / 
after: x x * sin 

before: x yay + sin tanh 10000e-4 / 
after: x yay + sin tanh 

before: x x + -0.00000e0 x x + * ^ 
after: 1 

before: x asin arccos acos -0.00e0 x x + sech cos * ^ 
after: 1 

before: 0.0000000e0 x x + ^ 
after: nan 

before: -0.0000e0 x x1 + sin cos ^ 
after: nan 

before: x x + x x / ^ 
after: x x + 

before: x asin arccos acos x sin x sin / ^ 
after: x asin arccos acos 

before: 1 x x + ^ 
after: 1 

before: 1 -0.00e0 + x x1 + sin cos ^ 
after: 1 

before: 1.000e0 1 x x + ^ - cos 
after: 1 

before: 10e-1 1 -0.00e0 + x x1 + sin cos ^ - cos 
after: 1 

before: 1.0e0 1.000e0 1 x x + ^ - cos - sin 
after: 0 

before: 100.e-2 10e-1 1 -0.00e0 + x x1 + sin tanh ^ - cos - sin 
after: 0 

before: 1.0e0 1.000e0 1 x x + ^ - cos - sin tanh 
after: 0 

before: 100.e-2 10e-1 1 -0.00e0 + x x1 + sin tanh ^ - cos - sin sin tanh 
after: 0 

before: 1.0e0 1.000e0 1 x x + ^ - cos - sin tanh tanh sech 
after: 1 

before: 100.e-2 10e-1 1 -0.00e0 + x x1 + sin tanh ^ - cos - sin tanh tanh sech 
after: 1 

before: 1.0e0 1.000e0 1 x x + ^ - cos - sin tanh sin ~ 
after: 0 

before: 100.e-2 10e-1 1 -0.00e0 + x x1 + sin tanh ^ - cos - sin sin tanh tanh ~ 
after: 0 

before: 1.0e0 1.000e0 1 x x + ^ - cos - sin tanh sin ~ exp 
after: 1 

before: 100.e-2 10e-1 1 -0.00e0 + x x1 + sin tanh ^ - cos - sin sin tanh tanh ~ exp 
after: 1 

before: -0.00e0 xxxx - 
after: xxxx ~ 

before: 0.000e0 appleApple - 
after: appleApple ~ 

before: xxasdfasdfxx -0.000000e0 - 
after: xxasdfasdfxx 

before: appleApple123 0.000000e0 - 
after: appleApple123 

before: -0.00e0 xxxasdfx * 
after: 0 

before: 0.000000e0 appleApple24523 * 
after: 0 

before: xxasdfasdfxx -0.00000000e0 * 
after: 0 

before: appleApple123 -0.000000 * 
after: 0 

before: 100e-2 xxxasdfx * 
after: xxxasdfx 

before: 1000000e-6 appleApple24523 * 
after: appleApple24523 

before: xxasdfasdfxx 1.00000000 * 
after: xxasdfasdfxx 

before: appleApple123 100000000e-8 * 
after: appleApple123 

before: -0.00e0 xxxasdfx + 
after: xxxasdfx 

before: 0.00000e0 appleApple24523 + 
after: appleApple24523 

before: xxasdfasdfxx -0.00000000000e0 + 
after: xxasdfasdfxx 

before: appleApple1asdsq23 0.000000e-0 + 
after: appleApple1asdsq23 

before: -0.000 xxxasdfx / 
after: 0 

before: 0.00000000e-00 appleApple24523 / 
after: 0 

before: xxasdfasdfxx 1.00000000 / 
after: xxasdfasdfxx 

before: appleApple123 10000000e-07 / 
after: appleApple123 

before: xxasdfasdfxx -0.000000000e0 ^ 
after: 1 

before: appleApple1asdsq23 0.0000e-0 ^ 
after: 1 

before: -0.000000e00 xxxasdfx ^ 
after: nan 

before: 0.00000000e-0000 x23 ^ 
after: nan 

before: xxasdfasdfxx 1000000000e-09 ^ 
after: xxasdfasdfxx 

before: appleApple1asdsq23 1.0000e-0 ^ 
after: appleApple1asdsq23 

before: 1.000000 xxxasdfx ^ 
after: 1 

before: 1.0 x23 ^ 
after: 1 

before: 9736 x7 - -100.051731 -0.000000 x15 ^ / + x20 1075.000000 x5 * + -17064.107062 * + 
after: nan 

before: 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - 
after: 0.010000 x1 + sin x0 sin * ~ x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ + - 

before: 0.000 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - ^ 
after: nan 

before: sin tanh 0.000000e0 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - ^ 
after: nan 

before: 0.000 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * sin ~ 
after: 0 

before: sin tanh 0.000000e0 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * ~ 
after: 0 

before: inf 0.000 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * sin ~ + ~ 
after: -inf 

before: sin tanh 0.000000e0 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * ~ inf exp + ~ 
after: -inf 

before: x y + cos sin ~ sech 
after: x y + cos sin sech 

before: z apple sech + tanh ~ sech 
after: z apple sech + tanh sech 

before: x y + cos sin ~ sech 0.000 sin / 
after: nan 

before: z apple sech + tanh ~ sech 0 0.00e0 + tanh sin sin / 
after: nan 

before: inf 0.000 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * sin ~ + cos 
after: nan 

before: sin tanh 0.000000e0 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * ~ inf exp + exp ln log cos 
after: nan 

before: inf 0.000 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * sin ~ + exp ~ cos 
after: nan 

before: sin tanh 0.000000e0 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * ~ inf exp + exp ln log ~ ~ ~ cos 
after: nan 

before: inf 0.000 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * sin ~ + exp ~ ~ sin 
after: nan 

before: sin tanh 0.000000e0 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * ~ inf exp + exp ln log ~ ~ sin 
after: nan 

before: inf 0.000 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * sin ~ + exp ~ ~ ~ sin 
after: nan 

before: sin tanh 0.000000e0 0.000000e0 0.010000 x1 + sin 100000e-005 x0 sin * * - x1 -10 / 0.02 * 0.010000 6.283190 x0 + ^ 0.0000 + + - * ~ inf exp + exp ln log ~ sin 
after: nan 

before: x y + cos sin ~ sech 0 1 - * 
after: x y + cos sin sech ~ 

before: x y + cos cos sech sin ~ sech 0 1.0000e0 - * 
after: x y + cos cos sech sin sech ~ 

before: -1.000e0 x y + cos x y + sech + tanh * 
after: x y + cos x y + sech + tanh ~ 

before: -1000.0e-3 x x x + + sech * 
after: x x x + + sech ~ 

before: x y + cos sin ~ sech 00.000e0 1 - / 
after: x y + cos sin sech ~ 

before: x y + cos cos sech sin ~ sech 0000 1.0000e0 - / 
after: x y + cos cos sech sin sech ~ 

before: 0.000 sinx y + cos sin ~ sech 00.000e0 1 - / * log 
after: -inf 

before: 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / ln 
after: -inf 

before: 0.000e0 x y + cos sin ~ sech 00.000e0 1 - * ^ log 
after: nan 

before: nan 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / ^ ln 
after: nan 

before: inf exp x y + cos sin ~ sech 00.000e0 1 - * + log 
after: inf x y + cos sin sech ~ + log 

before: inf 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln 
after: inf 

before: 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin 
after: 0 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 
after: 0 

before: nan 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh + asin 
after: nan 

before: 0.000e0 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln ^ arcsin 
after: nan 

before: inf 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + asin 
after: nan 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 1.00 inf ln log * + arcsin 
after: nan 

before: 1.000e0 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos 
after: 0 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 1.00 1 * + arccos 
after: 0 

before: nan 1.000e0 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos * acos 
after: nan 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 1.00 nan * + arccos 
after: nan 

before: inf 1.000e0 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos + acos 
after: nan 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 1.00 inf ln * + arccos 
after: nan 

before: 100e-2 1.000e0 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt 
after: 1 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 
after: 1 

before: 100e-2 1.000e0 0.000 x ^ x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt 
after: nan 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0.00e0 * 0000 1.0000e0 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 
after: nan 

before: inf 100e-2 1.000e0 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt * sqrt 
after: inf 

before: 0 ln ~ 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt * sqrt 
after: inf 

before: 1 100e-2 1.000e0 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt 
after: 0 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt 
after: 0 

before: 1 100e-2 1.000e0 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln cos 
after: nan 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt 0 + log cos 
after: nan 

before: 1 100e-2 1.000e0 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln 1000.0e-4 * cos 
after: nan 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt inf - cos 
after: nan 

before: 1 1000e-3 1.000e0 0.000 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln sin 
after: nan 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt 0.00000e0 + log sin 
after: nan 

before: 1 100e-2 1.000e0 0.000 exp 0 * x y + cos sech sin ~ ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln 1000.0e-4 * sin 
after: nan 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 10000e-4 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt inf - sin 
after: nan 

before: 1 1000e-3 1.000e0 0.000e0 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln ~ 
after: inf 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt 0.000000000e0 + log ~ 
after: inf 

before: 1 100e-2 1.000e0 0.000 exp 0 * x y + cos sech sin ~ ~ ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln 1000.0e-4 * ~ 
after: inf 

before: 100e-2 0.000e0 x y + cos cos tanh sin ~ sech 0000 10000e-4 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt inf - ~ 
after: inf 

before: 1 1000e-3 1.000e0 0.000e0 exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln ~ ~ ln 
after: nan 

before: 100e-2 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 100e-2 0.0000e0 exp * + sqrt 0.000 exp - sqrt 0.000000000e0 + log ~ ~ ~ ~ log 
after: nan 

before: 1 100e-2 1.000e0 0.000 exp 0 * x y + cos sech sin ~ ~ ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln 1000.0e-4 * ~ ~ log 
after: nan 

before: 100e-2 0.000e0 x y + 1 * cos cos tanh sin ~ sech 0000 100000.e-4 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt inf - ~ ~ ln 
after: nan 

before: 1 1000e-3 1.000e0 0.000e0 exp ln exp 0 * x y + cos sin ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln ~ ~ asin 
after: nan 

before: 1000e-3 0 + 0.000e0 x y + cos cos sech sin ~ sech 0000 1.0000e0 - / / + ln arcsin 100e-2 0.0000e0 exp * + sqrt 0.000 exp - sqrt 0.000000000e0 + log ~ ~ ~ ~ arcsin 
after: nan 

before: 1 0.000e0 + 100e-2 1.000e0 0.000 exp 0 * x y + cos sech sin ~ ~ ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln 1000.0e-4 * ~ ~ asin 
after: nan 

before: 100000.e-5 0. + 0.000e0 x y + 1 * cos cos tanh sin ~ sech 0000 100000.e-4 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt inf - ~ ~ arcsin 
after: nan 

before: 1 1000e-3 1.000e0 0.000e0 exp ln exp 0 * x y + cos ln exp sin ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln ~ ~ acos 
after: nan 

before: 1000e-3 0 + 0.000e0 x y + cos cos sech sin ~ ~ ~ sech 0000 1.0000e0 - / / + ln arcsin 100e-2 0.0000e0 exp * + sqrt 0.000 exp - sqrt 0.000000000e0 + log ~ ~ ~ ~ arccos 
after: nan 

before: 1 0.000e0 asin + 100e-2 1.000e0 0.000 exp 0 * x y + cos sech sin ~ ~ ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln 1000.0e-4 * ~ ~ ~ ~ acos 
after: nan 

before: 1000000.e-6 0. + 0.000e0 x 1.000e0 * y + 1 * cos cos tanh sin ~ sech 0000 100000.e-4 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt inf - ~ ~ arccos 
after: nan 

before: 1.000000 1000e-3 1.000e0 0 + 0.000e0 exp ln exp 0 * x y + cos ln exp sin ~ sech 00.000e0 1000e-3 - * * tanh asin + arccos + sqrt - sqrt ln ~ ~ sqrt 
after: nan 

before: 1000e-3 0 + 0.000e0 x yessss 1 * + cos cos sech sin ~ ~ ~ sech 0000 1.0000e0 - / / + ln arcsin 100e-2 0.0000e0 exp * + sqrt 0.000 exp - sqrt 0.000000000e0 + log ~ ~ ~ ~ sqrt 
after: nan 

before: 1 0.000e0 ~ ~ asin + 100e-2 1.000e0 0.000 exp 0 * x y + cos sech sin ~ ~ ~ sech 00.000e0 1 - * * tanh asin + arccos + sqrt - sqrt ln 1000.0e-4 * ~ ~ ~ ~ sqrt 
after: nan 

before: 1000000.e-6 0. + 0.000e0 x ln 1.000e0 * y + 1 * cos cos tanh sin ~ sech 0000 100000.e-4 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt inf - ~ ~ sqrt 
after: nan 

before: 1000000.e-6 0. + 0.000e0 x ln 1.000e0 * y + 1 * cos cos tanh sin ~ sech 0000 100000.e-4 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt -1 - ~ 
after: -1 

before: 1000000.e-6 0. + 0.000e0 x ln 1.000e0 * y + 1 * cos cos tanh sin ~ sech 0000 100000.e-4 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt -100e-2 - ~ ~ ~ 
after: -1 

before: 1000000.e-6 0. + 0.0000000e0 xasdfasdf ln 1.000e0 * y + 1 * cos cos tanh sin ~ sech 0000 100000.e-4 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt -1 - ~ ~ 
after: 1 

before: 1000000.e-6 0. + 0.000e0 x x + xxx + ln 1.000e0 * y + 1 * cos cos tanh sin ~ sech 0000 100000.e-4 - / / + ln arcsin 1.00 0.0000e0 exp * + sqrt 0.000 exp - sqrt -100e-2 - ~ ~ ~ ~ 
after: 1 

before: 0.7911530997475994 x0 x0 cos - sech sqrt 1.048576e+06 0 x0 10.000000 - ^ tanh 1.0001 / arccos log ~ 4 / cos cos ^ * ^ 
after: nan 

before: 1e500 3.444 1.22 cos + * 
after: inf 

before: -2e390 3.444 1.22 cos + 3.444 1.22 sin + + * 
after: -inf 
'''
print(before==after)
