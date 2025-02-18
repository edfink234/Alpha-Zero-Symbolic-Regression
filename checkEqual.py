after='''before: x sin x sin - x sin + 
after: x sin 

before: x 1 / 
after: x 

before: x x * 1 / 
after: x x * 1 / 

before: x x cos * 1 / 
after: x x cos * 1 / 

before: 0 x x * / 
after: 0 x x * / 

before: 0 x x cos * / 
after: 0 x x cos * / 

before: 0 x sin x sech * / 
after: 0 x sin x sech * / 

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
after: x 

before: 0 x ^ x - 
after: x ~ 

before: x cos 0 x ^ - 
after: x cos 

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
after: 1 x exp * ln 

before: x 1 x exp * ln - 
after: x 1 x exp * ln - 

before: x 1 x exp * ln - cos 
after: x 1 x exp * ln - cos 

before: y y * exp ln 
after: y y * 

before: x x * exp y ln * 
after: x x * exp y ln * 

before: y y - exp exp x sin + 
after: 2.718282 x sin + 

before: x x * sin y / 
after: x x * sin y / 

before: x cos sin y / 
after: x cos sin y / 

before: x x - sqrt cos 
after: 1 

before: x x * x x * - sqrt tanh sin 
after: x x * x x * - sqrt tanh sin 

before: x x cos * x x cos * - sqrt sqrt 
after: x x cos * x x cos * - sqrt sqrt 

before: x x - arcsin cos 
after: 1 

before: x x ^ x x ^ - asin tanh sin 
after: x x ^ x x ^ - asin tanh sin 

before: x x sin * x x sin * - arcsin asin 
after: x x sin * x x sin * - arcsin asin 

before: x tanh x tanh - acos exp 
after: x tanh x tanh - acos exp 

before: x x / x x / - arccos sech sech 
after: 0.925521 

before: x x sech - x x sech - - arccos acos 
after: x x sech - x x sech - - arccos acos 

before: x x exp * x x exp * - tanh acos 
after: x x exp * x x exp * - tanh acos 

before: x x exp * x x exp * - sech asin 
after: x x exp * x x exp * - sech asin 

before: x x sech - x x sech - - sech acos 
after: x x sech - x x sech - - sech acos 

before: x0 x0 cos / tanh acos cos 
after: x0 x0 cos / tanh 

before: 0 x tanh tanh * ~ x + 
after: 0 x tanh tanh * ~ x + 

before: y 1 * x2 1 * * 
after: y x2 * 

before: 0 y + 0 x2 + * 
after: y x2 * 

before: x 0 + 0 y + * 
after: x y * 

before: x3 0 + 0 y + / 
after: x3 y / 

before: 0 x3 / 1 y / / 
after: 0 1 y / / 

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
'''
before='''before: x sin x sin - x sin + 
after: x sin 

before: x 1 / 
after: x 

before: x x * 1 / 
after: x x * 1 / 

before: x x cos * 1 / 
after: x x cos * 1 / 

before: 0 x x * / 
after: 0 x x * / 

before: 0 x x cos * / 
after: 0 x x cos * / 

before: 0 x sin x sech * / 
after: 0 x sin x sech * / 

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
after: x 

before: 0 x ^ x - 
after: x ~ 

before: x cos 0 x ^ - 
after: x cos 

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
after: 1 x exp * ln 

before: x 1 x exp * ln - 
after: x 1 x exp * ln - 

before: x 1 x exp * ln - cos 
after: x 1 x exp * ln - cos 

before: y y * exp ln 
after: y y * 

before: x x * exp y ln * 
after: x x * exp y ln * 

before: y y - exp exp x sin + 
after: 2.718282 x sin + 

before: x x * sin y / 
after: x x * sin y / 

before: x cos sin y / 
after: x cos sin y / 

before: x x - sqrt cos 
after: 1 

before: x x * x x * - sqrt tanh sin 
after: x x * x x * - sqrt tanh sin 

before: x x cos * x x cos * - sqrt sqrt 
after: x x cos * x x cos * - sqrt sqrt 

before: x x - arcsin cos 
after: 1 

before: x x ^ x x ^ - asin tanh sin 
after: x x ^ x x ^ - asin tanh sin 

before: x x sin * x x sin * - arcsin asin 
after: x x sin * x x sin * - arcsin asin 

before: x tanh x tanh - acos exp 
after: x tanh x tanh - acos exp 

before: x x / x x / - arccos sech sech 
after: 0.925521 

before: x x sech - x x sech - - arccos acos 
after: x x sech - x x sech - - arccos acos 

before: x x exp * x x exp * - tanh acos 
after: x x exp * x x exp * - tanh acos 

before: x x exp * x x exp * - sech asin 
after: x x exp * x x exp * - sech asin 

before: x x sech - x x sech - - sech acos 
after: x x sech - x x sech - - sech acos 

before: x0 x0 cos / tanh acos cos 
after: x0 x0 cos / tanh 

before: 0 x tanh tanh * ~ x + 
after: 0 x tanh tanh * ~ x + 

before: y 1 * x2 1 * * 
after: y x2 * 

before: 0 y + 0 x2 + * 
after: y x2 * 

before: x 0 + 0 y + * 
after: x y * 

before: x3 0 + 0 y + / 
after: x3 y / 

before: 0 x3 / 1 y / / 
after: 0 1 y / / 

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
'''
print(before==after)
