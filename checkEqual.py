after='''before: 0 w / y 1 / / 
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
after: -inf 

before: x x ^ x x ^ - asin tanh sin x x - * 0 / 
after: inf 
'''
before='''before: 0 w / y 1 / / 
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
after: -inf 

before: x x ^ x x ^ - asin tanh sin x x - * 0 / 
after: inf 
'''
print(after==before)
