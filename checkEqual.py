before='''before: inf ~ tanh 
after: -1 

before: 0 inf - tanh 
after: -1 

before: x x ^ x x ^ - asin tanh sin x x - * 0 / 1 ^ sech 
after: 0 

before: x x + x x - - asin tanh sin x x - * 0 / tanh 0 / sech 
after: 0 

before: x x ^ x x ^ - asin tanh sin x x - * 0 / 1 ^ ~ sech 
after: 0 

before: x x + x x - - asin tanh sin x x - * 0 / tanh 0 / ~ sech 
after: 0 
'''
after='''before: inf ~ tanh 
after: -1 

before: 0 inf - tanh 
after: -1 

before: x x ^ x x ^ - asin tanh sin x x - * 0 / 1 ^ sech 
after: 0 

before: x x + x x - - asin tanh sin x x - * 0 / tanh 0 / sech 
after: 0 

before: x x ^ x x ^ - asin tanh sin x x - * 0 / 1 ^ ~ sech 
after: x x ^ x x ^ - asin tanh sin 0 * 0 / 1 ^ ~ sech 

before: x x + x x - - asin tanh sin x x - * 0 / tanh 0 / ~ sech 
after: x x + asin tanh sin 0 * 0 / tanh 0 / ~ sech 
'''

