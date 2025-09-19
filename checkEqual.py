before='''before: 1 x x * + 0 / ~ 
after: 1 x x * + 0 / ~ 

before: 1 x sin x * + 0 / ~ 
after: 1 x sin x * + 0 / ~ 

before: 1 x x * + 1 x x * + - exp 
after: 2.718281828459045 

before: 1 x sin x * + 1 x sin x * + - exp 
after: 2.718281828459045 

before: 1 x x * + 0 / exp 
after: 1 x x * + 0 / exp 

before: 1 x sin x * + 0 / exp 
after: 1 x sin x * + 0 / exp 

before: 1 x x * + 0 / ~ exp 
after: 1 x x * + 0 / ~ exp 

before: 1 x sin x * + 0 / ~ exp 
after: 1 x sin x * + 0 / ~ exp 
'''
after='''before: 1 x x * + 0 / ~ 
after: 1 x x * + 0 / ~ 

before: 1 x sin x * + 0 / ~ 
after: 1 x sin x * + 0 / ~ 

before: 1 x x * + 1 x x * + - exp 
after: 2.718281828459045 

before: 1 x sin x * + 1 x sin x * + - exp 
after: 2.718281828459045 

before: 1 x x * + 0 / exp 
after: 1 x x * + 0 / exp 

before: 1 x sin x * + 0 / exp 
after: 1 x sin x * + 0 / exp 

before: 1 x x * + 0 / ~ exp 
after: 1 x x * + 0 / ~ exp 

before: 1 x sin x * + 0 / ~ exp 
after: 1 x sin x * + 0 / ~ exp 
'''
print(before==after)
