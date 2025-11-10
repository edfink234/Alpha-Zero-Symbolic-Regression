before='''before: cos + - ^ x 0.000 1.00000 ~ inf 
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
'''
after='''before: cos + - ^ x 0.000 1.00000 ~ inf 
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
'''
print(before==after)
