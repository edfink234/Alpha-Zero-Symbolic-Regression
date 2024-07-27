#include <iostream>
#include "expression.h"
#include "parser.h"

int main()
{
  Ev3::ExpressionParser parser;
  parser.SetVariableID("x1", 0);
  parser.SetVariableID("x2", 1);
  int nerr = 0;
  Ev3::Expression expr = parser.Parse("x1*sin(x2)", nerr);
  Ev3::Expression derivative = Ev3::Diff(expr, 1);
  std::cout << derivative->ToString() << std::endl;
}

//g++ -o test test.cxx expression.cxx operand.cxx parser.cxx -O2
