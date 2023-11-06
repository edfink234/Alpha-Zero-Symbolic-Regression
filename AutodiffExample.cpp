#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>
 
using namespace std;
 

Eigen::AutoDiffScalar<Eigen::VectorXd> scalarFunctionTwo(std::vector<Eigen::AutoDiffScalar<Eigen::VectorXd>> x) {
  return 2*x[0]*x[0] + 3*x[0] + 3*x[0]*x[1]*x[1] + 2*x[1] + 1;
};
 
void checkFunctionTwo(double & x, double & y, double & dfdx, double & dfdy ) {
  dfdx = 4*x + 3 + 3*y*y;
  dfdy = 6*x*y + 2;
}
 
int main () {
   
    double x, y, z, f, g, dfdx, dgdy, dgdz;
    Eigen::AutoDiffScalar<Eigen::VectorXd> gA;

    std::vector<Eigen::AutoDiffScalar<Eigen::VectorXd>> params(2);
    cout << "Testing scalar function with 2 inputs..." << endl;

    params[0].value() = 1;
    params[1].value() = 2;

    params[0].derivatives() = Eigen::VectorXd::Unit(2, 0);
    params[1].derivatives() = Eigen::VectorXd::Unit(2, 1);

    gA = scalarFunctionTwo(params);
    
    Eigen::VectorXf grad(2);
//
    grad(0) = gA.derivatives()[0];
    grad(1) = gA.derivatives()[1];

    cout << "  AutoDiff:" << endl;
    cout << "    Function output: " << gA.value() << endl;
    cout << "    Derivative: " << grad << '\n';

    y = 1;
    z = 2;
    checkFunctionTwo(y, z, dgdy, dgdz);

    cout << "  Hand differentiation:" << endl;
    cout << "    Derivative: " << dgdy << ", "
    << dgdz << endl;

    return EXIT_SUCCESS;
   
}
