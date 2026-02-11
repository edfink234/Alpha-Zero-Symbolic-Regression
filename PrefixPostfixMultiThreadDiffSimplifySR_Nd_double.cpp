#include <vector>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <algorithm>
#include <future>         // std::async, std::future
#include <limits>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <ctime>
#include <cstdlib>
#include <stack>
#include <numeric>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <thread>
#include <mutex>
#include <atomic>
#include <charconv>
//#include <latch>
#include <tuple>
#include <functional>
//#include <numbers>
#include <LBFGS.h>
#include <LBFGSB.h>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/AutoDiff>
#include <boost/unordered/concurrent_flat_map.hpp>
#include <boost/spirit/include/qi.hpp> //For fast string-to-double conversion!

#ifndef DBL_MAX
    #define DBL_MAX std::numeric_limits<double>::max()
#endif
#define RANDOM_SEED -1

//TODO: Need to make this robust againt -Wnarrow

/*
Search Directories to add:
 - C:\Users\finkelsteine\test_codes\eigen\
 - C:\Users\finkelsteine\test_codes\eigen\unsupported
 - C:\Users\finkelsteine\test_codes\boost_1_88_0
 - C:\Users\finkelsteine\test_codes\LBFGSpp\include

Set Compilers installation directory in Toolchain Executables to:
 - C:\msys64\ucrt64
*/

using Clock = std::chrono::high_resolution_clock;

//Returns the number of seconds since `start_time`
template <typename T>
double timeElapsedSince(T start_time)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_time).count()/1e9;
}
//double Stod(const std::string& param)
//{
//    try
//    {
//        double val = std::stod(param);
//        return val;
//    }
//    catch (const std::out_of_range&)
//    {
//        if (!param.empty() && param[0] == '-')
//        {
//            return -std::numeric_limits<double>::infinity();
//        }
//        else
//        {
//            return std::numeric_limits<double>::infinity();
//        }
//    }
//}

//Checks if vector `x` has size > 0 and all elements in `x` have size > 0
template <typename T>
bool all_check(const T& x)
{
    if (x.size() == 0)
    {
        return false;
    }
    for (const auto& w : x)
    {
        if (w.size() == 0)
        {
            return false;
        }
    }
    return true;
}

//Checks if vector of vectors `y` has size > 0 and all vectors in `y` have size > 0
template <typename T>
bool all_checks(const T& y)
{
    if (y.size() == 0)
    {
        return false;
    }
    for (const auto& x : y)
    {
        if (!all_check(x))
        {
            return false;
        }
    }
    return true;
}

std::string to_string_general(double v)
{
    if (std::isnan(v))
    {
        return "nan";
    }
    if (std::isinf(v))
    {
        return (v < 0 ? "-inf" : "inf");
    }
    thread_local std::string out(32, '\0'); // start with a small buffer
    while (true)
    {
        auto [p, ec] = std::to_chars(out.data(), out.data() + out.size(), v, std::chars_format::general);
        if (ec == std::errc{})
        {
            out.resize(p - out.data());
            return out;
        }
        out.resize(out.size() * 2);            // grow and retry (rare)
    }
}
// https://www.geeksforgeeks.org/cpp/how-to-split-string-by-delimiter-in-cpp/
std::vector<std::string> split(const std::string& str)
{
    // Create a stringstream object
    // to str
    std::stringstream ss(str);
    std::vector<std::string> vec;

      // Temporary object to store
      // the splitted string
    std::string t;

      // Delimiter
    char del = ' ';

       // Splitting the str string
       // by delimiter
    while (std::getline(ss, t, del))
    {
        vec.push_back(t);
    }

    return vec;
}

Eigen::MatrixXd load_csv(const std::string& path, int rows, int cols, bool header = true)
{
    std::ifstream file(path);
    Eigen::MatrixXd data(rows, cols);
    std::string line;
    int i = 0;
    if (header)
    {
        std::getline(file, line); //header row
    }
    while (std::getline(file, line) && (i < rows))
    {
        std::stringstream ss(line);
        std::string cell;
        int j = 0;
        while (std::getline(ss, cell, ','))
        {
            try
            {
                data(i, j++) = std::stod(cell);
            }
            catch (const std::invalid_argument& e)
            {
                std::cout << "Error caught at element (" << i << ',' << j << ")\n";
            }
            
        }
        i++;
    }
    return data;
}

namespace std
{
    class latch
    {
        std::atomic<std::ptrdiff_t> counter_;
        mutable std::mutex mut_;
        mutable std::condition_variable cv_;

    public:
        explicit latch(std::ptrdiff_t count) : counter_(count)
        {
            if (count < 0) throw std::invalid_argument("latch count must be non-negative");
        }

        latch(const latch&) = delete;
        latch& operator=(const latch&) = delete;

        void count_down(std::ptrdiff_t n = 1)
        {
            if (n <= 0) return;
            auto old = counter_.fetch_sub(n, std::memory_order_acq_rel);
            if (old < n) throw std::runtime_error("latch count went negative");

            if (old == n)
            {
                std::lock_guard<std::mutex> lock(mut_);
                cv_.notify_all();
            }
        }

        void wait() const
        {
            if (try_wait()) return;
            std::unique_lock<std::mutex> lock(mut_);
            cv_.wait(lock, [this]
            {
                return (counter_.load(std::memory_order_acquire) == 0);
            });
        }

        bool try_wait() const noexcept
        {
            return (counter_.load(std::memory_order_acquire) == 0);
        }

        void arrive_and_wait(std::ptrdiff_t n = 1)
        {
            count_down(n);
            wait();
        }

        static constexpr std::ptrdiff_t max() noexcept
        {
            return std::numeric_limits<std::ptrdiff_t>::max();
        }
    };
}
//https://medium.com/@ryan_forrester_/c-check-if-string-is-number-practical-guide-c7ba6db2febf
bool isdouble(const std::string& s)
{
    enum State { START, INT, FRAC, EXP, EXP_NUM };
    State state = START;
    bool has_digits = false;

    if ((s.rfind("nan", 0) != std::string::npos)
        || (s.rfind("inf", 0) != std::string::npos)
        || (s.rfind("-inf", 0) != std::string::npos))
    {
        return true;
    }

    for (char c : s)
    {
        switch (state)
        {
            case START:
                if (c == '+' || c == '-') state = INT;
                else if (std::isdigit(c)) { state = INT; has_digits = true; }
                else if (c == '.') state = FRAC;
                else return false;
                break;
            case INT:
                if (std::isdigit(c)) has_digits = true;
                else if (c == '.') state = FRAC;
                else if (c == 'e' || c == 'E') state = EXP;
                else return false;
                break;
            case FRAC:
                if (std::isdigit(c)) has_digits = true;
                else if (c == 'e' || c == 'E') state = EXP;
                else return false;
                break;
            case EXP:
                if (c == '+' || c == '-' || std::isdigit(c)) state = EXP_NUM;
                else return false;
                break;
            case EXP_NUM:
                if (!std::isdigit(c)) return false;
                break;
        }
    }
    return has_digits && (state == INT || state == FRAC || state == EXP_NUM);
}

bool parse_double_spirit(const std::string& s, double& out)
{
    namespace qi   = boost::spirit::qi;
    namespace ascii= boost::spirit::ascii;

    auto f = s.begin(), l = s.end();
    // Skip leading/trailing ASCII whitespace; require full consumption (eoi).
    // qi::double_ already yields ±inf/NaN where appropriate.
    bool ok = qi::phrase_parse(f, l, qi::double_ >> qi::eoi, ascii::space, out);
    return ok; // f==l guaranteed by eoi
}

double Stod(const std::string& param)
{
    double val;
    parse_double_spirit(param, val);
    return val;
}

bool checkEqual(const std::string &str1, const std::string &str2)
{
    if (!isdouble(str1))
    {
        return false;
    }
    double val1; parse_double_spirit(str1, val1);
    double val2 = 0.0;
    if (str2 == "0")
    {
        val2 = 0.0;
    }
    else if (str2 == "1")
    {
        val2 = 1.0;
    }
    else if (str2 == "-1")
    {
        val2 = -1.0;
    }
    return (val1 == val2);
}

// Function to create a matrix with linspace columns. std::vector<double> min and
// std::vector<double> max must have size == cols
Eigen::MatrixXd createLinspaceMatrix(int rows, int cols, std::vector<double> min_vec, std::vector<double> max_vec)
{
    assert( (cols == static_cast<int>(min_vec.size())) && (cols == static_cast<int>(max_vec.size())) );
    Eigen::MatrixXd mat(rows, cols);
    for (int col = 0; col < cols; ++col)
    {
        for (int row = 0; row < rows; ++row)
        {
            mat(row, col) = min_vec[col] + (max_vec[col] - min_vec[col]) * row / (rows - 1);
        }
    }
    return mat;
}

// Helper function to create a linspace vector
std::vector<double> linspace(double min_val, double max_val, int num_points)
{
    std::vector<double> linspaced(num_points);
    double step = (max_val - min_val) / (num_points - 1);
    for (int i = 0; i < num_points; ++i)
    {
        linspaced[i] = min_val + i * step;
    }
    return linspaced;
}

//`std::vector<double> min_vec` and `std::vector<double> max_vec` must have `size == col`
Eigen::MatrixXd createMeshgridWithLambda(int rows, int cols, std::vector<double> min_vec, std::vector<double> max_vec, const std::function<double(const Eigen::RowVectorXd&)>& lambda)
{
    assert((cols == static_cast<int>(min_vec.size())) && (cols == static_cast<int>(max_vec.size())));

    // Create linspaces for each variable (column)
    std::vector<std::vector<double>> linspaces;
    for (int col = 0; col < cols; ++col)
    {
        linspaces.push_back(linspace(min_vec[col], max_vec[col], rows)); // Assuming linspace function is defined elsewhere
    }

    // Calculate the total number of combinations (flattened meshgrid size)
    int total_combinations = 1;
    for (int col = 0; col < cols; ++col)
    {
        total_combinations *= rows;
    }

    // Create a matrix to store all combinations with the additional column for the lambda function result
    Eigen::MatrixXd matrix(total_combinations, cols + 1);

    // Fill in the matrix with all combinations of linspace values and apply the lambda function
    for (int col = 0; col < cols; ++col)
    {
        int repeat_count = 1;
        for (int i = col + 1; i < cols; ++i)
        {
            repeat_count *= rows;
        }

        int num_repeats = total_combinations / (repeat_count * rows);
        for (int repeat = 0; repeat < num_repeats; ++repeat)
        {
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < repeat_count; ++j)
                {
                    int index = repeat * repeat_count * rows + i * repeat_count + j;
                    matrix(index, col) = linspaces[col][i];
                }
            }
        }
    }

    // Apply the lambda function to each row and store the result in the last column
    for (int i = 0; i < total_combinations; ++i)
    {
        matrix(i, cols) = lambda(matrix.row(i).head(cols));
    }

    return matrix;
}

Eigen::MatrixXd addColumnWithLambda(const Eigen::MatrixXd& matrix, const std::function<double(const Eigen::RowVectorXd&)>& lambda) {
    // Get the number of rows and columns of the input matrix
    int rows = matrix.rows();
    int cols = matrix.cols();

    // Create a new matrix with an additional column
    Eigen::MatrixXd newMatrix(rows, cols + 1);

    // Copy the original matrix into the new matrix (without the last column)
    newMatrix.block(0, 0, rows, cols) = matrix;

    // Apply the lambda function to each row and store the result in the last column
    for (int i = 0; i < rows; ++i)
    {
        newMatrix(i, cols) = lambda(matrix.row(i));
    }

    return newMatrix;
}

/*
    In the function below, `min_vec` and `max_vec` are the min and max vals for each attribute,
    `num_cols` is the number of attributes, and `rows` is the number of linearly-spaced data-points
*/
Eigen::MatrixXd createMeshgridVectors(int rows, int cols, std::vector<double> min_vec, std::vector<double> max_vec)
{
    assert((min_vec.size() < INT_MAX) && (max_vec.size() < INT_MAX));
    assert( (cols == static_cast<int>(min_vec.size())) && (cols == static_cast<int>(max_vec.size())) );

    // Create linspaces for each variable (column)
    std::vector<std::vector<double>> linspaces;
    for (int col = 0; col < cols; ++col)
    {
        linspaces.push_back(linspace(min_vec[col], max_vec[col], rows));
    }

    // Calculate the total number of combinations (flattened meshgrid size)
    int total_combinations = 1;
    for (int col = 0; col < cols; ++col)
    {
        total_combinations *= rows;
    }

    // Create a matrix to store all combinations (rows = total_combinations, cols = number of variables)
    Eigen::MatrixXd matrix(total_combinations, cols);

    // Fill in the matrix with all combinations of linspace values
    for (int col = 0; col < cols; ++col)
    {
        int repeat_count = 1;
        for (int i = col + 1; i < cols; ++i)
        {
            repeat_count *= rows;
        }

        int num_repeats = total_combinations / (repeat_count * rows);
        for (int repeat = 0; repeat < num_repeats; ++repeat)
        {
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < repeat_count; ++j)
                {
                    int index = repeat * repeat_count * rows + i * repeat_count + j;
                    matrix(index, col) = linspaces[col][i];
                }
            }
        }
    }

    return matrix;
}

Eigen::MatrixXd hstack(const Eigen::MatrixXd& mat1, const Eigen::MatrixXd& mat2)
{
    Eigen::MatrixXd result(mat1.rows(), mat1.cols() + mat2.cols());
    result << mat1, mat2; // Concatenate horizontally
    return result;
}

int trueMod(int N, int M)
{
    return ((N % M) + M) % M;
};

bool isInvalid(double x)
{
    return (std::isnan(x) || std::isinf(x));
}

double Variance(const Eigen::VectorXd& vec)
{
    return (vec.array() - vec.mean()).square().sum() / vec.size();
}

Eigen::VectorXd VarianceVec(const Eigen::VectorXd& vec)
{
    return (vec.array() - vec.mean()).square() / vec.size();
}

std::vector<Eigen::VectorXd> Variance(const std::vector<Eigen::VectorXd>& vec)
{
    size_t sz = vec.size();
    std::vector<Eigen::VectorXd> temp(sz);

    for (size_t idx = 0; idx < sz; idx++)
    {
        temp[idx] = VarianceVec(vec[idx]);
    }
    return temp;
}

double VarianceSum(const std::vector<Eigen::VectorXd>& vec)
{
    size_t sz = vec.size();
    double temp = 0.0;

    for (size_t idx = 0; idx < sz; idx++)
    {
        temp += Variance(vec[idx]);
    }
    return temp;
}

Eigen::MatrixXd deg2rad(const Eigen::VectorXd& vec)
{
    // Define the conversion factor from degrees to radians
    // Conversion: radians = degrees * (M_PI / 180.0)
    const double deg_to_rad = M_PI / 180.0;
    
    // Perform element-wise multiplication on the vector
    // The result is an Eigen::VectorXd (which is also a type of Eigen::MatrixXd
    // where the number of columns is 1).
    return vec * deg_to_rad;
}

/*
||=== Build file: "no target" in "no project" (compiler: unknown) ===|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In function 'bool isZero(const Eigen::VectorXd&, double)':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|415|warning: comparison of integer expressions of different signedness: 'size_t' {aka 'long long unsigned int'} and 'Eigen::EigenBase<Eigen::Matrix<double, -1, 1> >::Index' {aka 'long long int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In function 'bool isZero(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::Matrix<double, -1, 1> >, -1>&, double)':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|431|warning: comparison of integer expressions of different signedness: 'size_t' {aka 'long long unsigned int'} and 'Eigen::EigenBase<Eigen::Matrix<Eigen::AutoDiffScalar<Eigen::Matrix<double, -1, 1> >, -1, 1, 0, -1, 1> >::Index' {aka 'long long int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In function 'bool isConstant(const Eigen::VectorXd&, double)':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|451|warning: comparison of integer expressions of different signedness: 'size_t' {aka 'long long unsigned int'} and 'Eigen::EigenBase<Eigen::Matrix<double, -1, 1> >::Index' {aka 'long long int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In function 'bool isConstant(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::Matrix<double, -1, 1> >, -1>&, double)':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|467|warning: comparison of integer expressions of different signedness: 'size_t' {aka 'long long unsigned int'} and 'Eigen::EigenBase<Eigen::Matrix<Eigen::AutoDiffScalar<Eigen::Matrix<double, -1, 1> >, -1, 1, 0, -1, 1> >::Index' {aka 'long long int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In constructor 'Board::Board(std::vector<std::vector<std::__cxx11::basic_string<char> > > (*)(Board&), size_t, bool, const std::vector<int>&, const std::string&, size_t, std::string, int, std::string, const Eigen::MatrixXd&, bool, bool, bool, double, bool, bool, int)':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|725|warning: comparison of integer expressions of different signedness: 'int' and 'size_t' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In member function 'std::string Board::print_expression_params()':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|861|warning: comparison of integer expressions of different signedness: 'size_t' {aka 'long long unsigned int'} and 'Eigen::EigenBase<Eigen::Matrix<double, -1, 1> >::Index' {aka 'long long int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In member function 'void Board::graspSimplifyPrefixHelper(std::vector<std::__cxx11::basic_string<char> >&, int, int, std::vector<int>&, std::vector<std::__cxx11::basic_string<char> >&, bool)':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|1273|warning: comparison of integer expressions of different signedness: 'int' and 'std::vector<std::__cxx11::basic_string<char> >::size_type' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|1305|warning: comparison of integer expressions of different signedness: 'int' and 'std::vector<std::__cxx11::basic_string<char> >::size_type' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In member function 'std::pair<int, bool> Board::getRPNdepth(const std::vector<std::__cxx11::basic_string<char> >&, int, size_t, size_t, bool, bool, bool, bool)':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|2516|warning: comparison of integer expressions of different signedness: 'std::vector<std::vector<int> >::size_type' {aka 'long long unsigned int'} and 'int' [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|2644|warning: comparison of integer expressions of different signedness: 'std::vector<std::vector<int> >::size_type' {aka 'long long unsigned int'} and 'int' [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In member function 'std::vector<std::__cxx11::basic_string<char> > Board::get_legal_moves(int)':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|2672|warning: comparison of integer expressions of different signedness: 'std::vector<int>::size_type' {aka 'long long unsigned int'} and 'int' [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In member function 'std::string Board::_to_infix(bool)':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|2810|warning: comparison of integer expressions of different signedness: 'int' and 'size_t' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In member function 'std::string Board::expression(bool)':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|2865|warning: comparison of integer expressions of different signedness: 'int' and 'size_t' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In member function 'std::string Board::_to_infix(const std::vector<std::vector<std::__cxx11::basic_string<char> > >&, bool)':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|2927|warning: comparison of integer expressions of different signedness: 'int' and 'size_t' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In member function 'std::string Board::expression(const std::vector<std::vector<std::__cxx11::basic_string<char> > >&, bool)':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|2939|warning: comparison of integer expressions of different signedness: 'int' and 'size_t' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In member function 'double Board::operator()(Eigen::VectorXd&, Eigen::VectorXd&)':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|3487|warning: comparison of integer expressions of different signedness: 'size_t' {aka 'long long unsigned int'} and 'Eigen::EigenBase<Eigen::Matrix<double, -1, 1> >::Index' {aka 'long long int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|3488|warning: comparison of integer expressions of different signedness: 'size_t' {aka 'long long unsigned int'} and 'int' [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|3489|warning: comparison of integer expressions of different signedness: 'Eigen::EigenBase<Eigen::Matrix<double, -1, 1> >::Index' {aka 'long long int'} and 'size_t' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In member function 'double Board::fitFunctionToData()':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|3657|warning: comparison of integer expressions of different signedness: 'int' and 'std::vector<std::vector<std::__cxx11::basic_string<char> > >::size_type' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|3734|warning: comparison of integer expressions of different signedness: 'int' and 'std::vector<std::vector<std::__cxx11::basic_string<char> > >::size_type' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|3820|warning: comparison of integer expressions of different signedness: 'int' and 'std::vector<std::vector<std::__cxx11::basic_string<char> > >::size_type' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In member function 'double Board::complete_status(int, bool)':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|3848|warning: comparison of integer expressions of different signedness: 'std::vector<std::vector<int> >::size_type' {aka 'long long unsigned int'} and 'int' [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|3849|warning: comparison of integer expressions of different signedness: 'std::vector<int>::size_type' {aka 'long long unsigned int'} and 'int' [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|3859|warning: comparison of integer expressions of different signedness: 'std::vector<std::vector<int> >::size_type' {aka 'long long unsigned int'} and 'int' [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|3865|warning: comparison of integer expressions of different signedness: 'int' and 'std::vector<std::vector<std::__cxx11::basic_string<char> > >::size_type' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|3879|warning: comparison of integer expressions of different signedness: 'int' and 'std::vector<std::vector<std::__cxx11::basic_string<char> > >::size_type' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|3889|warning: comparison of integer expressions of different signedness: 'int' and 'std::vector<std::vector<std::__cxx11::basic_string<char> > >::size_type' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|3895|warning: comparison of integer expressions of different signedness: 'int' and 'std::vector<std::vector<std::__cxx11::basic_string<char> > >::size_type' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|3927|warning: comparison of integer expressions of different signedness: 'int' and 'std::vector<std::vector<std::__cxx11::basic_string<char> > >::size_type' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|3941|warning: comparison of integer expressions of different signedness: 'int' and 'size_t' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In function 'std::vector<std::vector<std::__cxx11::basic_string<char> > > SolitonWaveFengEq14and15Laser(Board&)':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|5487|warning: comparison of integer expressions of different signedness: 'int' and 'std::vector<std::vector<std::__cxx11::basic_string<char> > >::size_type' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|6536|warning: comparison of integer expressions of different signedness: 'int' and 'size_t' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|6543|warning: comparison of integer expressions of different signedness: 'int' and 'size_t' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|6683|warning: comparison of integer expressions of different signedness: 'std::vector<std::vector<std::__cxx11::basic_string<char> > >::size_type' {aka 'long long unsigned int'} and 'int' [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|7686|warning: comparison of integer expressions of different signedness: 'boost::unordered::concurrent_flat_map<std::__cxx11::basic_string<char>, Eigen::Matrix<double, -1, 1> >::size_type' {aka 'long long unsigned int'} and 'int' [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|7714|warning: comparison of integer expressions of different signedness: 'int' and 'std::vector<std::vector<std::__cxx11::basic_string<char> > >::size_type' {aka 'long long unsigned int'} [-Wsign-compare]|
||=== Build finished: 0 error(s), 36 warning(s) (0 minute(s), 38 second(s)) ===|

*/

template<typename Derived>
typename Derived::Scalar median( Eigen::DenseBase<Derived>& d )
{
    auto r { d.reshaped() };
    std::sort( r.begin(), r.end() );
    return r.size() % 2 == 0 ?
        r.segment( (r.size()-2)/2, 2 ).mean() :
        r( r.size()/2 );
}

template<typename Derived>
typename Derived::Scalar median( const Eigen::DenseBase<Derived>& d )
{
    typename Derived::PlainObject m { d.replicate(1,1) };
    return median(m);
}

/*
 Checks if vec is "zero-ish"
    - Returns `false` if either max(abs(vec)) > tolerance or if median(vec) > tolerance
    - Returns `true` otherwise
 */
bool isZero(const Eigen::VectorXd& vec, double tolerance = 1e-5)
{
    if (vec.size() <= 1)
    {
        return true; // A vector with 0 or 1 element is trivially constant
    }
    if (vec.array().isNaN().any() || vec.array().isInf().any())
    {
        return true; // Return true if any NaN is present so it'll be weeded out
    }
    for (decltype(vec.size()) i = 0; i < vec.size(); ++i)
    {
        if (isInvalid(vec[i]))
        {
            return true; // Return true if any NaN is present in values
        }
    }
    return ((vec.array().abs().maxCoeff() <= tolerance) && (median(vec) <= tolerance));
    //MARK: If we want to make sure that the max value AND the median are both bigger than `tolerance`, then we should change the `&&` to `||` (and maybe also change `median(vec)` to `median(abs(vec))`)
}

/*
 Checks if vec is "zero-ish"
    - Returns `false` if either max(abs(vec)) > tolerance or if median(vec) > tolerance
    - Returns `true` otherwise
 */
bool isZero(const Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic>& vec, double tolerance = 1e-5)
{
    if (vec.size() <= 1)
    {
        return true; // A vector with 0 or 1 element is trivially constant
    }
    for (decltype(vec.size()) i = 0; i < vec.size(); ++i)
    {
        if (isInvalid(vec[i].value()))
        {
            return true; // Return true if any NaN is present in values
        }
    }
    return ((vec.array().abs().maxCoeff() <= tolerance) && (median(vec) <= tolerance));
    //MARK: If we want to make sure that the max value AND the median are both bigger than `tolerance`, then we should change the `&&` to `||` (and maybe also change `median(vec)` to `median(abs(vec))`)
}

bool isConstant(const Eigen::VectorXd& vec, double tolerance = 1e-5)
{
    if (vec.size() <= 1)
    {
        return true; // A vector with 0 or 1 element is trivially constant
    }
    if (vec.array().isNaN().any() || vec.array().isInf().any())
    {
        return true; // Return true if any NaN is present so it'll be weeded out
    }
    for (decltype(vec.size()) i = 0; i < vec.size(); ++i)
    {
        if (isInvalid(vec[i]))
        {
            return true; // Return true if any NaN is present in values
        }
    }
    return (Variance(vec) <= tolerance);
}

bool isConstant(const Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic>& vec, double tolerance = 1e-5)
{
    if (vec.size() <= 1)
    {
        return true; // A vector with 0 or 1 element is trivially constant
    }
    for (decltype(vec.size()) i = 0; i < vec.size(); ++i)
    {
        if (isInvalid(vec[i].value()))
        {
            return true; // Return true if any NaN is present in values
        }
    }
    auto firstElement = vec(0);
    return (vec.array() - firstElement).abs().maxCoeff() <= tolerance;
}

class Data
{
    Eigen::MatrixXd data;
    std::unordered_map<std::string, Eigen::VectorXd> features;
    std::vector<Eigen::VectorXd> rows;

public:

    long num_columns, num_rows;
    Data() = default; //so we can have a static Data attribute

    // Assignment operator
    Data& operator=(const Eigen::MatrixXd& theData)
    {
        this->data = theData;
        this->num_columns = data.cols();
        this->num_rows = data.rows();
        for (long i = 0; i < this->num_columns; i++) //for each column
        {
            this->features["x"+std::to_string(i)] = Eigen::VectorXd(this->num_rows);
            for (long j = 0; j < this->num_rows; j++)
            {
                this->features["x"+std::to_string(i)](j) = this->data(j,i);
            }
//            printf("this->features[x%ld].sum() = %lf\n", i, this->features["x"+std::to_string(i)].sum());
        }
//        exit(1);
        this->rows.resize(this->num_rows);

        for (long i = 0; i < num_rows; i++)
        {
            this->rows[i] = data.row(i);
        }

        return *this;
    }

    bool operator==( Data& other)
    {
        return this->data == other.data;
    }

    const Eigen::VectorXd& operator[] (int i){return rows[i];}
    Eigen::VectorXd& operator[] (const std::string& i)
    {
        return features[i];
    }
    long numRows() const {return num_rows;}
    long numCols() const {return num_columns;}

    friend std::ostream& operator<<(std::ostream& os, const Data& matrix)
    {
        return (os << matrix.data);
    }
};

template <typename T, typename U>
std::ostream& operator<<(std::ostream& os, const std::pair<T, U>& data)
{
    return (os << '(' << data.first << ", " << data.second << ')');
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<T>>& data)
{
    for (const auto& vec: data)
    {
        for (const auto& elem: vec)
        {
            os << elem << ' ';
        }
        os << '\n';
    }
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& data)
{
    for (const auto& i: data)
    {
        os << i << ' ';
    }
    return os;
}

void print_container(const std::vector<std::string>& c, int low, int up)
{
    for (int i = low; i <= up; i++)
        std::cout << c[i] << ' ';
    puts("");
}

double SNE(const Eigen::VectorXd& actual)
{
    return actual.squaredNorm();
}

double SNE(const std::vector<Eigen::VectorXd>& actual)
{
    double temp = 0.0;
    for (decltype(actual.size()) i = 0; i < actual.size(); ++i)
    {
        temp += actual[i].squaredNorm();
    }

    return temp;
}

double SNE(const std::vector<Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic>>& actual)
{
    double temp = 0.0;
    size_t count = 0;

    for (const auto& vec : actual)
    {
        for (decltype(vec.size()) i = 0; i < vec.size(); ++i)
        {
            // Access the value of the AutoDiffScalar element
            temp += vec[i].value() * vec[i].value();
        }
        ++count;
    }

    return count > 0 ? temp / count : DBL_MAX;
}

double SNE(const Eigen::VectorXd& actual, const Eigen::VectorXd& predicted)
{
//    if (actual.size() != predicted.size())
//    {
//        throw std::invalid_argument("Vectors must be of the same size");
//    }
    assert((actual.size() == predicted.size()) && "Vectors must be of the same size");
    return (actual - predicted).squaredNorm();
}

Eigen::AutoDiffScalar<Eigen::VectorXd> SNE(const Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic>& actual)
{
    return actual.squaredNorm();
}

double loss_func(const Eigen::VectorXd& actual)
{
    return (1.0/(1.0+SNE(actual)));
}

double loss_func(const std::vector<Eigen::VectorXd>& actual)
{
    double sne = 0.0;
    for (decltype(actual.size()) i = 0; i < actual.size(); ++i)
    {
        sne += SNE(actual[i]);
    }
    return (1.0/(1.0+sne));
}

double loss_func(const Eigen::VectorXd& actual, const Eigen::VectorXd& predicted)
{
    return (1.0/(1.0+SNE(actual, predicted)));
}

struct Board
{
    struct ExprDag
    {
        enum class Kind : uint8_t
        {
            Leaf,
            Unary,
            Binary
        };

        struct Node
        {
            Kind kind = Kind::Leaf;

            // For Leaf: token stores the leaf token (e.g. "const3", "x", "0", "1", "2", "4", "3.14")
            // For Unary/Binary: token stores the operator (e.g. "sin", "+", "^", "~")
            std::string token;

            // Child indices in nodes vector. Unused for Leaf.
            int child0 = -1; // unary child, or binary left child
            int child1 = -1; // binary right child
        };

        std::vector<Node> nodes;
        int root = -1;
    };
    
    struct DagNodeKey
    {
        ExprDag::Kind kind;
        std::string token;
        int c0;
        int c1;

        bool operator==(const DagNodeKey& other) const
        {
            return kind == other.kind && token == other.token && c0 == other.c0 && c1 == other.c1;
        }
    };

    struct DagNodeKeyHash
    {
        std::size_t operator()(const DagNodeKey& k) const
        {
            // A simple hash combine
            std::size_t h = 1469598103934665603ULL; // FNV offset basis
            auto mix = [&](std::size_t x) {
                h ^= x;
                h *= 1099511628211ULL;
            };

            mix(static_cast<std::size_t>(k.kind));
            mix(std::hash<std::string>{}(k.token));
            mix(std::hash<int>{}(k.c0));
            mix(std::hash<int>{}(k.c1));
            return h;
        }
    };
    
    static boost::concurrent_flat_map<std::string, Eigen::VectorXd> inline expression_dict;
    static constexpr size_t max_expression_dict_sz = 100000000; //one-hundred million
    static std::atomic<double> inline fit_time = 0.0;

    static constexpr double K = 0.0884956;
    static constexpr double phi_1 = 2.8;
    static constexpr double phi_2 = 1.3;
    static int inline __num_features;
    static std::vector<std::string> inline __input_vars;
    static std::vector<std::string> inline __unary_operators;
    static std::vector<std::string> inline __binary_operators;
    static size_t inline num_unary_ops;
    static size_t inline num_binary_ops;
    static size_t inline num_leaf_operands;
    static std::unordered_set<std::string> inline __unary_operators_uset;
    static std::unordered_set<std::string> inline __binary_operators_uset;
    static std::unordered_set<std::string> inline __bad_operators_uset;
    static std::vector<std::string> inline __operators;
    static std::vector<std::string> inline __other_tokens; //denotes the other leaf-nodes besides the input variables
    static std::vector<std::string> inline __tokens;
    Eigen::VectorXd params; //store the parameters of the expression of the current episode after it's completed
    static Data inline data;

    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> vel_dist, pos_dist;
    static std::uniform_int_distribution<int> inline unary_dist; // A random integer generator which generates an index corresponding to a unary operator
    static std::uniform_int_distribution<int> inline binary_dist; // A random integer generator which generates an index corresponding to a binary operator
    static std::uniform_int_distribution<int> inline leaf_dist; // A random integer generator which generates an index corresponding to an operand

    static std::once_flag inline initialization_flag;  // Flag for std::call_once
    static std::unordered_map<std::string, std::pair<std::string, std::string>> inline feature_mins_maxes;

    size_t reserve_amount;
    int num_fit_iter;
    int num_objectives;
    double SNE_curr;
    std::vector<double> SNE_curr_vec;
    std::string fit_method;
    std::string fit_grad_method;
    std::unordered_map<std::string, Eigen::VectorXd> subs_dict;

    bool cache;
    bool use_const_pieces;
    std::vector<std::vector<int>> stack;
    std::vector<int> depth, num_binary, num_leaves, idx;
    static std::unordered_map<bool, std::unordered_map<bool, std::unordered_map<bool, std::vector<std::string>>>> inline una_bin_leaf_legal_moves_dict;
    std::vector<int> simplify_grasp;

    std::vector<int> n; //depth of RPN/PN trees
    std::string expression_type, expression_string;
    size_t num_consts_diff;
    static std::mutex inline thread_locker; //static because it needs to protect static members
    std::vector<std::vector<std::string>> pieces, temp_pieces; // Create the empty expression list and backup
    std::vector<std::string> derivat;// Vector to store the derivative.
    bool visualize_exploration, is_primary;
    std::vector<std::vector<std::string>> (*diffeq)(Board&, bool); //differential equation we want to solve
    size_t num_diff_eqns; //number of equations in the system `diffeq`
    std::vector<std::vector<std::string>> diffeq_result;
    double isConstTol;
    bool simplify_original;
    bool mustHaveAllFeatures;
    std::vector<std::vector<std::string>> customFeatures;
    bool graph_eval;
    bool complete_Tree;
    bool add_additive;
    std::vector<int> maxSize; //max size (number of tokens) for each expression
    std::vector<std::vector<std::string>> additiveCorrections; //Starting-point, s.t candidate-expression += additiveCorrections (user-implemented though)

    Board(std::vector<std::vector<std::string>> (*diffeq)(Board&, bool),
          size_t num_diff_eqns,
          bool primary = true,
          const std::vector<int>& depth = {},
          const std::string& expression_type = "prefix",
          size_t num_consts_diff = 0,
          std::string fitMethod = "LevenbergMarquardt",
          int numFitIter = 1,
          std::string fitGradMethod = "naive_numerical",
          const Eigen::MatrixXd& theData = {},
          bool visualize_exploration = false,
          bool cache = false,
          bool const_tokens = false,
          double isConstTol = 1e-1,
          bool use_const_pieces = false,
          bool simplifyOriginal = true,
          int numDataCols = 0,
          bool must_have_all_features = true,
          const std::vector<std::vector<std::string>>& custom_features = {},
          std::vector<int> max_size = {},
          const std::vector<std::vector<std::string>>& additive_corrections = {},
          bool graphEval = false,
          bool completeTree = false,
          const std::vector<std::string> bad_operators = {}) :
            gen{rd()}, vel_dist{-1.0, 1.0}, pos_dist{0.0, 1.0}, num_fit_iter{numFitIter}, fit_method{fitMethod}, fit_grad_method{fitGradMethod}, n{depth}, is_primary{primary}, simplify_original{simplifyOriginal}, mustHaveAllFeatures{must_have_all_features}, customFeatures{custom_features}, graph_eval{graphEval}, complete_Tree{completeTree}, maxSize{max_size}, additiveCorrections{additive_corrections}
    {
        assert(n.size());
        assert(((!maxSize.size()) || (maxSize.size() && maxSize.size() == n.size())) && "if `maxSize` is not empty it much be equal in size to the depth-vector `n`");
        assert(((!additiveCorrections.size()) || (additiveCorrections.size() && additiveCorrections.size() == n.size())) && "if `additiveCorrections` is not empty it must be equal in size to the depth-vector `n`");
        this->add_additive = additiveCorrections.size();
        this->num_objectives = n.size();
        int max_n = n[0];
        int counter = 0;
        for (int i: n)
        {
            if (i > 30)
            {
                throw(std::runtime_error("Complexity cannot be larger than 30, sorry!"));
            }
            this->pieces.emplace_back();
            this->stack.emplace_back();
            this->depth.emplace_back();
            this->num_binary.emplace_back();
            this->num_leaves.emplace_back();
            this->idx.emplace_back();
            this->pieces[counter].reserve(2*std::pow(2,i)-1);
            if (i > max_n)
            {
                max_n = i;
            }
            this->stack[counter++].reserve(i);
        }

        this->expression_type = expression_type;
        this->num_consts_diff = num_consts_diff;
        this->use_const_pieces = use_const_pieces;
        this->visualize_exploration = visualize_exploration;
        this->reserve_amount = 2*std::pow(2,max_n)-1;
        this->cache = cache;
        this->diffeq = diffeq;
//        this->diffeq_result = {};
//        printf("this->diffeq_result = %lu\n", diffeq_result.size());
        this->num_diff_eqns = num_diff_eqns;
        this->isConstTol = isConstTol;

        if (is_primary)
        {
            std::call_once(initialization_flag, [&]()
            {
                if (use_const_pieces)
                {
                    assert(const_tokens);
                }
                Board::data = theData;
                assert((Board::data.num_rows > 0));
                Board::__num_features = Board::data[0].size() - numDataCols;
                assert(Board::__num_features > 0);
                printf("Number of features = %d\n", Board::__num_features);
                Board::__input_vars.clear();
                Board::expression_dict.clear();
                Board::__input_vars.reserve(Board::__num_features);
                for (auto i = 0; i < Board::__num_features; i++)
                {
                    Board::__input_vars.push_back("x"+std::to_string(i));
                }
                Board::__unary_operators = {"~", "log", "ln", "exp", "cos", "sin", "sqrt", "asin", "arcsin", "acos", "arccos", "tanh", "sech"};
                Board::__binary_operators = {"+", "-", "*", "/", "^"};
                std::copy(Board::__unary_operators.begin(), Board::__unary_operators.end(), std::inserter(Board::__unary_operators_uset, Board::__unary_operators_uset.end()));
                std::copy(Board::__binary_operators.begin(), Board::__binary_operators.end(), std::inserter(Board::__binary_operators_uset, Board::__binary_operators_uset.end()));
                for (const std::string& i: bad_operators){Board::__bad_operators_uset.emplace(i);}
                for (const std::string& i: Board::__unary_operators_uset) {std::cout << i << ' ';}puts("");
                for (const std::string& i: Board::__binary_operators_uset) {std::cout << i << ' ';}puts("");
                Board::__operators.clear();
                for (std::string& i: Board::__unary_operators)
                {
                    Board::__operators.push_back(i);
                }
                for (std::string& i: Board::__binary_operators)
                {
                    Board::__operators.push_back(i);
                }

                if (const_tokens) //then add non-optimizable constants only
                {
                    Board::__other_tokens = {"0", "1", "2", "4"};
                    for (const std::string& i: Board::__input_vars)
                    {
                        std::string minCoeff_i = std::to_string(Board::data[i].minCoeff());
                        std::string maxCoeff_i = std::to_string(Board::data[i].maxCoeff());
                        Board::__other_tokens.push_back(minCoeff_i); //add smallest element
                        Board::__other_tokens.push_back(maxCoeff_i); //add largest element
                        feature_mins_maxes[i] = std::make_pair(minCoeff_i, maxCoeff_i);
                        std::cout << "feature_mins_maxes[" << i << "] = " << feature_mins_maxes[i] << '\n';
                    }
                }
                if (this->use_const_pieces) //then add "const"
                {
                    Board::__other_tokens.push_back("const");
                }
                for (size_t i = 0; i < this->num_consts_diff; i++) //then add "const0", "const1", ..., "const{this->num_consts_diff-1}"
                {
                    Board::__other_tokens.push_back("const"+std::to_string(i));
                }
                Board::__tokens = Board::__operators;

                for (auto& i: this->Board::__input_vars)
                {
                    Board::__tokens.push_back(i);
                }
                for (auto& i: Board::__other_tokens)
                {
                    Board::__tokens.push_back(i);
                }
                assert((!(this->num_consts_diff || this->use_const_pieces)) || (Board::__tokens.back().compare(0, 5, "const") == 0));
                Board::una_bin_leaf_legal_moves_dict.clear();
                /*
                 [true][true][true]
                 [true][true][false]
                 [true][false][true]
                 [true][false][false]
                 [false][true][true]
                 [false][true][false]
                 [false][false][true]
                 [false][false][false]
                 */
                Board::una_bin_leaf_legal_moves_dict[true][true][true] = Board::__tokens;
                Board::una_bin_leaf_legal_moves_dict[true][true][false] = Board::__operators;
                Board::una_bin_leaf_legal_moves_dict[true][false][true] = Board::__unary_operators; //1
                Board::una_bin_leaf_legal_moves_dict[true][false][false] = Board::__unary_operators;
                Board::una_bin_leaf_legal_moves_dict[false][true][true] = Board::__binary_operators; //2
                Board::una_bin_leaf_legal_moves_dict[false][true][false] = Board::__binary_operators;

                for (const std::string &i: Board::__input_vars)
                {
                    Board::una_bin_leaf_legal_moves_dict[true][false][true].push_back(i); //1
                    Board::una_bin_leaf_legal_moves_dict[false][true][true].push_back(i); //2
                    Board::una_bin_leaf_legal_moves_dict[false][false][true].push_back(i); //3
                }
                for (const std::string &i: Board::__other_tokens)
                {
                    Board::una_bin_leaf_legal_moves_dict[true][false][true].push_back(i); //1
                    Board::una_bin_leaf_legal_moves_dict[false][true][true].push_back(i); //2
                    Board::una_bin_leaf_legal_moves_dict[false][false][true].push_back(i); //3
                }
                Board::num_unary_ops = Board::__unary_operators.size();
                Board::num_binary_ops = Board::__binary_operators.size();
                Board::num_leaf_operands = Board::una_bin_leaf_legal_moves_dict[false][false][true].size();
                std::cout << "Board::leaf_operands = " << Board::una_bin_leaf_legal_moves_dict[false][false][true] << '\n';
                std::cout << "Board::num_unary_ops = " << Board::num_unary_ops << '\n';
                std::cout << "Board::num_binary_ops = " << Board::num_binary_ops << '\n';
                std::cout << "Board::num_leaf_operands = " << Board::num_leaf_operands << '\n';
                std::cout << "Board::__tokens.size() = " << Board::__tokens.size() << '\n';
                Board::unary_dist = std::uniform_int_distribution<int>(0, Board::num_unary_ops - 1);
                Board::binary_dist = std::uniform_int_distribution<int>(0, Board::num_binary_ops - 1);
                Board::leaf_dist = std::uniform_int_distribution<int>(0, Board::num_leaf_operands - 1);
            });

        }
    }

    std::string print_expression_params()
    {
        std::stringstream outstringstream;
        outstringstream << '{';
        constexpr const char* const_label = "const";
        //below is really just `for(type i = this->num_consts_diff; i < this->params.size(); i++)`,
        //where `type` is just the type of `this->params.size()`
        for (decltype(this->params.size()) i = static_cast<decltype(this->params.size())>(this->num_consts_diff); i < this->params.size(); i++)
        {
            outstringstream << '(' << const_label+std::to_string(i) << ", " << this->params[i] << "), ";
        }
        outstringstream << '}';
        std::string x = outstringstream.str();
        if (x.size() > 2)
        {
            return x.replace(x.size() - 3, std::string::npos, "}");
        }
        return x;
    }

    std::string print_diff_params()
    {
        std::stringstream outstringstream;
        outstringstream << '{';
        constexpr const char* const_label = "const";
        for (size_t i = 0; i < this->num_consts_diff; i++)
        {
            outstringstream << '(' << const_label+std::to_string(i) << ", " << this->params[i] << "), ";
        }
        outstringstream << '}';
        std::string x = outstringstream.str();
        if (x.size() > 2)
        {
            return x.replace(x.size() - 3, std::string::npos, "}");
        }
        return x;
    }

    std::string operator[](size_t index) const
    {
        assert((index < Board::__tokens.size()));
        return Board::__tokens[index];
//        if (index < Board::__tokens.size())
//        {
//            return Board::__tokens[index];
//        }
//        throw std::out_of_range("Index out of range");
    }

    int __num_binary_ops(int i) const
    {
        assert(pieces.size() <= INT_MAX);
        assert((i >= 0) && (i < static_cast<int>(pieces.size())));
        int count = 0;
        for (const std::string& token : pieces[i])
        {
            if (std::find(Board::__binary_operators.begin(), Board::__binary_operators.end(), token) != Board::__binary_operators.end())
            {
                count++;
            }
        }
        return count;
    }

    int __num_unary_ops(int i) const
    {
        assert(pieces.size() <= INT_MAX);
        assert((i >= 0) && (i < static_cast<int>(pieces.size())));
        int count = 0;
        for (const std::string& token : pieces[i])
        {
            if (std::find(Board::__unary_operators.begin(), Board::__unary_operators.end(), token) != Board::__unary_operators.end())
            {
                count++;
            }
        }
        return count;
    }

    int __num_leaves(int i) const
    {
        int count = 0;
        assert(pieces.size() <= INT_MAX);
        assert((i >= 0) && (i < static_cast<int>(pieces.size())));
        for (const std::string& token : pieces[i])
        {
            if (!is_unary(token) && !is_binary(token))
            {
                count++;
            }
        }
        return count;
    }
    
    int __num_consts() const
    {
        int count = 0;
        for (const std::vector<std::string>& vec: this->pieces)
        {
            for (const std::string& piece: vec)
            {
                if (piece.compare(0, 5, "const") == 0)
                {
                    count++;
                }
            }
        }
        return count;
    }

    int __num_consts(int jdx) const
    {
        assert(pieces.size() <= INT_MAX);
        assert((jdx >= 0) && (jdx < static_cast<int>(pieces.size())));
        int count = 0;
        for (const std::string& piece: this->pieces[jdx])
        {
            if (piece.compare(0, 5, "const") == 0)
            {
                count++;
            }
        }
        return count;
    }

    bool is_unary(const std::string& token) const
    {
        return ((Board::__unary_operators_uset.find(token) != Board::__unary_operators_uset.end()) || (token == "abs"));
    }

    bool is_binary(const std::string& token) const
    {
        return (Board::__binary_operators_uset.find(token) != Board::__binary_operators_uset.end());
    }
    
    bool is_bad_op(const std::string& token) const
    {
        return (Board::__bad_operators_uset.find(token) != Board::__bad_operators_uset.end());
    }

    bool is_operator(const std::string& token) const
    {
        return (is_binary(token) || is_unary(token));
    }

    bool is_const(const std::string& token) const
    {
        return ((!is_unary(token)) && (!is_binary(token)));
    }

    std::string simplifyString(const std::string& x)
    {
        if ((x.size() == 2) && (x[0] == '-') && (x[1] == '0')) //"-0" -> "0"
        {
            return "0";
        }
        unsigned long jdx = x.find(".");
        if (jdx == std::string::npos) //if there's no '.' in x
        {
            return x;
        }
        for (unsigned long i = jdx + 1; i < x.size(); i++) //checking if all the characters after the decimal in x are 0; if not, then return x
        {
            if (x[i] != '0')
            {
                return x;
            }
        }
        std::string temp = x.substr(0, jdx);
        if ((temp.size() == 2) && (temp[0] == '-') && (temp[1] == '0')) //"-0.0000" (repeating) -> "0"
        {
            return "0";
        }
        return temp; //"x.0000000" (repeating) -> "x"
    }

    //((1+5) * 1) -> 1 5 1 * +, + 1 * 5 1 -> traverse into a tree data-structure -> apply simplification algorithm
    //-> traverse tree again to get prefix/postfix simplified expression -> convert to infix
    //prefix/postifx -> simplified prefix/postfix
    void graspSimplifyPrefixHelper(std::vector<std::string>& expression, int low, int up, std::vector<int>& grasp, std::vector<std::string>& new_expression, bool setGRvar = false)
    {
        if (!setGRvar)
        {
            grasp.clear();
            setPrefixGR(expression, grasp);
        }
    //    print_container(expression, low, up);
    //    print_container(new_expression, 0, new_expression.size() - 1);
        if (expression[low] == "+" || expression[low] == "-") // +/- x y
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]);
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true);
            int first_arg_idx_high = new_expression.size();
            graspSimplifyPrefixHelper(expression, temp+1, temp+1+grasp[temp+1], grasp, new_expression, true);
            int second_arg_idx_high = new_expression.size();
            int step;

            if (checkEqual(new_expression[first_arg_idx_high], "0")) //+/- x 0 -> x
            {
                //puts("hi 177");
                if (first_arg_idx_high == static_cast<int>(new_expression.size()) - 1)
                {
                    new_expression.pop_back();
                }
                else
                {
                    new_expression.erase(new_expression.begin() + first_arg_idx_high, new_expression.end());
                }
                new_expression.erase(new_expression.begin() + op_idx); //remove +/- operator at beginning
            }

            else if (checkEqual(new_expression[first_arg_idx_low], "0"))
            {
                if (expression[low] == "+") //+ 0 y -> y
                {
                    //puts("hi 176");
                    new_expression.erase(new_expression.begin() + op_idx, new_expression.begin() + first_arg_idx_high); //remove '+' and '0'
                }
                else //- 0 y -> ~ y
                {
                    //puts("hi 184");
                    new_expression[op_idx] = "~";
                    new_expression.erase(new_expression.begin() + first_arg_idx_low); //'0'
                }
            }
            
            else if ((new_expression[first_arg_idx_low] == "nan") || (new_expression[first_arg_idx_high] == "nan")) //+/- nan x -> nan, +/- x nan -> nan, +/- nan nan -> nan
            {
                //puts("hi 273");
                assert(new_expression[op_idx] == expression[low]);
                new_expression[op_idx] = "nan"; //change "+/-" to "nan";
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.begin() + second_arg_idx_high);
            }
            
            else if ((expression[low] == "-") && ((step = (second_arg_idx_high - first_arg_idx_high)) == (first_arg_idx_high - first_arg_idx_low)) && (areExpressionRangesEqual(first_arg_idx_low, first_arg_idx_high, step, new_expression))) //- x x
            {
                //puts("hi 221");
                assert(new_expression[op_idx] == expression[low]);
                new_expression[op_idx] = "0"; //change "-" to "0";
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.begin() + second_arg_idx_high);
            }
        }
        else if (expression[low] == "*") //* x y
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); //*
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); //* x
            int first_arg_idx_high = new_expression.size();
            graspSimplifyPrefixHelper(expression, temp+1, temp+1+grasp[temp+1], grasp, new_expression, true); //* x y
            //int second_arg_idx_high = new_expression.size();
            //int step;
            if ((new_expression[first_arg_idx_high] == "nan") || (new_expression[first_arg_idx_low] == "nan")) //* nan x -> nan, * x nan -> nan, * nan nan -> nan
            {
                //puts("hi 300");
                new_expression[op_idx] = "nan"; //change '*' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_high], "0")) //* x 0 -> 0 (because, since prefix operators come at the beginning, if the beginning of the second argument of '*' is 0, then the whole second argument MUST be 0, therefore the expression reduces to * x 0, which is 0)
            {
                //puts("hi 239");
                new_expression[op_idx] = "0"; //change '*' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) //* 0 x -> 0
            {
                //puts("hi 245");
                new_expression[op_idx] = "0"; //change '*' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_high], "1")) //* x 1 -> x (because, since prefix operators come at the beginning, if the beginning of the second argument of '*' is 1, then the whole second argument MUST be 1, therefore the expression reduces to * x 1, which is x)
            {
                //puts("hi 251");
                //erase the '1' at the end
                if (first_arg_idx_high == static_cast<int>(new_expression.size()) - 1)
                {
                    new_expression.pop_back();
                }
                else
                {
                    new_expression.erase(new_expression.begin() + first_arg_idx_high, new_expression.end());
                }
                new_expression.erase(new_expression.begin() + op_idx); //erase the '*'
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "1")) //* 1 x -> x
            {
                //puts("hi 265");
                new_expression.erase(new_expression.begin() + op_idx, new_expression.begin() + op_idx + 2); //erase the '*' and the '1'
            }
            else if (checkEqual(new_expression[first_arg_idx_high], "-1")) //* x -1 -> ~ x (because, since prefix operators come at the beginning, if the beginning of the second argument of '*' is -1, then the whole second argument MUST be -1, therefore the expression reduces to * x -1, which is ~ x)
            {
    //            puts("hi 382");
                //erase the '-1' at the end
                if (first_arg_idx_high == static_cast<int>(new_expression.size()) - 1)
                {
                    new_expression.pop_back();
                }
                else
                {
                    new_expression.erase(new_expression.begin() + first_arg_idx_high, new_expression.end());
                }
                new_expression[op_idx] = "~"; //change the '*' to a '~'
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "-1")) //* -1 x -> ~ x
            {
    //            puts("hi 396");
                new_expression[op_idx] = "~"; //change the '*' to a '~'
                new_expression.erase(new_expression.begin() + op_idx + 1); //erase the '-1'
            }
        }
        else if (expression[low] == "/") // / x y
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // /
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // / x
            int first_arg_idx_high = new_expression.size();
            graspSimplifyPrefixHelper(expression, temp+1, temp+1+grasp[temp+1], grasp, new_expression, true); // / x y
            int second_arg_idx_high = new_expression.size();
            int step;
            
            if ((new_expression[first_arg_idx_low] == "nan") || (new_expression[first_arg_idx_high] == "nan")) // / nan x -> nan, / x nan -> nan, / nan nan -> nan
            {
                //puts("hi 350");
                new_expression[op_idx] = "nan"; //change '/' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((checkEqual(new_expression[first_arg_idx_low], "0")) && (checkEqual(new_expression[first_arg_idx_high], "0"))) // / 0 0 -> nan
            {
                //puts("hi 290");
                new_expression[op_idx] = "nan"; //change '/' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_high], "0")) // / x 0 -> nan (for now, because, since prefix operators come at the beginning, if the beginning of the second argument of '/' is 0, then the whole second argument MUST be 0, therefore the expression reduces to / x 0, which is, for now, assumed to be nan for simplicity)
            {
                //puts("hi 282");
                //TODO: need to come up with a more robust way that actually checks if this is nan anywhere;
                //for now we weed it out because annoying not to; giving this up seems like the better deal...
                new_expression[op_idx] = "nan";//(new_expression[first_arg_idx_low] != "~") ? "inf": "-inf"; //change '/' to 'inf' or '-inf'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) // / 0 x -> 0
            {
                //puts("hi 295");
                new_expression[op_idx] = "0"; //change '/' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_high], "1")) // / x 1 -> x (because, since prefix operators come at the beginning, if the beginning of the second argument of '/' is 1, then the whole second argument MUST be 1, therefore the expression reduces to / x 1, which is x)
            {
                //puts("hi 301");
                //erase the '1' at the end
                if (first_arg_idx_high == static_cast<int>(new_expression.size()) - 1)
                {
                    new_expression.pop_back();
                }
                else
                {
                    new_expression.erase(new_expression.begin() + first_arg_idx_high, new_expression.end());
                }
                new_expression.erase(new_expression.begin() + op_idx); //erase the '/'
            }
            else if (checkEqual(new_expression[first_arg_idx_high], "-1")) // / x -1 -> ~ x (because, since prefix operators come at the beginning, if the beginning of the second argument of '/' is -1, then the whole second argument MUST be -1, therefore the expression reduces to / x -1, which is -x)
            {
                //puts("hi 454");
                //erase the '-1' at the end
                if (first_arg_idx_high == static_cast<int>(new_expression.size()) - 1)
                {
                    new_expression.pop_back();
                }
                else
                {
                    new_expression.erase(new_expression.begin() + first_arg_idx_high, new_expression.end());
                }
                new_expression[op_idx] = "~"; //change the '/' to a '~'
            }
            else if ((expression[low] == "/") && ((step = (second_arg_idx_high - first_arg_idx_high)) == (first_arg_idx_high - first_arg_idx_low)) && (areExpressionRangesEqual(first_arg_idx_low, first_arg_idx_high, step, new_expression))) // / x x
            {
                //puts("hi 315");
                assert(new_expression[op_idx] == expression[low]);
                new_expression[op_idx] = "1"; //change "-" to "1";
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.begin() + second_arg_idx_high);
            }

            //TODO:
                /*
                x*y       y
                ---  -->  -
                x*z       z
                */
        }
        else if (expression[low] == "^") // ^ x y
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // ^
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // ^ x
            int first_arg_idx_high = new_expression.size();
            graspSimplifyPrefixHelper(expression, temp+1, temp+1+grasp[temp+1], grasp, new_expression, true); // ^ x y
            //int second_arg_idx_high = new_expression.size();
            //int step;
            if ((new_expression[first_arg_idx_low] == "nan") || (new_expression[first_arg_idx_high] == "nan")) // ^ nan x -> nan, ^ x nan -> nan, ^ nan nan -> nan
            {
                //puts("hi 417");
                new_expression[op_idx] = "nan"; //change '^' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_high], "0")) //^ x 0 -> 1 (because, since prefix operators come at the beginning, if the beginning of the second argument of '^' is 0, then the whole second argument MUST be 0, therefore the expression reduces to ^ x 0, which is 1)
            {
                //puts("hi 334");
                new_expression[op_idx] = "1"; //change '^' to '1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) // ^ 0 x -> nan (for now)
            {
                //puts("hi 340");
                //TODO: need to come up with a more robust way that actually checks if this is nan anywhere;
                //for now we weed it out because annoying not to; giving this up seems like the better deal...
                new_expression[op_idx] = "nan";//new_expression[op_idx] = "0"; //change '^' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_high], "1")) // ^ x 1 -> x (because, since prefix operators come at the beginning, if the beginning of the second argument of '^' is 1, then the whole second argument MUST be 1, therefore the expression reduces to ^ x 1, which is 1)
            {
                //puts("hi 346");
                //erase the '1' at the end
                if (first_arg_idx_high == static_cast<int>(new_expression.size()) - 1)
                {
                    new_expression.pop_back();
                }
                else
                {
                    new_expression.erase(new_expression.begin() + first_arg_idx_high, new_expression.end());
                }
                new_expression.erase(new_expression.begin() + op_idx); //erase the '^'
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "1")) // ^ 1 x -> 1
            {
                //puts("hi 360");
                new_expression[op_idx] = "1"; //change '^' to '1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        else if (expression[low] == "cos") // cos x
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // cos
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // cos x
            if (new_expression[first_arg_idx_low] == "nan") // cos nan -> nan
            {
                //puts("hi 464");
                new_expression[op_idx] = "nan"; //change 'cos' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) // cos 0 -> 1
            {
    //            puts("hi 374");
                new_expression[op_idx] = "1"; //change 'cos' to '1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "inf" || new_expression[first_arg_idx_low] == "-inf") // cos +/- inf -> nan
            {
    //            puts("hi 554");
                new_expression[op_idx] = "nan"; //change 'cos' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "inf")) // cos ~ inf -> nan
            {
    //            puts("hi 560");
                new_expression[op_idx] = "nan"; //change 'cos' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        else if (expression[low] == "sin") // sin x
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // sin
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // sin x
            if (new_expression[first_arg_idx_low] == "nan") // sin nan -> nan
            {
                //puts("hi 484");
                new_expression[op_idx] = "nan"; //change 'sin' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) // sin 0 -> 0
            {
                //puts("hi 388");
                new_expression[op_idx] = "0"; //change 'sin' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "inf" || new_expression[first_arg_idx_low] == "-inf") // sin +/- inf -> nan
            {
    //            puts("hi 586");
                new_expression[op_idx] = "nan"; //change 'sin' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "inf")) // sin ~ inf -> nan
            {
    //            puts("hi 592");
                new_expression[op_idx] = "nan"; //change 'sin' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        else if (expression[low] == "tanh") // tanh x
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // tanh
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // tanh x
            if (new_expression[first_arg_idx_low] == "nan") // tanh nan -> nan
            {
                //puts("hi 504");
                new_expression[op_idx] = "nan"; //change 'tanh' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) // tanh 0 -> 0
            {
                //puts("hi 402");
                new_expression[op_idx] = "0"; //change 'tanh' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "inf") // tanh inf -> 1
            {
                //puts("hi 408");
                new_expression[op_idx] = "1"; //change 'tanh' to '1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "-inf") // tanh -inf -> -1
            {
                //puts("hi 414");
                new_expression[op_idx] = "-1"; //change 'tanh' to '-1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "inf")) // tanh ~ inf -> 1
            {
    //            puts("hi 421");
                new_expression[op_idx] = "-1"; //change 'tanh' to '-1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        else if (expression[low] == "sech") // sech x
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // sech
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // sech x

            if (new_expression[first_arg_idx_low] == "nan") // sech nan -> nan
            {
                //puts("hi 548");
                new_expression[op_idx] = "nan"; //change 'sech' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) // sech 0 -> 1
            {
                //puts("hi 416");
                new_expression[op_idx] = "1"; //change 'sech' to '1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "inf") // sech inf -> 0
            {
                //puts("hi 440");
                new_expression[op_idx] = "0"; //change 'sech' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "-inf") // sech -inf -> 0
            {
    //            puts("hi 446");
                new_expression[op_idx] = "0"; //change 'sech' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "inf")) // sech ~ inf -> 0
            {
    //            puts("hi 452");
                new_expression[op_idx] = "0"; //change 'sech' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        else if (expression[low] == "~") // ~ x
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // ~
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // ~ x

            if (new_expression[first_arg_idx_low] == "nan") // ~ nan -> nan
            {
                //puts("hi 592");
                new_expression[op_idx] = "nan"; //change '~' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) // ~ 0 -> 0
            {
    //            puts("hi 466");
                new_expression[op_idx] = "0"; //change '~' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "1")) // ~ 1 -> -1
            {
                //puts("hi 696");
                new_expression[op_idx] = "-1"; //change '~' to '-1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "-1")) // ~ -1 -> 1
            {
                //puts("hi 702");
                new_expression[op_idx] = "1"; //change '~' to '1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "inf") // ~ inf -> -inf
            {
                //puts("hi 507");
                new_expression[op_idx] = "-inf"; //change '~' to '-inf'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "-inf") // ~ -inf -> inf
            {
                //puts("hi 702");
                new_expression[op_idx] = "inf"; //change '~' to 'inf'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "inf")) // ~ ~ inf -> inf
            {
                //puts("hi 708");
                new_expression[op_idx] = "inf"; //change '~' to 'inf'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        else if (expression[low] == "exp") // exp x
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // exp
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // exp x
            
            if (new_expression[first_arg_idx_low] == "nan") // exp nan -> nan
            {
                //puts("hi 618");
                new_expression[op_idx] = "nan"; //change 'exp' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) // exp 0 -> 1
            {
                //puts("hi 521");
                new_expression[op_idx] = "1"; //change 'exp' to '1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "inf") // exp inf -> inf
            {
                //puts("hi 527");
                new_expression[op_idx] = "inf"; //change 'exp' to 'inf'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "-inf") // exp -inf -> 0
            {
                //puts("hi 741");
                new_expression[op_idx] = "0"; //change 'exp' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "inf")) // exp ~ inf -> 0
            {
                //puts("hi 747");
                new_expression[op_idx] = "0"; //change 'exp' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        else if ((expression[low] == "log") || (expression[low] == "ln")) // ln x
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // ln
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // ln x
            
            if (new_expression[first_arg_idx_low] == "nan") // ln nan -> nan
            {
                //puts("hi 726");
                new_expression[op_idx] = "nan"; //change 'ln' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) // ln 0 -> -inf
            {
                //puts("hi 732");
                new_expression[op_idx] = "-inf"; //change 'ln' to '-inf'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "inf") // ln inf -> inf
            {
                //puts("hi 738");
                new_expression[op_idx] = "inf"; //change 'ln' to 'inf'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "-inf") // ln -inf -> nan
            {
                //puts("hi 780");
                new_expression[op_idx] = "nan"; //change 'ln' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "inf")) // ln ~ inf -> nan
            {
                //puts("hi 786");
                new_expression[op_idx] = "nan"; //change 'ln' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        else if ((expression[low] == "asin") || (expression[low] == "arcsin")) // asin x
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // asin
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // asin x
            
            if (new_expression[first_arg_idx_low] == "nan") // asin nan -> nan
            {
                //puts("hi 753");
                new_expression[op_idx] = "nan"; //change 'asin' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) // asin 0 -> 0
            {
                //puts("hi 759");
                new_expression[op_idx] = "0"; //change 'asin' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "inf") // asin inf -> nan
            {
                //puts("hi 765");
                new_expression[op_idx] = "nan"; //change 'asin' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "-inf") // asin -inf -> nan
            {
                //puts("hi 819");
                new_expression[op_idx] = "nan"; //change 'asin' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "inf")) // asin ~ inf -> nan
            {
                //puts("hi 825");
                new_expression[op_idx] = "nan"; //change 'asin' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        else if ((expression[low] == "acos") || (expression[low] == "arccos")) // acos x
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // acos
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // acos x
            
            if (new_expression[first_arg_idx_low] == "nan") // acos nan -> nan
            {
                //puts("hi 781");
                new_expression[op_idx] = "nan"; //change 'acos' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "1")) // acos 1 -> 0
            {
                //puts("hi 787");
                new_expression[op_idx] = "0"; //change 'acos' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "inf") // arccos inf -> nan
            {
                //puts("hi 793");
                new_expression[op_idx] = "nan"; //change 'acos' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "-inf") // acos -inf -> nan
            {
                //puts("hi 858");
                new_expression[op_idx] = "nan"; //change 'acos' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "inf")) // acos ~ inf -> nan
            {
                //puts("hi 864");
                new_expression[op_idx] = "nan"; //change 'acos' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        else if (expression[low] == "sqrt") // sqrt x
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // sqrt
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // sqrt x
            
            if (new_expression[first_arg_idx_low] == "nan") // sqrt nan -> nan
            {
    //            puts("hi 808");
                new_expression[op_idx] = "nan"; //change 'sqrt' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) // sqrt 0 -> 0
            {
    //            puts("hi 814");
                new_expression[op_idx] = "0"; //change 'sqrt' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "1")) // sqrt 1 -> 1
            {
    //            puts("hi 820");
                new_expression[op_idx] = "1"; //change 'sqrt' to '1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "inf") // sqrt inf -> inf
            {
    //            puts("hi 826");
                new_expression[op_idx] = "inf"; //change 'sqrt' to 'inf'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "-inf") // sqrt -inf -> nan
            {
    //            puts("hi 903");
                new_expression[op_idx] = "nan"; //change 'sqrt' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "inf")) // sqrt ~ inf -> nan
            {
    //            puts("hi 909");
                new_expression[op_idx] = "nan"; //change 'sqrt' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "-1")) // sqrt -1 -> nan
            {
    //            puts("hi 915");
                new_expression[op_idx] = "nan"; //change 'sqrt' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "1")) // sqrt ~ 1 -> nan
            {
    //            puts("hi 921");
                new_expression[op_idx] = "nan"; //change 'sqrt' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        
        else if (expression[low] == "abs") // abs x
        {
            new_expression.push_back(expression[low]); // abs
            int temp = low+1+grasp[low+1];
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // abs x
        }
        else
        {
            for (int i = low; i <= up; i++)
            {
                //assert(i < expression.size() && i >= 0);
                new_expression.push_back(expression[i]);
            }
        }
    }

    void graspSimplifyPrefix(std::vector<std::string>& expression, int low, int up, std::vector<int>& grasp)
    {
        std::vector<std::string> new_expression;
        new_expression.reserve(expression.size());
        graspSimplifyPrefixHelper(expression, low, up, grasp, new_expression, false);
        expression = new_expression;
    }

    //scans entire `expression` for the following:
    //     bin_op number1 number2 -> numberResult
    //     un_op number1 -> numberResult
    //the function repeatedly iterates over `expression` until no more
    //instances of the above are found.
    //+ cos - + 1 3 x + * cos 2 sin 3 + arcsin - 8 7 sin tanh cos * 3 4
    //+ cos - 4 x + * cos 2 sin 3 + arcsin - 8 7 sin tanh cos * 3 4
    //+ cos - 4 x + * -0.42 sin 3 + arcsin - 8 7 sin tanh cos * 3 4
    //+ cos - 4 x + * -0.42 0.41 + arcsin - 8 7 sin tanh cos * 3 4
    void simplifyPN_Helper(std::vector<std::string>& expression)
    {
        if (expression.empty()) return;

        bool isdouble1, isdouble2, isConst1, isConst2;
        thread_local std::vector<std::string> temp;
        temp.clear();
        temp.reserve(expression.size());

        for (size_t i = 0; i < expression.size(); ++i)
        {
            // Convenience bounds flags
            bool has1 = (i + 1 < expression.size());
            bool has2 = (i + 2 < expression.size());

            // ==== BINARY OPERATORS ====
            if (is_binary(expression[i]))
            {
                // If we don't have two operands, just copy token and move on
                if (!has1 || !has2)
                {
                    temp.push_back(expression[i]);
                    continue;
                }

                isdouble1 = isdouble(expression[i+1]);
                isdouble2 = isdouble(expression[i+2]);

                if (isdouble1 && isdouble2)
                {
                    if (expression[i] == "+")
                    {
                        temp.push_back(simplifyString(
                            to_string_general(Stod(expression[i+1]) + Stod(expression[i+2]))
                        ));
                        i += 2;
                        continue;
                    }
                    else if (expression[i] == "-")
                    {
                        temp.push_back(simplifyString(
                            to_string_general(Stod(expression[i+1]) - Stod(expression[i+2]))
                        ));
                        i += 2;
                        continue;
                    }
                    else if (expression[i] == "*")
                    {
                        temp.push_back(simplifyString(
                            to_string_general(Stod(expression[i+1]) * Stod(expression[i+2]))
                        ));
                        i += 2;
                        continue;
                    }
                    else if (expression[i] == "/")
                    {
                        temp.push_back(simplifyString(
                            to_string_general(Stod(expression[i+1]) / Stod(expression[i+2]))
                        ));
                        i += 2;
                        continue;
                    }
                    else if (expression[i] == "^")
                    {
                        temp.push_back(simplifyString(
                            to_string_general(std::pow(Stod(expression[i+1]), Stod(expression[i+2])))
                        ));
                        i += 2;
                        continue;
                    }
                }

                isConst1 = is_const(expression[i+1]);
                isConst2 = is_const(expression[i+2]);

                if ((isConst1 && isConst2) &&
                    ((expression[i+1].find("nan") != std::string::npos) ||
                     (expression[i+2].find("nan") != std::string::npos)))
                {
                    // binary_op nan x = binary_op x nan = nan
                    temp.push_back("nan");
                    i += 2;
                    continue;
                }
                else if (expression[i] == "-")
                {
                    if ((isConst1 && isConst2) && (expression[i+1] == expression[i+2])) // - x x => 0
                    {
                        temp.push_back("0");
                        i += 2;
                        continue;
                    }
                    else if (checkEqual(expression[i+1], "0")) // - 0 x -> ~ x
                    {
                        temp.push_back("~");
                        temp.push_back(expression[i+2]);
                        i += 2;
                        continue;
                    }
                    else if (checkEqual(expression[i+2], "0") && isConst1) // - x 0 -> x
                    {
                        temp.push_back(expression[i+1]);
                        i += 2;
                        continue;
                    }
                }
                else if (expression[i] == "*")
                {
                    if (checkEqual(expression[i+1], "0") && isConst2) //* 0 x -> 0
                    {
                        temp.push_back("0");
                        i += 2;
                        continue;
                    }
                    else if (checkEqual(expression[i+2], "0") && isConst1) //* x 0 -> 0
                    {
                        temp.push_back("0");
                        i += 2;
                        continue;
                    }
                    else if (checkEqual(expression[i+1], "1") && isConst2) //* 1 x -> x
                    {
                        temp.push_back(expression[i+2]);
                        i += 2;
                        continue;
                    }
                    else if (checkEqual(expression[i+2], "1") && isConst1) //* x 1 -> x
                    {
                        temp.push_back(expression[i+1]);
                        i += 2;
                        continue;
                    }
                }
                else if (expression[i] == "+")
                {
                    if (checkEqual(expression[i+1], "0") && isConst2) // + 0 x -> x
                    {
                        temp.push_back(expression[i+2]);
                        i += 2;
                        continue;
                    }
                    else if (checkEqual(expression[i+2], "0") && isConst1) // + x 0 -> x
                    {
                        temp.push_back(expression[i+1]);
                        i += 2;
                        continue;
                    }
                }
                else if (expression[i] == "/")
                {
                    if (checkEqual(expression[i+1], "0") && isConst2) // / 0 x -> 0
                    {
                        temp.push_back("0");
                        i += 2;
                        continue;
                    }
                    else if (checkEqual(expression[i+2], "0") && isConst1) // / x 0 -> nan
                    {
                        temp.push_back("nan");
                        i += 2;
                        continue;
                    }
                    else if (checkEqual(expression[i+2], "1") && isConst1) // / x 1 -> x
                    {
                        temp.push_back(expression[i+1]);
                        i += 2;
                        continue;
                    }
                    else if (isConst1 && isConst2 && (expression[i+1] == expression[i+2])) // / x x -> 1
                    {
                        temp.push_back("1");
                        i += 2;
                        continue;
                    }
                }
                else if (expression[i] == "^")
                {
                    if (checkEqual(expression[i+2], "0") && isConst1) // ^ x 0 -> 1
                    {
                        temp.push_back("1");
                        i += 2;
                        continue;
                    }
                    else if (checkEqual(expression[i+1], "0") && isConst2) // ^ 0 x -> nan
                    {
                        temp.push_back("nan");
                        i += 2;
                        continue;
                    }
                    else if (checkEqual(expression[i+1], "1") && isConst2) // ^ 1 x -> 1
                    {
                        temp.push_back("1");
                        i += 2;
                        continue;
                    }
                    else if (checkEqual(expression[i+2], "1") && isConst1) // ^ x 1 -> x
                    {
                        temp.push_back(expression[i+1]);
                        i += 2;
                        continue;
                    }
                }

                // If none of the simplifications fired, just copy the operator.
                temp.push_back(expression[i]);
                continue;
            }

            // ==== UNARY OPERATORS WITH NUMERIC ARG ====
            else if (is_unary(expression[i]) && has1 && isdouble(expression[i+1]))
            {
                if (expression[i] == "cos")
                {
                    temp.push_back(simplifyString(
                        to_string_general(cos(Stod(expression[i+1])))
                    ));
                    i += 1;
                    continue;
                }
                else if (expression[i] == "~")
                {
                    if (expression[i+1] == "inf") //~ inf -> -inf
                    {
                        temp.push_back("-inf");
                    }
                    else if (checkEqual(expression[i+1], "0"))
                    {
                        temp.push_back("0");
                    }
                    else
                    {
                        temp.push_back(simplifyString(
                            to_string_general(-(Stod(expression[i+1])))
                        ));
                    }
                    i += 1;
                    continue;
                }
                else if (expression[i] == "sin")
                {
                    temp.push_back(simplifyString(
                        to_string_general(sin(Stod(expression[i+1])))
                    ));
                    i += 1;
                    continue;
                }
                else if ((expression[i] == "ln") || (expression[i] == "log"))
                {
                    temp.push_back(simplifyString(
                        to_string_general(log(Stod(expression[i+1])))
                    ));
                    i += 1;
                    continue;
                }
                else if (expression[i] == "asin" || expression[i] == "arcsin")
                {
                    temp.push_back(simplifyString(
                        to_string_general(asin(Stod(expression[i+1])))
                    ));
                    i += 1;
                    continue;
                }
                else if (expression[i] == "acos" || expression[i] == "arccos")
                {
                    temp.push_back(simplifyString(
                        to_string_general(acos(Stod(expression[i+1])))
                    ));
                    i += 1;
                    continue;
                }
                else if (expression[i] == "exp")
                {
                    temp.push_back(simplifyString(
                        to_string_general(exp(Stod(expression[i+1])))
                    ));
                    i += 1;
                    continue;
                }
                else if (expression[i] == "sech")
                {
                    temp.push_back(simplifyString(
                        to_string_general(1 / cosh(Stod(expression[i+1])))
                    ));
                    i += 1;
                    continue;
                }
                else if (expression[i] == "tanh")
                {
                    temp.push_back(simplifyString(
                        to_string_general(tanh(Stod(expression[i+1])))
                    ));
                    i += 1;
                    continue;
                }
                else if (expression[i] == "sqrt")
                {
                    temp.push_back(simplifyString(
                        to_string_general(sqrt(Stod(expression[i+1])))
                    ));
                    i += 1;
                    continue;
                }
                else if (expression[i] == "abs")
                {
                    temp.push_back(simplifyString(
                        to_string_general(abs(Stod(expression[i+1])))
                    ));
                    i += 1;
                    continue;
                }
                else
                {
                    temp.push_back(expression[i]);
                    continue;
                }
            }

            // ==== UNARY OPERATORS WITH SYMBOLIC ARG ====
            else if (is_unary(expression[i]))
            {
                // If we don't even have one operand, just copy and continue
                if (!has1)
                {
                    temp.push_back(expression[i]);
                    continue;
                }

                if (expression[i] == "~" && expression[i+1] == "~")
                {
                    // ~~x -> x (we just skip both ~)
                    i += 1;
                    continue;
                }
                else if (expression[i] == "exp" && (expression[i+1] == "ln" || expression[i+1] == "log"))
                {
                    // exp ln x -> x (we skip exp, ln; x will be processed later)
                    i += 1;
                    continue;
                }
                else if (expression[i+1] == "exp" && (expression[i] == "ln" || expression[i] == "log"))
                {
                    // ln exp x -> x (skip ln, exp)
                    i += 1;
                    continue;
                }
                else if (expression[i] == "cos" && (expression[i+1] == "acos" || expression[i+1] == "arccos"))
                {
                    // cos(acos x) -> x; skip cos, acos
                    i += 1;
                    continue;
                }
                else if ((expression[i] == "cos") && (expression[i+1] == "~")) // cos(-x) = cos(x)
                {
                    temp.push_back(expression[i]);
                    i += 1; // skip the ~
                    continue;
                }
                else if (expression[i+1] == "cos" && (expression[i] == "acos" || expression[i] == "arccos"))
                {
                    // acos(cos x) -> x; skip acos, cos
                    i += 1;
                    continue;
                }
                else if (expression[i] == "sin" && (expression[i+1] == "asin" || expression[i+1] == "arcsin"))
                {
                    // sin(asin x) -> x
                    i += 1;
                    continue;
                }
                else if (expression[i+1] == "sin" && (expression[i] == "asin" || expression[i] == "arcsin"))
                {
                    // asin(sin x) -> x
                    i += 1;
                    continue;
                }
                else if ((expression[i] == "sech") && (expression[i+1] == "~")) // sech(-x) = sech(x)
                {
                    temp.push_back(expression[i]);
                    i += 1; // skip the ~
                    continue;
                }
                else
                {
                    temp.push_back(expression[i]);
                    continue;
                }
            }

            // ==== NOT AN OPERATOR ====
            else
            {
                temp.push_back(expression[i]);
                continue;
            }
        }

        if (expression.size() != temp.size())
        {
            expression = temp;
        }
    }

    void simplifyPN(std::vector<std::string>& expression)
    {
        size_t size_before, size_after;
        do
        {
            size_before = expression.size();
            simplifyPN_Helper(expression);
            this->simplify_grasp.reserve(expression.size());
            graspSimplifyPrefix(expression, 0, expression.size() - 1, this->simplify_grasp);
            simplifyPN_Helper(expression);
            size_after = expression.size();
        } while (size_before != size_after);
    }

    //0 x x tanh sin cos + * -> 0 * (cos(sin(tanh(x)+x)
    //e.g. 0*x, 1*x, x*0, x*1, x-x, x+0, 0+x, 0-x, x-0, ...
    void graspSimplifyPostfixHelper(std::vector<std::string>& expression, int low, int up, std::vector<int>& grasp, std::vector<std::string>& new_expression, bool setGRvar = false)
    {
        if (!setGRvar)
        {
            grasp.clear();
            setPostfixGR(expression, grasp);
        }
    //    print_container(expression, low, up);
        if (expression[up] == "+" || expression[up] == "-") // x y +/-
        {
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPostfixHelper(expression, low, up-2-grasp[up-1], grasp, new_expression, true);
            int first_arg_idx_high = new_expression.size();
            graspSimplifyPostfixHelper(expression, up-1-grasp[up-1], up-1, grasp, new_expression, true);
            int second_arg_idx_high = new_expression.size();
            int step;
            
            if (checkEqual(new_expression.back(), "0")) // x 0 +/- -> x
            {
                //puts("hi 181");
                new_expression.pop_back();
            }
            
            else if (checkEqual(new_expression[first_arg_idx_high - 1], "0")) // 0 x +/- -> x
            {
                //puts("hi 184");
                //erase elements from new_expression[first_arg_idx_low] to new_expression[first_arg_idx_high-1] inclusive
                new_expression.erase(new_expression.begin() + first_arg_idx_low, new_expression.begin() + first_arg_idx_high); //0 y + -> y
                if (expression[up] == "-")
                {
                    //puts("hi 187");
                    new_expression.push_back("~"); //0 y - -> y ~
                }
            }
            
            else if ((new_expression[first_arg_idx_high - 1] == "nan") || (new_expression.back() == "nan")) //nan x +/- -> nan, x nan +/- -> nan, nan nan +/- -> nan
            {
                //puts("hi 269");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            
            else if ((expression[up] == "-") && ((step = (first_arg_idx_high - first_arg_idx_low)) == (second_arg_idx_high - first_arg_idx_high)) && (areExpressionRangesEqual(first_arg_idx_low, first_arg_idx_high, step, new_expression))) //x x - -> 0
            {
                //puts("hi 215");
                new_expression[first_arg_idx_low] = "0"; //change first symbol of x to 0
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.begin() + second_arg_idx_high); //erase the rest of x and y
            }
            
            else
            {
                new_expression.push_back(expression[up]);
            }
        }
        else if (expression[up] == "*") //x y *
        {
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPostfixHelper(expression, low, up-2-grasp[up-1], grasp, new_expression, true); //x
            int first_arg_idx_high = new_expression.size();
            graspSimplifyPostfixHelper(expression, up-1-grasp[up-1], up-1, grasp, new_expression, true); //y
            //int second_arg_idx_high = new_expression.size();
            //int step;
            
            if ((new_expression[first_arg_idx_high - 1] == "nan") || (new_expression.back() == "nan")) //nan x * -> nan, x nan * -> nan, nan nan * -> nan
            {
                //puts("hi 297");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (checkEqual(new_expression.back(), "0")) // x 0 * -> 0 (because, since postfix operators come at the end, if the end of the second argument of '*' is 0, then the whole second argument MUST be 0, therefore the expression reduces to x 0 *, which is 0)
            {
                //puts("hi 235");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (checkEqual(new_expression[first_arg_idx_high - 1], "0")) //0 x * -> 0 (because, since postfix operators come at the end, if the end of the first argument of '*' is 0, then the whole second argument MUST be 0, therefore the expression reduces to 0 x *, which is 0)
            {
                //puts("hi 241");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (checkEqual(new_expression.back(), "1")) // x 1 * -> x (because, since postfix operators come at the end, if the end of the second argument of '*' is 1, then the whole second argument MUST be 1, therefore the expression reduces to x 1 *, which is x)
            {
                //puts("hi 247");
                new_expression.pop_back(); //erase the '1'
            }
            else if (checkEqual(new_expression[first_arg_idx_high - 1], "1")) //1 x * -> x (because, since postfix operators come at the end, if the end of the first argument of '*' is 1, then the whole first argument MUST be 1, therefore the expression reduces to 1 x *, which is x)
            {
                //puts("hi 252");
                new_expression.erase(new_expression.begin() + first_arg_idx_high - 1); //erase the '1'
            }
            else if (checkEqual(new_expression.back(), "-1")) // x -1 * -> x ~ (because, since postfix operators come at the end, if the end of the second argument of '*' is -1, then the whole second argument MUST be -1, therefore the expression reduces to x -1 *, which is x ~)
            {
    //            puts("hi 368");
                new_expression.back() = "~"; //change the '-1' to a "~"
                
            }
            else if (checkEqual(new_expression[first_arg_idx_high - 1], "-1")) //-1 x * -> x ~ (because, since postfix operators come at the end, if the end of the first argument of '*' is -1, then the whole first argument MUST be -1, therefore the expression reduces to -1 x *, which is x ~)
            {
    //            puts("hi 374");
                new_expression.erase(new_expression.begin() + first_arg_idx_high - 1); //erase the '-1'
                new_expression.push_back("~"); //add a "~" at the end
            }
            else
            {
                new_expression.push_back(expression[up]);
            }
        }
        else if (expression[up] == "/") //x y /
        {
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPostfixHelper(expression, low, up-2-grasp[up-1], grasp, new_expression, true); //x
            int first_arg_idx_high = new_expression.size();
            graspSimplifyPostfixHelper(expression, up-1-grasp[up-1], up-1, grasp, new_expression, true); //y
            int second_arg_idx_high = new_expression.size();
            int step;
            
            if ((new_expression[first_arg_idx_high - 1] == "nan") || (new_expression.back() == "nan")) //nan x / -> nan, x nan / -> nan, nan nan / -> nan
            {
                //puts("hi 339");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if ((checkEqual(new_expression.back(), "0")) && (checkEqual(new_expression[first_arg_idx_high - 1], "0"))) // 0 0 / -> nan
            {
                //puts("hi 279");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (checkEqual(new_expression.back(), "0")) // x 0 / -> nan (for now, because, since postfix operators come at the end, if the end of the second argument of '/' is 0, then the whole second argument MUST be 0, therefore the expression reduces to x 0 /, which is, for now, assumed to be nan for simplicity)
            {
                //puts("hi 280");
                //TODO: need to come up with a more robust way that actually checks if this is nan anywhere;
                //for now we weed it out because annoying not to; giving this up seems like the better deal...
                new_expression[first_arg_idx_low] = "nan";//(new_expression[first_arg_idx_high - 1] == "~") ? "-inf" : "inf";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (checkEqual(new_expression[first_arg_idx_high - 1], "0")) //0 x / -> 0 (because, since postfix operators come at the end, if the end of the first argument of '/' is 0, then the whole second argument MUST be 0, therefore the expression reduces to 0 x /, which is 0)
            {
                //puts("hi 286");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (checkEqual(new_expression.back(), "1")) // x 1 / -> x (because, since postfix operators come at the end, if the end of the second argument of '/' is 1, then the whole second argument MUST be 1, therefore the expression reduces to x 1 /, which is x)
            {
                //puts("hi 292");
                new_expression.pop_back(); //erase the '1'
            }
            else if (checkEqual(new_expression.back(), "-1")) // x -1 / -> x ~ (because, since postfix operators come at the end, if the end of the second argument of '/' is -1, then the whole second argument MUST be -1, therefore the expression reduces to x -1 /, which is x ~)
            {
    //            puts("hi 425");
                new_expression.back() = "~"; //change the '-1' to a "~"
            }
            else if ((expression[up] == "/") && ((step = (first_arg_idx_high - first_arg_idx_low)) == (second_arg_idx_high - first_arg_idx_high)) && (areExpressionRangesEqual(first_arg_idx_low, first_arg_idx_high, step, new_expression))) // / x x -> 1
            {
                //puts("hi 297");
                new_expression[first_arg_idx_low] = "1"; //change first symbol of x to 1
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.begin() + second_arg_idx_high); //erase the rest of x and y
            }
            //TODO:
                /*
                x*y       y
                ---  -->  -
                x*z       z
                 
                  x              1
                ------ --> x * ------
                number         number
                 
                x num * num1 * -> x num num1 * *
                */
            else
            {
                new_expression.push_back(expression[up]);
            }
        }
        else if (expression[up] == "^") //x y ^
        {
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPostfixHelper(expression, low, up-2-grasp[up-1], grasp, new_expression, true); //x
            int first_arg_idx_high = new_expression.size();
            graspSimplifyPostfixHelper(expression, up-1-grasp[up-1], up-1, grasp, new_expression, true); //y
            //int second_arg_idx_high = new_expression.size();
            //int step;
            
            if ((new_expression[first_arg_idx_high - 1] == "nan") || (new_expression.back() == "nan")) //nan x ^ -> nan, x nan ^ -> nan, nan nan ^ -> nan
            {
                //puts("hi 397");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (checkEqual(new_expression.back(), "0")) // x 0 ^ -> 1 (because, since postfix operators come at the end, if the end of the second argument of '^' is 0, then the whole second argument MUST be 0, therefore the expression reduces to x 0 ^, which is 1)
            {
                //puts("hi 318");
                new_expression[first_arg_idx_low] = "1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (checkEqual(new_expression[first_arg_idx_high - 1], "0")) //0 x ^ -> nan (for now, because, since postfix operators come at the end, if the end of the first argument of '^' is 0, then the whole second argument MUST be 0, therefore the expression reduces to 0 x ^, which is, for now, assumed to be nan for simplicity)
            {
                //puts("hi 324");
                //TODO: need to come up with a more robust way that actually checks if this is nan anywhere;
                //for now we weed it out because annoying not to; giving this up seems like the better deal...
                new_expression[first_arg_idx_low] = "nan";// "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (checkEqual(new_expression.back(), "1")) // x 1 ^ -> x (because, since postfix operators come at the end, if the end of the second argument of '^' is 1, then the whole second argument MUST be 1, therefore the expression reduces to x 1 ^, which is x)
            {
                //puts("hi 330");
                new_expression.pop_back(); //erase the '1'
            }
            else if (checkEqual(new_expression[first_arg_idx_high - 1], "1")) //1 x ^ -> 1 (because, since postfix operators come at the end, if the end of the first argument of '^' is 1, then the whole second argument MUST be 1, therefore the expression reduces to 1 x ^, which is 1)
            {
                //puts("hi 335");
                new_expression[first_arg_idx_low] = "1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else
            {
                new_expression.push_back(expression[up]);
            }
        }
        else if (expression[up] == "cos") //x cos
        {
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
            
            if (new_expression.back() == "nan") // nan cos -> nan (because, since postfix operators come at the end, if the end of the argument of 'cos' is nan, then the whole argument MUST be nan, therefore the expression reduces to nan cos, which is nan)
            {
                //puts("hi 439");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "0")) // 0 cos -> 1 (because, since postfix operators come at the end, if the end of the argument of 'cos' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 cos, which is 1)
            {
                //puts("hi 350");
                new_expression[first_arg_idx_low] = "1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "inf") || (new_expression.back() == "-inf")) // +/- inf cos -> nan (because, since postfix operators come at the end, if the end of the argument of 'cos' is +/- inf, then the whole argument MUST be +/- inf, therefore the expression reduces to +/- inf cos, which is nan)
            {
    //            puts("hi 499");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ cos -> nan (because, since postfix operators come at the end, if the end of the argument of 'cos' is inf ~, then the whole argument MUST be inf ~, therefore the expression reduces to inf ~ cos, which is nan)
            {
    //            puts("hi 505");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else
            {
                new_expression.push_back(expression[up]);
            }
        }
        else if (expression[up] == "sin") //x sin
        {
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
            
            if (new_expression.back() == "nan") // nan sin -> nan (because, since postfix operators come at the end, if the end of the argument of 'sin' is nan, then the whole argument MUST be nan, therefore the expression reduces to nan sin, which is nan)
            {
                //puts("hi 462");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "0")) // 0 sin -> 0 (because, since postfix operators come at the end, if the end of the argument of 'sin' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 sin, which is 0)
            {
                //puts("hi 365");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "inf") || (new_expression.back() == "-inf")) // +/- inf sin -> nan (because, since postfix operators come at the end, if the end of the argument of 'sin' is +/- inf, then the whole argument MUST be +/- inf, therefore the expression reduces to +/- inf sin, which is nan)
            {
    //            puts("hi 533");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ sin -> nan (because, since postfix operators come at the end, if the end of the argument of 'sin' is inf ~, then the whole argument MUST be inf ~, therefore the expression reduces to inf ~ sin, which is nan)
            {
    //            puts("hi 539");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else
            {
                new_expression.push_back(expression[up]);
            }
        }
        else if (expression[up] == "tanh") //x tanh
        {
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
            
            if (new_expression.back() == "nan") // nan tanh -> nan (because, since postfix operators come at the end, if the end of the argument of 'tanh' is nan, then the whole argument MUST be nan, therefore the expression reduces to nan tanh, which is nan)
            {
                //puts("hi 485");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "0")) // 0 tanh -> 0 (because, since postfix operators come at the end, if the end of the argument of 'tanh' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 tanh, which is 0)
            {
                //puts("hi 380");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "inf") // inf tanh -> 1 (because, since postfix operators come at the end, if the end of the argument of 'tanh' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf tanh, which is 1)
            {
                //puts("hi 386");
                new_expression[first_arg_idx_low] = "1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "-inf") // -inf tanh -> -1 (because, since postfix operators come at the end, if the end of the argument of 'tanh' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf tanh, which is -1)
            {
                //puts("hi 392");
                new_expression[first_arg_idx_low] = "-1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ tanh -> -1 (because, since postfix operators come at the end, if the end of the argument of 'tanh' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf tanh, which is -1)
            {
                //puts("hi 392");
                new_expression[first_arg_idx_low] = "-1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else
            {
                new_expression.push_back(expression[up]);
            }
        }
        else if (expression[up] == "sech") //x sech
        {
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
            
            if (new_expression.back() == "nan") // nan sech -> nan (because, since postfix operators come at the end, if the end of the argument of 'sech' is nan, then the whole argument MUST be nan, therefore the expression reduces to nan sech, which is nan)
            {
                //puts("hi 525");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "0")) // 0 sech -> 1 (because, since postfix operators come at the end, if the end of the argument of 'sech' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 sech, which is 1)
            {
                //puts("hi 395");
                new_expression[first_arg_idx_low] = "1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "inf") // inf sech -> 0 (because, since postfix operators come at the end, if the end of the argument of 'sech' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf sech, which is 0)
            {
    //            puts("hi 419");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "-inf") // -inf sech -> 0 (because, since postfix operators come at the end, if the end of the argument of 'sech' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf sech, which is 0)
            {
                //puts("hi 425");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ sech -> 0 (because, since postfix operators come at the end, if the end of the argument of 'sech' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf sech, which is 0)
            {
                //puts("hi 431");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else
            {
                new_expression.push_back(expression[up]);
            }
        }
        else if (expression[up] == "~") //x ~
        {
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
            
            if (new_expression.back() == "nan") // nan ~ -> nan (because, since postfix operators come at the end, if the end of the argument of '~' is nan, then the whole argument MUST be nan, therefore the expression reduces to nan ~, which is nan)
            {
                //puts("hi 565");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "0")) // 0 ~ -> 0 (because, since postfix operators come at the end, if the end of the argument of '~' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 ~, which is 0)
            {
    //            puts("hi 445");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "1")) // 1 ~ -> -1 (because, since postfix operators come at the end, if the end of the argument of '~' is 1, then the whole argument MUST be 1, therefore the expression reduces to 1 ~, which is -1)
            {
    //            puts("hi 663");
                new_expression[first_arg_idx_low] = "-1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "-1")) // -1 ~ -> 1 (because, since postfix operators come at the end, if the end of the argument of '~' is -1, then the whole argument MUST be -1, therefore the expression reduces to -1 ~, which is 1)
            {
    //            puts("hi 669");
                new_expression[first_arg_idx_low] = "1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "inf") // inf ~ -> -inf (because, since postfix operators come at the end, if the end of the argument of '~' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf ~, which is -inf)
            {
                //puts("hi 507");
                new_expression[first_arg_idx_low] = "-inf";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "-inf") // -inf ~ -> inf (because, since postfix operators come at the end, if the end of the argument of '~' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf ~, which is inf)
            {
    //            puts("hi 669");
                new_expression[first_arg_idx_low] = "inf";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ ~ -> 0 (because, since postfix operators come at the end, if the end of the argument of '~' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf ~, which is inf)
            {
    //            puts("hi 675");
                new_expression[first_arg_idx_low] = "inf";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            
            else
            {
                new_expression.push_back(expression[up]);
            }
        }
        else if (expression[up] == "exp") //x exp
        {
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
            
            if (new_expression.back() == "nan") // nan exp -> nan (because, since postfix operators come at the end, if the end of the argument of 'exp' is nan, then the whole argument MUST be nan, therefore the expression reduces to nan exp, which is nan)
            {
                //puts("hi 593");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "0")) // 0 exp -> 1 (because, since postfix operators come at the end, if the end of the argument of 'exp' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 exp, which is 1)
            {
                //puts("hi 524");
                new_expression[first_arg_idx_low] = "1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "inf") // inf exp -> inf (because, since postfix operators come at the end, if the end of the argument of 'exp' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf exp, which is inf)
            {
                //puts("hi 530");
                new_expression[first_arg_idx_low] = "inf";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "-inf") // -inf exp -> 0 (because, since postfix operators come at the end, if the end of the argument of 'exp' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf exp, which is 0)
            {
    //            puts("hi 697");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ exp -> 0 (because, since postfix operators come at the end, if the end of the argument of 'exp' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf exp, which is 0)
            {
    //            puts("hi 703");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else
            {
                new_expression.push_back(expression[up]);
            }
        }
        else if ((expression[up] == "ln") || (expression[up] == "log")) //x ln
        {
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
            
            if (new_expression.back() == "nan") // nan ln -> nan (because, since postfix operators come at the end, if the end of the argument of 'ln' is nan, then the whole argument MUST be nan, therefore the expression reduces to nan ln, which is nan)
            {
    //            puts("hi 707");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "0")) // 0 ln -> -inf (because, since postfix operators come at the end, if the end of the argument of 'ln' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 ln, which is -inf)
            {
    //            puts("hi 713");
                new_expression[first_arg_idx_low] = "-inf";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "inf") // inf ln -> inf (because, since postfix operators come at the end, if the end of the argument of 'ln' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf ln, which is inf)
            {
    //            puts("hi 719");
                new_expression[first_arg_idx_low] = "inf";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "-inf") // -inf ln -> nan (because, since postfix operators come at the end, if the end of the argument of 'ln' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf ln, which is nan)
            {
    //            puts("hi 750");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ ln -> nan (because, since postfix operators come at the end, if the end of the argument of 'ln' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf ln, which is nan)
            {
    //            puts("hi 756");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else
            {
                new_expression.push_back(expression[up]);
            }
        }
        else if ((expression[up] == "asin") || (expression[up] == "arcsin")) //x asin
        {
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
            
            if (new_expression.back() == "nan") // nan asin -> nan (because, since postfix operators come at the end, if the end of the argument of 'asin' is nan, then the whole argument MUST be nan, therefore the expression reduces to nan asin, which is nan)
            {
    //            puts("hi 735");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "0")) // 0 asin -> 0 (because, since postfix operators come at the end, if the end of the argument of 'asin' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 asin, which is 0)
            {
    //            puts("hi 741");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "inf") // inf asin -> nan (because, since postfix operators come at the end, if the end of the argument of 'asin' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf asin, which is nan)
            {
    //            puts("hi 747");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "-inf") // -inf asin -> nan (because, since postfix operators come at the end, if the end of the argument of 'asin' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf asin, which is nan)
            {
    //            puts("hi 790");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ asin -> nan (because, since postfix operators come at the end, if the end of the argument of 'asin' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf asin, which is nan)
            {
    //            puts("hi 796");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else
            {
                new_expression.push_back(expression[up]);
            }
        }
        else if ((expression[up] == "acos") || (expression[up] == "arccos")) //x acos
        {
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
            
            if (new_expression.back() == "nan") // nan acos -> nan (because, since postfix operators come at the end, if the end of the argument of 'acos' is nan, then the whole argument MUST be nan, therefore the expression reduces to nan acos, which is nan)
            {
    //            puts("hi 763");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "1")) // 1 acos -> 0 (because, since postfix operators come at the end, if the end of the argument of 'acos' is 1, then the whole argument MUST be 1, therefore the expression reduces to 1 acos, which is 0)
            {
    //            puts("hi 769");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "inf") // inf acos -> nan (because, since postfix operators come at the end, if the end of the argument of 'acos' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf acos, which is nan)
            {
    //            puts("hi 775");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "-inf") // -inf acos -> nan (because, since postfix operators come at the end, if the end of the argument of 'acos' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf acos, which is nan)
            {
    //            puts("hi 830");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ acos -> nan (because, since postfix operators come at the end, if the end of the argument of 'acos' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf acos, which is nan)
            {
    //            puts("hi 836");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else
            {
                new_expression.push_back(expression[up]);
            }
        }
        else if (expression[up] == "sqrt") //x sqrt
        {
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
            
            if (new_expression.back() == "nan") // nan sqrt -> nan (because, since postfix operators come at the end, if the end of the argument of 'sqrt' is nan, then the whole argument MUST be nan, therefore the expression reduces to nan sqrt, which is nan)
            {
    //            puts("hi 791");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "0")) // 0 sqrt -> 0 (because, since postfix operators come at the end, if the end of the argument of 'sqrt' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 sqrt, which is 0)
            {
    //            puts("hi 797");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "1")) // 1 sqrt -> 1 (because, since postfix operators come at the end, if the end of the argument of 'sqrt' is 1, then the whole argument MUST be 1, therefore the expression reduces to 1 sqrt, which is 1)
            {
    //            puts("hi 803");
                new_expression[first_arg_idx_low] = "1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "-1")) // -1 sqrt -> nan (because, since postfix operators come at the end, if the end of the argument of 'sqrt' is -1, then the whole argument MUST be -1, therefore the expression reduces to -1 sqrt, which is nan)
            {
    //            puts("hi 870");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "inf") // inf sqrt -> inf (because, since postfix operators come at the end, if the end of the argument of 'sqrt' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf sqrt, which is inf)
            {
    //            puts("hi 809");
                new_expression[first_arg_idx_low] = "inf";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "-inf") // -inf sqrt -> nan (because, since postfix operators come at the end, if the end of the argument of 'sqrt' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf sqrt, which is nan)
            {
    //            puts("hi 876");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ sqrt -> nan (because, since postfix operators come at the end, if the end of the argument of 'sqrt' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf sqrt, which is nan)
            {
    //            puts("hi 882");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else
            {
                new_expression.push_back(expression[up]);
            }
        }
        else if (expression[up] == "abs") //x abs
        {
            graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
            new_expression.push_back(expression[up]); //abs
        }
        else
        {
            for (int i = low; i <= up; i++)
            {
                //assert(i < expression.size() && i >= 0);
                new_expression.push_back(expression[i]);
            }
        }
    }

    void graspSimplifyPostfix(std::vector<std::string>& expression, int low, int up, std::vector<int>& grasp)
    {
        std::vector<std::string> new_expression;
        new_expression.reserve(expression.size());
        graspSimplifyPostfixHelper(expression, low, up, grasp, new_expression, false);
        expression = new_expression;
    }

    // 0 0 0 1 + - * tanh sech 1 /
    // 0 0 1 - * tanh sech 1 /
    // 0 -1 * tanh sech 1 /
    // 0 tanh sech 1 /
    // 0 sech 1 /
    // 1 1 /
    // 1
    void simplifyRPN_Helper(std::vector<std::string>& expression)
    {
        if (expression.size() < 2) return; //just an operand
        
        bool isdouble1, isdouble2, isConst1, isConst2;
        thread_local std::vector<std::string> temp;
        temp.clear();
        temp.reserve(expression.size());
    //    printf("expression before = ");for (const auto& i: expression){std::cout << i << ' ';}puts("");
        for (int i = static_cast<int>(expression.size()) - 1; i >= 0; i--)
        {
    //        std::cout << "i = " << i << '\n';// << expression[i] << '\n';
            if (is_binary(expression[i]))
            {
                isdouble1 = isdouble(expression[i-1]);
                isdouble2 = isdouble(expression[i-2]);
                
                if (isdouble1 && isdouble2)
                {
                    if (expression[i] == "+")
                    {
                        temp.push_back(simplifyString(to_string_general(Stod(expression[i-2]) + Stod(expression[i-1]))));
                        i -= 2;
                        continue;
                    }
                    else if (expression[i] == "-")
                    {
                        temp.push_back(simplifyString(to_string_general(Stod(expression[i-2]) - Stod(expression[i-1]))));
                        i -= 2;
                        continue;
                    }
                    else if (expression[i] == "*")
                    {
                        temp.push_back(simplifyString(to_string_general(Stod(expression[i-2]) * Stod(expression[i-1]))));
                        i -= 2;
                        continue;
                    }
                    else if (expression[i] == "/")
                    {
                        temp.push_back(simplifyString(to_string_general(Stod(expression[i-2]) / Stod(expression[i-1]))));
                        i -= 2;
                        continue;
                    }
                    else if (expression[i] == "^")
                    {
                        temp.push_back(simplifyString(to_string_general(std::pow(Stod(expression[i-2]), Stod(expression[i-1])))));
    //                            printf("hi 566, res = %s\n", expression[i].c_str());
                        i -= 2;
                        continue;
                    }
                }
                
                isConst1 = is_const(expression[i-1]);
                isConst2 = is_const(expression[i-2]);
                
                if ((isConst1 && isConst2) && ((expression[i-1].find("nan") != std::string::npos) || (expression[i-2].find("nan") != std::string::npos))) //x nan binary_op = nan x binary_op = nan
                {
    //                        puts("hi 549");
                    temp.push_back("nan");
                    i -= 2;
                    continue;
                }
                else if (expression[i] == "-")
                {
                    if ((isConst1 && isConst2) && (expression[i-1] == expression[i-2])) //x x - => 0
                    {
                        temp.push_back("0");
                        i -= 2;
                        continue;
                    }
                    else if (checkEqual(expression[i-2], "0") && isConst1) //"0 x -" -> "x ~"
                    {
                        temp.push_back("~");
                        temp.push_back(expression[i-1]);
                        

                        i -= 2;
                        continue;
                    }
                    else if (checkEqual(expression[i-1], "0")) //"x 0 -" -> "x"
                    {
                        temp.push_back(expression[i-2]);
                        i -= 2;
                        continue;
                    }
                }

                else if (expression[i] == "*")
                {
                    if (checkEqual(expression[i-2], "0") && isConst1) //"0 x *" -> "0"
                    {
                        //puts("hi 131");
                        temp.push_back("0");
                        i -= 2;
                        continue;
                    }
                    else if (checkEqual(expression[i-1], "0") && isConst2) //"x 0 *" -> "0"
                    {
                        //puts("hi 139");
                        temp.push_back("0");
                        i -= 2;
                        continue;
                    }
                    else if (checkEqual(expression[i-2], "1") && isConst1) //"1 x *" -> "x"
                    {
                        //puts("hi 147");
                        temp.push_back(expression[i-1]);
                        i -= 2;
                        continue;
                    }
                    else if (checkEqual(expression[i-1], "1") && isConst2) //"x 1 *" -> "x"
                    {
                        //puts("hi 155");
                        temp.push_back(expression[i-2]);
                        i -= 2;
                        continue;
                    }
                }

                else if (expression[i] == "+")
                {
                    if (checkEqual(expression[i-2], "0") && isConst1) //"0 x +" -> "x"
                    {
                        //puts("hi 167");
                        temp.push_back(expression[i-1]);
                        i -= 2;
                        continue;
                    }
                    else if (checkEqual(expression[i-1], "0") && isConst2) //"x 0 +" -> "x"
                    {
                        //puts("hi 175");
                        temp.push_back(expression[i-2]);
                        i -= 2;
                        continue;
                    }
                }

                else if (expression[i] == "/")
                {
                    if (checkEqual(expression[i-2], "0") && isConst1) // "0 x /" -> "0"
                    {
                        //puts("hi 187");
                        temp.push_back("0");
                        i -= 2;
                        continue;
                    }
                    else if (checkEqual(expression[i-1], "1") && isConst2) // "x 1 /" -> "x"
                    {
                        //puts("hi 195");
                        temp.push_back(expression[i-2]);
                        i -= 2;
                        continue;
                    }
                    else if (checkEqual(expression[i-1], "0") && isConst2) // "x 0 /" -> "nan"
                    {
                        //puts("hi 859");
                        temp.push_back("nan");
                        i -= 2;
                        continue;
                    }
                    else if (isConst1 && isConst2 && (expression[i-1] == expression[i-2])) // "x x /" -> "1"
                    {
                        //puts("hi 203");
                        temp.push_back("1");
                        i -= 2;
                        continue;
                    }
                }

                else if (expression[i] == "^")
                {
                    if (checkEqual(expression[i-1], "0") && isConst2) // "x 0 ^" -> "1"
                    {
                        //puts("hi 223");
                        temp.push_back("1");
                        i -= 2;
                        continue;
                    }
                    else if (checkEqual(expression[i-2], "0") && isConst1) // "0 x ^" -> "nan"
                    {
                        //puts("hi 880");
                        temp.push_back("nan");
                        i -= 2;
                        continue;
                    }
                    else if (checkEqual(expression[i-2], "1") && isConst1) // "1 x ^" -> "1"
                    {
                        //puts("hi 231");
                        temp.push_back("1");
                        i -= 2;
                        continue;
                    }
                    else if (checkEqual(expression[i-1], "1") && isConst2) // "x 1 ^" -> "x"
                    {
                        //puts("hi 239");
                        temp.push_back(expression[i-2]);
                        i -= 2;
                        continue;
                    }
                }
                temp.push_back(expression[i]);
                continue;
            }
            
            else if (is_unary(expression[i]) && isdouble(expression[i-1]))
            {
                if (expression[i] == "cos")
                {
                    temp.push_back(simplifyString(to_string_general(cos(Stod(expression[i-1])))));
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "~")
                {
                    if (checkEqual(expression[i-1], "0")) //0 ~ -> 0
                    {
                        //puts("hi 917");
                        temp.push_back("0");
                    }
                    else if (expression[i-1] == "inf") //inf ~ -> -inf
                    {
                        //puts("hi 924");
                        temp.push_back("-inf"); // change 'inf' to '-inf'
                    }
                    else
                    {
                        temp.push_back(simplifyString(to_string_general(-(Stod(expression[i-1])))));
                    }
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "sin")
                {
                    temp.push_back(simplifyString(to_string_general(sin(Stod(expression[i-1])))));
                    i -= 1;
                    continue;
                }
                else if ((expression[i] == "ln") || (expression[i] == "log"))
                {
                    temp.push_back(simplifyString(to_string_general(log(Stod(expression[i-1]))))); // Natural log (ln)
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "asin" || expression[i] == "arcsin")
                {
                    temp.push_back(simplifyString(to_string_general(asin(Stod(expression[i-1])))));
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "acos" || expression[i] == "arccos")
                {
                    temp.push_back(simplifyString(to_string_general(acos(Stod(expression[i-1])))));
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "exp")
                {
                    temp.push_back(simplifyString(to_string_general(exp(Stod(expression[i-1])))));
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "sech")
                {
                    temp.push_back(simplifyString(to_string_general(1.0 / cosh(Stod(expression[i-1]))))); // sech(x) = 1 / cosh(x)
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "tanh")
                {
                    temp.push_back(simplifyString(to_string_general(tanh(Stod(expression[i-1])))));
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "sqrt")
                {
                    temp.push_back(simplifyString(to_string_general(sqrt(Stod(expression[i-1])))));
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "abs")
                {
                    temp.push_back(simplifyString(to_string_general(abs(Stod(expression[i-1])))));
                    i -= 1;
                    continue;
                }
            }
            
            else if (is_unary(expression[i]))
            {
                if (expression[i] == "~" && expression[i-1] == "~")
                {
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "exp" && (expression[i-1] == "ln" || expression[i-1] == "log"))
                {
                    //puts("hi 360");
                    i -= 1;
                    continue;
                }
                else if (expression[i-1] == "exp" && (expression[i] == "ln" || expression[i] == "log"))
                {
                    //puts("hi 368");
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "cos" && (expression[i-1] == "acos" || expression[i-1] == "arccos"))
                {
                    //puts("hi 408");
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "cos" && expression[i-1] == "~") //cos(-x) -> cos(x)
                {
                    //puts("hi 688");
                    temp.push_back(expression[i]);
                    i -= 1;
                    continue;
                }
                else if (expression[i-1] == "cos" && (expression[i] == "acos" || expression[i] == "arccos"))
                {
                    //puts("hi 416");
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "sin" && (expression[i-1] == "asin" || expression[i-1] == "arcsin"))
                {
                    //puts("hi 424");
                    i -= 1;
                    continue;
                }
                else if (expression[i-1] == "sin" && (expression[i] == "asin" || expression[i] == "arcsin"))
                {
                    //puts("hi 432");
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "sech" && expression[i-1] == "~") //'x ~ sech' -> 'x sech'
                {
                    temp.push_back(expression[i]);
                    i -= 1;
                    continue;
                }
                else
                {
                    temp.push_back(expression[i]);
                    continue;
                }
            }
            else
            {
                temp.push_back(expression[i]);
                continue;
            }
        }
        
        if (expression.size() != temp.size()) //if expression was simplified, assign it to original expression vector
        {
            expression.resize(temp.size());
            for (size_t i = 0; i < expression.size(); i++)
            {
                expression[i] = temp[expression.size() - i - 1];
            }
        }
    //    printf("expression after = ");for (const auto& i: expression){std::cout << i << ' ';}puts("");

    }

    
    void simplifyRPN(std::vector<std::string>& expression)
    {
        size_t size_before, size_after;
        do
        {
            size_before = expression.size();
            simplifyRPN_Helper(expression);
            this->simplify_grasp.reserve(expression.size());
            graspSimplifyPostfix(expression, 0, expression.size() - 1, this->simplify_grasp);
            simplifyRPN_Helper(expression);
            size_after = expression.size();
        } while (size_before != size_after);
    }

    /*
     Returns a pair containing the depth of the sub-expression from start to stop, and whether or not it's complete
     Algorithm adopted from here: https://stackoverflow.com/a/77180279
     */
    std::pair<int, bool> getPNdepth(const std::vector<std::string>& expression, int idx, size_t start = 0, size_t stop = 0, bool cache = false, bool modify = false, bool binary = false, bool unary = false, bool leaf = false)
    {
        if (expression.empty())
        {
            return std::make_pair(0, false);
        }

        if (stop == 0)
        {
            stop = expression.size();
        }

        if (!cache)
        {
            this->stack[idx].clear();
            this->depth[idx] = 0, this->num_binary[idx] = 0, this->num_leaves[idx] = 0;
            for (size_t i = start; i < stop; i++)
            {
                if (is_binary(expression[i]))
                {
                    this->stack[idx].push_back(2);  // Number of operands
                    this->num_binary[idx]++;
                }
                else if (is_unary(expression[i]))
                {
                    this->stack[idx].push_back(1);
                }
                else
                {
                    this->num_leaves[idx]++;
                    while (!this->stack[idx].empty() && this->stack[idx].back() == 1) //so the this->stack will shrink one by one from the back until it's empty and/or the last element is NOT 1
                    {
                        this->stack[idx].pop_back();  // Remove fulfilled operators
                    }
                    if (!this->stack[idx].empty())
                    {
                        this->stack[idx].back()--;  // Indicate an operand is consumed
                    }
                }
                this->depth[idx] = std::max(this->depth[idx], static_cast<int>(this->stack[idx].size()) + 1);
            }
        }
        else //optimize with caching
        {
            if (not modify) //get_legal_moves()
            {
                if (binary) //Gives the this->depth and completeness of the current PN expression + a binary operator
                {
                    return std::make_pair(std::max(this->depth[idx], static_cast<int>(this->stack[idx].size()) + 2) - 1, this->num_leaves[idx] == this->num_binary[idx] + 2);
                }
                else if (unary) //Gives the this->depth and completeness of the current PN expression + a unary operator
                {
                    return std::make_pair(std::max(this->depth[idx], static_cast<int>(this->stack[idx].size()) + 2) - 1, this->num_leaves[idx] == this->num_binary[idx] + 1);
                }
                else if (leaf) //Gives the this->depth and completeness of the current PN expression + a leaf node
                {
                    auto last_filled_op_it = std::find_if(this->stack[idx].rbegin(), this->stack[idx].rend(), [](int i){return i != 1;}); //Find the first element from the back that's not 1
                    return std::make_pair(std::max(this->depth[idx], static_cast<int>(this->stack[idx].rend() - last_filled_op_it) /* this->stack.size() */ + 1) - 1, this->num_leaves[idx] == this->num_binary[idx]);
                }
            }
            else //modify -> complete_status()
            {
                if (is_binary(expression[this->idx[idx]]))
                {
                    this->stack[idx].push_back(2);  // Number of operands
                    this->num_binary[idx]++;
                }
                else if (is_unary(expression[this->idx[idx]]))
                {
                    this->stack[idx].push_back(1);
                }
                else
                {
                    this->num_leaves[idx]++;
                    while (!this->stack[idx].empty() && this->stack[idx].back() == 1) //so the this->stack will shrink one-by-one from the back until it's empty and/or the last element is NOT 1
                    {
                        this->stack[idx].pop_back();  // Remove fulfilled operators
                    }
                    if (!this->stack[idx].empty())
                    {
                        this->stack[idx].back()--;  // Indicate an operand is consumed
                    }
                }
                this->depth[idx] = std::max(this->depth[idx], static_cast<int>(this->stack[idx].size()) + 1);
                this->idx[idx]++;
            }
        }
        return std::make_pair(this->depth[idx] - 1, this->num_leaves[idx] == this->num_binary[idx] + 1);
    }

    /*
     Returns a pair containing the depth of the sub-expression from start to stop, and whether or not it's complete
     Algorithm adopted from here: https://stackoverflow.com/a/77128902
     */
    std::pair<int, bool> getRPNdepth(const std::vector<std::string>& expression, int idx, size_t start = 0, size_t stop = 0, bool cache = false, bool modify = false, bool unary = false, bool leaf = false)
    {
        if (expression.empty())
        {
            return std::make_pair(0, false);
        }

        if (stop == 0)
        {
            stop = expression.size();
        }
        assert(this->stack.size() > static_cast<decltype(this->stack.size())>(idx));

        if (!cache)
        {
            this->stack[idx].clear();
            bool complete = true;
//            std::cout << "expression = " << expression << '\n';
            for (size_t i = start; i < stop; i++)
            {
//                std::cout << "this->stack[idx] = "
//                << this->stack[idx] << '\n';
                if (is_unary(expression[i]))
                {
//                    std::cout << "expression["
//                    << i << "] is a unary operator\n";
                    this->stack[idx].back() += 1;
                }
                else if (is_binary(expression[i]))
                {
//                    std::cout << "expression["
//                    << i << "] is a binary operator\n";
                    int op2 = this->stack[idx].back();
                    this->stack[idx].pop_back();
                    int op1 = this->stack[idx].back();
                    this->stack[idx].pop_back();
                    this->stack[idx].push_back(std::max(op1, op2) + 1);
                }
                else //leaf
                {
//                    std::cout << "expression["
//                    << i << "] is a leaf node\n";
                    this->stack[idx].push_back(1);
                }
            }

            while (this->stack[idx].size() > 1)
            {
                int op2 = this->stack[idx].back();
                this->stack[idx].pop_back();
                int op1 = this->stack[idx].back();
                this->stack[idx].pop_back();
                this->stack[idx].push_back(std::max(op1, op2) + 1);
                complete = false;
            }

            /*
             e.g., assume this->stack = {1, 2, 3, 4, 5}, then:
             {1, 2, 3, 4, 5}
             {1, 2, 3, 6}
             {1, 2, 7}
             {1, 8}
             {9}
             */

            return std::make_pair(this->stack[idx].back() - 1, complete);
        }
        else //optimize with caching
        {
            if (not modify)  //get_legal_moves()
            {
                if (unary) //Gives the this->depth and completeness of the current RPN expression + a unary operator
                {
                    if (this->stack[idx].size() == 1)
                    {
                        return std::make_pair(this->stack[idx].back(), true);
                    }
                    else
                    {
                        assert(this->stack[idx].size() >= 2);
                        int curr_max = std::max(this->stack[idx].back()+1, *(this->stack[idx].end()-2))+1;
                        for (int i = this->stack[idx].size() - 2; i >= 1; i--)
                        {
                            curr_max = std::max(curr_max, this->stack[idx][i-1])+1;
                        }

                        /*
                         e.g., assume this->stack = {1, 2, 3, 4, 5}, then:
                         curr_max = max(5, 4)+1 = 6;
                         curr_max = max(6, 3)+1 = 7;
                         curr_max = max(7, 2)+1 = 8;
                         curr_max = max(8, 1)+1 = 9;
                         */

                        return std::make_pair(curr_max - 1, false);
                    }
                }
                else if (leaf) //Gives the this->depth and completeness of the current RPN expression + a leaf node
                {
                    if (this->stack[idx].empty())
                    {
                        return std::make_pair(0, true);
                    }
                    else
                    {
                        assert(this->stack[idx].size() >= 1);
                        int curr_max = std::max(this->stack[idx].back(), 1)+1;
                        for (int i = this->stack[idx].size() - 1; i >= 1; i--)
                        {
                            curr_max = std::max(curr_max, this->stack[idx][i-1])+1;
                        }
                        /*
                         e.g., assume this->stack = {1, 2, 3, 4, 5}, then:
                         curr_max = max(5, 4)+1 = 6;
                         curr_max = max(6, 3)+1 = 7;
                         curr_max = max(7, 2)+1 = 8;
                         curr_max = max(8, 1)+1 = 9;
                         */

                        return std::make_pair(curr_max - 1, false);
                    }
                }
            }
            else //modify -> complete_status()
            {
                if (is_binary(expression[this->idx[idx]]))
                {
                    int op2 = this->stack[idx].back();
                    this->stack[idx].pop_back();
                    int op1 = this->stack[idx].back();
                    this->stack[idx].pop_back();
                    this->stack[idx].push_back(std::max(op1, op2) + 1);
                }
                else if (is_unary(expression[this->idx[idx]]))
                {
                    this->stack[idx].back() += 1;
                }
                else //leaf
                {
                    assert(this->stack.size() > static_cast<decltype(this->stack.size())>(idx));
//                    std::cout << "this->stack.size() = " << this->stack.size() << '\n';
                    this->stack[idx].push_back(1);
                }

                this->idx[idx]++;
                if (this->stack[idx].size() == 1)
                {
                    return std::make_pair(this->stack[idx].back() - 1, true);
                }

                else
                {
                    int curr_max = std::max(this->stack[idx].back(), *(this->stack[idx].end()-2))+1;
                    for (int i = this->stack[idx].size() - 2; i >= 1; i--)
                    {
                        curr_max = std::max(curr_max, this->stack[idx][i-1])+1;
                    }
                    return std::make_pair(curr_max - 1, false);
                }
            }

            return std::make_pair(this->stack[idx].back() - 1, true);
        }
    }
    
    std::vector<std::string> complete_tree(const std::vector<std::string>& expression_val, int idx)
    {
        int expr_depth;
        bool extended = true;
        std::vector<std::string> expression = expression_val;
        if (this->expression_type == "prefix")
        {
            expr_depth  = this->getPNdepth(expression, idx).first;
            std::vector<std::string> expr_to_insert = {"+", "0"};
            std::vector<std::string> temp_expression;
            while (extended)
            {
                extended = false;
                for (size_t i = 0; i < expression.size(); i++)
                {
                    if (is_const(expression[i]))
                    {
                        temp_expression = expression;
                        temp_expression.insert(temp_expression.begin() + i, expr_to_insert.begin(), expr_to_insert.end());
                        if (this->getPNdepth(temp_expression, idx).first == expr_depth)
                        {
                            expression = temp_expression;
                            extended = true;
                            break;
                        }
                    }
                }
            }
        }
        else //postfix
        {
            expr_depth  = this->getRPNdepth(expression, idx).first;
            while (extended)
            {
                extended = false;
                std::vector<std::string> temp_expression;
                for (size_t i = 0; i < expression.size(); i++)
                {
                    if (is_const(expression[i]))
                    {
                        temp_expression = expression;
                        temp_expression.insert(temp_expression.begin() + i, "0");
                        temp_expression.insert(temp_expression.begin() + i + 2, "+");
                        if (this->getRPNdepth(temp_expression, idx).first == expr_depth)
                        {
                            expression = temp_expression;
                            extended = true;
                            break;
                        }
                    }
                }
            }
        }
        return expression;
    }
    
    std::vector<std::vector<std::string>> complete_tree(const std::vector<std::vector<std::string>>& expression_val)
    {
        thread_local std::vector<std::vector<std::string>> temp_expr(expression_val.size());
        assert(temp_expr.size() == expression_val.size());
        
        for (auto &expression: temp_expr)
        {
            expression.clear();
            expression.reserve(100);
        }
        
        for (size_t i = 0; i < expression_val.size(); i++)
        {
            temp_expr[i] = complete_tree(expression_val[i], i);
        }
        return temp_expr;
    }

    std::vector<std::string> get_legal_moves(int idx)
    {
        assert(n.size() > static_cast<decltype(n.size())>(idx));
        if (this->expression_type == "prefix")
        {
            if (this->pieces[idx].empty()) //At the beginning, self.pieces[idx] is empty, so the only legal moves are the operators...
            {
                if (this->n[idx] != 0) // if the depth is not 0
                {
                    return Board::__operators;
                }
                else // else it's the leaves
                {
                    return Board::una_bin_leaf_legal_moves_dict[false][false][true];
                }
            }
            int num_binary = this->__num_binary_ops(idx);
            int num_leaves = this->__num_leaves(idx);

            if (this->cache)
            {
                //basic constraints for depth
                bool una_allowed = (getPNdepth(pieces[idx], idx, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, false /*binary*/, true /*unary*/, false /*leaf*/).first <= this->n[idx]);
                bool bin_allowed = (getPNdepth(pieces[idx], idx, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, true /*binary*/, false /*unary*/, false /*leaf*/).first <= this->n[idx]);
                bool leaf_allowed = (!((num_leaves == num_binary + 1) || (getPNdepth(pieces[idx], idx, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, false /*binary*/, false /*unary*/, true /*leaf*/).first < this->n[idx] && (num_leaves == num_binary))));
                std::vector<std::string> legal_moves = Board::una_bin_leaf_legal_moves_dict[una_allowed][bin_allowed][leaf_allowed];
                assert(legal_moves.size());
                return legal_moves;
            }

            else
            {
                bool una_allowed = false, bin_allowed = false, leaf_allowed = false;
                if (Board::__binary_operators.size() > 0)
                {
                    pieces[idx].push_back(Board::__binary_operators[0]);
                    bin_allowed = (getPNdepth(pieces[idx], idx).first <= this->n[idx]);
                }
                if (Board::__unary_operators.size() > 0)
                {
                    pieces[idx][pieces[idx].size() - 1] = Board::__unary_operators[0];
                    una_allowed = (getPNdepth(pieces[idx], idx).first <= this->n[idx]);
                }
                pieces[idx][pieces[idx].size() - 1] = Board::__input_vars[0];
                leaf_allowed = (!((num_leaves == num_binary + 1) || (getPNdepth(pieces[idx], idx).first < this->n[idx] && (num_leaves == num_binary))));
                pieces[idx].pop_back();
                assert(!(!una_allowed && !bin_allowed && !leaf_allowed));

                return Board::una_bin_leaf_legal_moves_dict[una_allowed][bin_allowed][leaf_allowed];
            }
        }

        else //postfix
        {
            if (this->pieces[idx].empty()) //At the beginning, self.pieces[idx] is empty, so the only legal moves are the features and const
            {
                return Board::una_bin_leaf_legal_moves_dict[false][false][true];
            }
            int num_binary = this->__num_binary_ops(idx);
            int num_leaves = this->__num_leaves(idx);

            if (this->cache)
            {
                return Board::una_bin_leaf_legal_moves_dict[((num_leaves >= 1) && (getRPNdepth(pieces[idx], idx, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, true /*unary*/, false /*leaf*/).first <= this->n[idx]))][(num_binary != num_leaves - 1)][(getRPNdepth(pieces[idx], idx, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, false /*unary*/, true /*leaf*/).first <= this->n[idx])];
            }

            else
            {
                bool una_allowed = false, bin_allowed = (num_binary != num_leaves - 1), leaf_allowed = false;
                if (Board::__unary_operators.size() > 0)
                {
                    pieces[idx].push_back(Board::__unary_operators[0]);
                    una_allowed = ((num_leaves >= 1) && (getRPNdepth(pieces[idx], idx).first <= this->n[idx]));
                }

                pieces[idx][pieces[idx].size() - 1] = Board::__input_vars[0];
                leaf_allowed = (getRPNdepth(pieces[idx], idx).first <= this->n[idx]);

                pieces[idx].pop_back();
                //                assert(!(!una_allowed && !bin_allowed && !leaf_allowed));

                return Board::una_bin_leaf_legal_moves_dict[una_allowed][bin_allowed][leaf_allowed];
            }
        }

    }

    std::string _to_infix(int idx, bool show_consts = true)
    {
        std::stack<std::string> stack;
        bool is_prefix = (expression_type == "prefix");
        std::string result, token;
        int sz = static_cast<int>((this->simplify_original) ? this->pieces[idx].size() : this->temp_pieces[idx].size());
        for (int i = (is_prefix ? (sz - 1) : 0); (is_prefix ? (i >= 0) : (i < sz)); (is_prefix ? (i--) : (i++)))
        {
            token = ((this->simplify_original) ? this->pieces[idx][i] : this->temp_pieces[idx][i]);
//            puts(("\ntoken = "+token+"\n").c_str());
            if (is_const(token)) // leaf
            {
                if (token.compare(0, 5, "const") == 0 && show_consts)
                {
                    try
                    {
                        stack.push(std::to_string((this->params)(std::stoi(token.substr(5)))));
                    }
                    catch (std::invalid_argument& e)
                    {
                        printf("stoi in `std::string _to_infix(int idx, bool show_consts = true)` failed.\ntoken = %s\nerror = %s\nexiting\n", token.c_str(), e.what());
                        exit(1);
                    }
                }
                else
                {
                    stack.push(token);
                }
            }
            else if (is_unary(token)) // Unary operator
            {
                std::string operand = stack.top();
                stack.pop();
                result = token + "(" + operand + ")";
                stack.push(result);
            }
            else // binary operator
            {
                std::string right_operand = stack.top();
                stack.pop();
                std::string left_operand = stack.top();
                stack.pop();
                if (expression_type == "prefix")
                {
                    result = "(" + right_operand + " " + token + " " + left_operand + ")";
                }
                else
                {
                    result = "(" + left_operand + " " + token + " " + right_operand + ")";
                }
                stack.push(result);
            }
        }

        return stack.top();
    }

    std::string _to_infix(bool show_consts = true)
    {
        std::string temp;
        int sz = static_cast<int>((this->simplify_original) ? (this->pieces.size() - 1) : (this->temp_pieces.size() - 1));
        for (int jdx = 0; jdx < sz; jdx++)
        {
            temp += _to_infix(jdx, show_consts) + ", ";
        }
        temp += _to_infix(sz, show_consts);
        return temp;
    }

    //Returns the `expression_type` string form of the expression stored in the vector<std::string> parameter pieces
    std::string expression(int idx, bool show_consts = true)
    {
        std::string temp, token;
        temp.reserve(2*((this->simplify_original) ? this->pieces[idx].size() : this->temp_pieces[idx].size()));
        size_t sz = ((this->simplify_original) ? this->pieces[idx].size() : this->temp_pieces[idx].size()) - 1;
        for (size_t i = 0; i <= sz; i++)
        {
            token = ((this->simplify_original) ? this->pieces[idx][i] : this->temp_pieces[idx][i]);

            if ((token.compare(0, 5, "const") == 0) && show_consts)
            {
                try
                {
                    temp += ((i!=sz) ? std::to_string((this->params)(std::stoi(token.substr(5)))) + " " : std::to_string((this->params)(std::stoi(token.substr(5)))));
                }
                catch (std::invalid_argument& e)
                {
                    printf("stoi in `std::string expression(int idx, bool show_consts = true)` failed.\ntoken = %s\nerror = %s\nexiting\n", token.c_str(), e.what());
                    exit(1);
                }
            }
            else
            {
                temp += ((i!=sz) ? token + " " : token);
            }
        }
        return temp;
    }

    std::string expression(const std::vector<std::string>& pieces, bool show_consts = true)
    {
        std::string temp, token;
        temp.reserve(2*pieces.size());
        size_t sz = pieces.size() - 1;
        for (size_t i = 0; i <= sz; i++)
        {
            token = pieces[i];

            if ((token.compare(0, 5, "const") == 0) && show_consts)
            {
                temp += ((i!=sz) ? std::to_string((this->params)(std::stoi(token.substr(5)))) + " " : std::to_string((this->params)(std::stoi(token.substr(5)))));
            }
            else
            {
                temp += ((i!=sz) ? token + " " : token);
            }
        }
        return temp;
    }

    std::string expression(bool show_consts = true)
    {
        std::string temp;
        int sz = static_cast<int>((this->simplify_original) ? (this->pieces.size() - 1) : (this->temp_pieces.size() - 1));
        for (int jdx = 0; jdx < sz; jdx++)
        {
            temp += expression(jdx, show_consts) + ", ";
        }
        temp += expression(sz, show_consts);
        return temp;
    }

    std::string _to_infix(const std::vector<std::string>& pieces, bool show_consts = true)
    {
        std::stack<std::string> stack;
        bool is_prefix = (expression_type == "prefix");
        std::string result, token;
        size_t pieces_sz = pieces.size();
        for (int i = (is_prefix ? (static_cast<int>(pieces_sz) - 1) : 0); (is_prefix ? (i >= 0) : (i < static_cast<int>(pieces_sz))); (is_prefix ? (i--) : (i++)))
        {
            token = pieces[i];
            //puts(("\ntoken = "+token+"\n").c_str());
            if (this->is_const(token)) // leaf
            {
                if ((token.compare(0, 5, "const") == 0) && show_consts)
                {
                    stack.push(std::to_string((this->params)(std::stoi(token.substr(5)))));
                }
                else
                {
                    stack.push(token);
                }
            }

            else if (this->is_unary(token)) // Unary operator
            {
                std::string operand = stack.top();
                stack.pop();
                result = token + "(" + operand + ")";
                stack.push(result);
            }
            else // binary operator
            {
                std::string right_operand = stack.top();
                stack.pop();
                std::string left_operand = stack.top();
                stack.pop();
                if (expression_type == "prefix")
                {
                    result = "(" + right_operand + " " + token + " " + left_operand + ")";
                }
                else
                {
                    result = "(" + left_operand + " " + token + " " + right_operand + ")";
                }
                stack.push(result);
            }
        }

        return stack.top();
    }

    std::string _to_infix(const std::vector<std::vector<std::string>>& pieces, bool show_consts = true)
    {
        std::string temp;
        if (!pieces.size())
        {
            return temp;
        }
        decltype(pieces.size()) sz = pieces.size() - 1;
        for (decltype(pieces.size()) jdx = 0; jdx < sz; jdx++)
        {
            temp += _to_infix(pieces[jdx], show_consts) + ", ";
        }
        temp += _to_infix(pieces[sz], show_consts);
        return temp;
    }

    std::string expression(const std::vector<std::vector<std::string>>& pieces, bool show_consts = true)
    {
        std::string temp;
        if (!pieces.size())
        {
            return temp;
        }
        decltype(pieces.size()) sz = pieces.size() - 1;
        for (decltype(pieces.size()) jdx = 0; jdx < sz; jdx++)
        {
            temp += expression(pieces[jdx], show_consts) + ", ";
        }
        temp += expression(pieces[sz], show_consts);
        return temp;
    }
    
    ExprDag pieces_to_dag(const std::vector<std::string>& pieces) const
    {
        ExprDag dag;
        dag.nodes.reserve(pieces.size()); // upper bound

        // Intern table: key -> node id
        std::unordered_map<DagNodeKey, int, DagNodeKeyHash> intern;
        intern.reserve(pieces.size() * 2);

        auto intern_node = [&](ExprDag::Kind kind,
                               const std::string& token,
                               int c0,
                               int c1) -> int
        {
            DagNodeKey key{kind, token, c0, c1};
            auto it = intern.find(key);
            if (it != intern.end())
                return it->second;

            int id = static_cast<int>(dag.nodes.size());
            ExprDag::Node n;
            n.kind = kind;
            n.token = token;
            n.child0 = c0;
            n.child1 = c1;
            dag.nodes.push_back(std::move(n));
            intern.emplace(std::move(key), id);
            return id;
        };

        std::stack<int> st;
        const bool is_prefix = (expression_type == "prefix");

        for (int i = (is_prefix ? (static_cast<int>(pieces.size()) - 1) : 0);
             (is_prefix ? (i >= 0) : (i < static_cast<int>(pieces.size())));
             (is_prefix ? --i : ++i))
        {
            const std::string& token = pieces[i];
            assert(!token.empty());

            if (is_const(token)) // leaf
            {
                int id = intern_node(ExprDag::Kind::Leaf, token, -1, -1);
                st.push(id);
            }
            else if (is_unary(token))
            {
                if (st.empty())
                    throw std::runtime_error("Malformed expression: unary operator with empty stack: " + token);

                int child = st.top(); st.pop();
                int id = intern_node(ExprDag::Kind::Unary, token, child, -1);
                st.push(id);
            }
            else // binary
            {
                if (st.size() < 2)
                    throw std::runtime_error("Malformed expression: binary operator with <2 operands: " + token);

                int first = st.top(); st.pop();
                int second = st.top(); st.pop();

                // IMPORTANT:
                // This matches your evaluator’s pop order + its postfix swap logic.
                // - postfix: right_operand is second pop, left_operand is first pop in code,
                //            but the operation is (right op left) => semantic left=second, right=first
                // - prefix (scanning reversed): operation is (left op right) with left=first, right=second
                int left  = (expression_type == "postfix") ? second : first;
                int right = (expression_type == "postfix") ? first  : second;

                int id = intern_node(ExprDag::Kind::Binary, token, left, right);
                st.push(id);
            }
        }

        if (st.empty())
            throw std::runtime_error("Malformed expression: empty result stack.");

        if (st.size() != 1)
            throw std::runtime_error("Malformed expression: stack has extra items at end.");

        dag.root = st.top();
        return dag;
    }

    Eigen::VectorXd evaluate_dag(const Eigen::VectorXd& params, const ExprDag& dag) const
    {
        if (dag.root < 0 || dag.root >= static_cast<int>(dag.nodes.size()))
            throw std::runtime_error("Invalid DAG root.");

        const int N = static_cast<int>(dag.nodes.size());
        std::vector<std::optional<Eigen::VectorXd>> memo(N);

        auto eval_leaf = [&](const std::string& token) -> Eigen::VectorXd
        {
            // This is your leaf logic, unchanged in behavior.
            if (token.compare(0, 5, "const") == 0)
            {
                int temp_idx = std::stoi(token.substr(5));
                if (temp_idx >= params.size())
                {
                    throw std::runtime_error("\ntemp_idx = " + std::to_string(temp_idx)
                                             + "\nparams.size() = " + std::to_string(params.size())
                                             + "\nnum_consts = " + std::to_string(this->__num_consts())
                                             + "\nBoard::expression_dict.size() = " + std::to_string(Board::expression_dict.size()));
                }
                return Eigen::VectorXd::Ones(Board::data.numRows()) * params(temp_idx);
            }
            else if (token == "0")
            {
                return Eigen::VectorXd::Zero(Board::data.numRows());
            }
            else if (token == "1")
            {
                return Eigen::VectorXd::Ones(Board::data.numRows());
            }
            else if (token == "2")
            {
                return Eigen::VectorXd::Ones(Board::data.numRows()) * 2.0;
            }
            else if (token == "4")
            {
                return Eigen::VectorXd::Ones(Board::data.numRows()) * 4.0;
            }
            else if (isdouble(token))
            {
                return Eigen::VectorXd::Ones(Board::data.numRows()) * Stod(token);
            }
            else if (this->subs_dict.size() && this->subs_dict.count(token))
            {
                return this->subs_dict.at(token);
            }
            else
            {
                return Board::data[token];
            }
        };

        // Recursive lambda needs std::function (or a y-combinator).
        std::function<const Eigen::VectorXd&(int)> dfs = [&](int id) -> const Eigen::VectorXd&
        {
            auto& slot = memo[id];
            if (slot.has_value())
                return *slot;

            const auto& node = dag.nodes[id];

            if (node.kind == ExprDag::Kind::Leaf)
            {
                slot = eval_leaf(node.token);
                return *slot;
            }

            if (node.kind == ExprDag::Kind::Unary)
            {
                const Eigen::VectorXd& x = dfs(node.child0);

                if (node.token == "cos")      slot = x.array().cos();
                else if (node.token == "exp") slot = x.array().exp();
                else if (node.token == "sqrt")slot = x.array().sqrt();
                else if (node.token == "sin") slot = x.array().sin();
                else if (node.token == "asin" || node.token == "arcsin") slot = x.array().asin();
                else if (node.token == "log"  || node.token == "ln")     slot = x.array().log();
                else if (node.token == "tanh") slot = x.array().tanh();
                else if (node.token == "sech") slot = 1.0 / x.array().cosh();
                else if (node.token == "acos" || node.token == "arccos") slot = x.array().acos();
                else if (node.token == "~")    slot = (-x.array()).matrix();
                else if (node.token == "abs")  slot = x.array().cwiseAbs();
                else
                    throw std::runtime_error("Unknown unary operator in DAG: " + node.token);

                return *slot;
            }

            // Binary
            const Eigen::VectorXd& L = dfs(node.child0);
            const Eigen::VectorXd& R = dfs(node.child1);

            if (node.token == "+")      slot = (L.array() + R.array()).matrix();
            else if (node.token == "-") slot = (L.array() - R.array()).matrix();
            else if (node.token == "*") slot = (L.array() * R.array()).matrix();
            else if (node.token == "/") slot = (L.array() / R.array()).matrix();
            else if (node.token == "^") slot = (L.array().pow(R.array())).matrix();
            else
                throw std::runtime_error("Unknown binary operator in DAG: " + node.token);

            return *slot;
        };

        return dfs(dag.root); // returns a copy (memo holds the stored value)
    }
    
    double expression_evaluator(const Eigen::VectorXd& params, const std::vector<std::string>& pieces, double t) const
    {
        std::stack<double> stack;
        std::string token;
        bool is_prefix = (expression_type == "prefix");
        for (int i = (is_prefix ? (static_cast<int>(pieces.size()) - 1) : 0); (is_prefix ? (i >= 0) : (i < static_cast<int>(pieces.size()))); (is_prefix ? (i--) : (i++)))
        {
            token = pieces[i];
            assert(token.size());
            if (is_const(token)) //not an operator, i.e., a leaf
            {
                if (token.compare(0, 5, "const") == 0)
                {
                    int temp_idx = std::stoi(token.substr(5));
                    assert(temp_idx < params.size());
                    stack.push(params(temp_idx));
                }
                else if (token == "0")
                {
                    stack.push(0.0);
                }
                else if (token == "1")
                {
                    stack.push(1.0);
                }
                else if (token == "2")
                {
                    stack.push(2.0);
                }
                else if (token == "4")
                {
                    stack.push(4.0);
                }
                else if (isdouble(token))
                {
                    stack.push(Stod(token));
                }
                else if (token == "x0")
                {
                    stack.push(t);
                }
                else
                {
                    throw(std::runtime_error("bad token"));
                }
            }
            else if (is_unary(token)) // Unary operator
            {
                if (token == "cos")
                {
                    double temp = stack.top();
                    stack.pop();
                    stack.push(cos(temp));
                }
                else if (token == "exp")
                {
                    double temp = stack.top();
                    stack.pop();
                    stack.push(exp(temp));
                }
                else if (token == "sqrt")
                {
                    double temp = stack.top();
                    stack.pop();
                    stack.push(sqrt(temp));
                }
                else if (token == "sin")
                {
                    double temp = stack.top();
                    stack.pop();
                    stack.push(sin(temp));
                }
                else if (token == "asin" || token == "arcsin")
                {
                    double temp = stack.top();
                    stack.pop();
                    stack.push(asin(temp));
                }
                else if (token == "log" || token == "ln")
                {
                    double temp = stack.top();
                    stack.pop();
                    stack.push(log(temp));
                }
                else if (token == "tanh")
                {
                    double temp = stack.top();
                    stack.pop();
                    stack.push(tanh(temp));
                }
                else if (token == "sech")
                {
                    double temp = stack.top();
                    stack.pop();
                    stack.push(1.0/cosh(temp));
                }
                else if (token == "acos" || token == "arccos")
                {
                    double temp = stack.top();
                    stack.pop();
                    stack.push(acos(temp));
                }
                else if (token == "~") //unary minus
                {
                    double temp = stack.top();
                    stack.pop();
                    stack.push(-temp);
                }
                else if (token == "abs") //unary abs
                {
                    double temp = stack.top();
                    stack.pop();
                    stack.push(abs(temp));
                }
            }
            else // binary operator
            {
                double left_operand = stack.top();
                stack.pop();
                double right_operand = stack.top();
                stack.pop();
                if (token == "+")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand + left_operand) : (left_operand + right_operand)));
                }
                else if (token == "-")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand - left_operand) : (left_operand - right_operand)));
                }
                else if (token == "*")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand * left_operand) : (left_operand * right_operand)));
                }
                else if (token == "/")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand / left_operand) : (left_operand / right_operand)));
                }
                else if (token == "^")
                {
                    stack.push((expression_type == "postfix") ? (std::pow(right_operand, left_operand)) : (std::pow(left_operand, right_operand)));
                }
            }

        }
        return stack.top();
    }

    Eigen::VectorXd expression_evaluator(const Eigen::VectorXd& params, const std::vector<std::string>& pieces) const
    {
        std::stack<Eigen::VectorXd> stack;
        std::string token;
        bool is_prefix = (expression_type == "prefix");
        for (int i = (is_prefix ? (static_cast<int>(pieces.size()) - 1) : 0); (is_prefix ? (i >= 0) : (i < static_cast<int>(pieces.size()))); (is_prefix ? (i--) : (i++)))
        {
            token = pieces[i];
            assert(token.size());
            if (is_const(token)) //not an operator, i.e., a leaf
            {
                if (token.compare(0, 5, "const") == 0)
                {
                    int temp_idx = std::stoi(token.substr(5));
//                    assert(temp_idx < params.size());
                    if (temp_idx >= params.size())
                    {
                        throw std::runtime_error("\ntemp_idx = "+std::to_string(temp_idx)
                                                 +"\nparams.size() = "+std::to_string(params.size())
                                                 +"\nnum_consts = "+std::to_string(this->__num_consts())
                                                 +"\nBoard::expression_dict.size() = "+std::to_string(Board::expression_dict.size()));
                    }
                    stack.push(Eigen::VectorXd::Ones(Board::data.numRows())*params(temp_idx));
                }
                else if (token == "0")
                {
                    stack.push(Eigen::VectorXd::Zero(Board::data.numRows()));
                }
                else if (token == "1")
                {
                    stack.push(Eigen::VectorXd::Ones(Board::data.numRows()));
                }
                else if (token == "2")
                {
                    stack.push(Eigen::VectorXd::Ones(Board::data.numRows())*2.0);
                }
                else if (token == "4")
                {
                    stack.push(Eigen::VectorXd::Ones(Board::data.numRows())*4.0);
                }
                else if (isdouble(token))
                {
                    stack.push(Eigen::VectorXd::Ones(Board::data.numRows())*Stod(token));
                }
                else if (this->subs_dict.size() && this->subs_dict.count(token))
                {
                    stack.push(this->subs_dict.at(token));
                }
                else
                {
//                    printf("token = %s\n", token.c_str());
                    assert(token[0] == 'x');
//                    if(token[0] != 'x'){throw std::runtime_error("token = " + token + " not entered into subs_dict!");}
                    stack.push(Board::data[token]);
                }
            }
            else if (is_unary(token)) // Unary operator
            {
                if (token == "cos")
                {
                    Eigen::VectorXd temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().cos());
                }
                else if (token == "exp")
                {
                    Eigen::VectorXd temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().exp());
                }
                else if (token == "sqrt")
                {
                    Eigen::VectorXd temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().sqrt());
                }
                else if (token == "sin")
                {
                    Eigen::VectorXd temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().sin());
                }
                else if (token == "asin" || token == "arcsin")
                {
                    Eigen::VectorXd temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().asin());
                }
                else if (token == "log" || token == "ln")
                {
                    Eigen::VectorXd temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().log());
                }
                else if (token == "tanh")
                {
                    Eigen::VectorXd temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().tanh());
                }
                else if (token == "sech")
                {
                    Eigen::VectorXd temp = stack.top();
                    stack.pop();
                    stack.push(1/temp.array().cosh());
                }
                else if (token == "acos" || token == "arccos")
                {
                    Eigen::VectorXd temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().acos());
                }
                else if (token == "~") //unary minus
                {
                    Eigen::VectorXd temp = stack.top();
                    stack.pop();
                    stack.push(-temp.array());
                }
                else if (token == "abs") //unary abs
                {
                    Eigen::VectorXd temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().cwiseAbs());
                }
            }
            else // binary operator
            {
                Eigen::VectorXd left_operand = stack.top();
                stack.pop();
                Eigen::VectorXd right_operand = stack.top();
                stack.pop();
                if (token == "+")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand.array() + left_operand.array()) : (left_operand.array() + right_operand.array())));
                }
                else if (token == "-")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand.array() - left_operand.array()) : (left_operand.array() - right_operand.array())));
                }
                else if (token == "*")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand.array() * left_operand.array()) : (left_operand.array() * right_operand.array())));
                }
                else if (token == "/")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand.array() / left_operand.array()) : (left_operand.array() / right_operand.array())));
                }
                else if (token == "^")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand.array().pow(left_operand.array())) : (left_operand.array().pow(right_operand.array()))));
                }
            }
        }
        assert(stack.size());
        return stack.top();
    }

    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic> expression_evaluator(const std::vector<Eigen::AutoDiffScalar<Eigen::VectorXd>>& parameters, const std::vector<std::string>& pieces) const
    {
        std::stack<Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic>> stack;
        std::string token;
        bool is_prefix = (expression_type == "prefix");
        for (int i = (is_prefix ? (static_cast<int>(pieces.size()) - 1) : 0); (is_prefix ? (i >= 0) : (i < static_cast<int>(pieces.size()))); (is_prefix ? (i--) : (i++)))
        {
            token = pieces[i];
            assert(token.size());
            if (is_const(token)) // leaf
            {
                if (token.compare(0, 5, "const") == 0)
                {
                    //                    std::cout << "\nparameters[" << const_count << "] = " << parameters[const_count].value() << '\n';
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic>::Constant(Board::data.numRows(), parameters[std::stoi(token.substr(5))]));
                }
                else if (token == "0")
                {
                    //                    std::cout << "\nparameters[" << const_count << "] = " << parameters[const_count].value() << '\n';
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic>::Constant(Board::data.numRows(), 0.0));
                }
                else if (token == "1")
                {
                    //                    std::cout << "\nparameters[" << const_count << "] = " << parameters[const_count].value() << '\n';
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic>::Constant(Board::data.numRows(), 1.0));
                }
                else if (token == "2")
                {
                    //                    std::cout << "\nparameters[" << const_count << "] = " << parameters[const_count].value() << '\n';
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic>::Constant(Board::data.numRows(), 2.0));
                }
                else if (token == "4")
                {
                    //                    std::cout << "\nparameters[" << const_count << "] = " << parameters[const_count].value() << '\n';
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic>::Constant(Board::data.numRows(), 4.0));
                }
                else if (isdouble(token))
                {
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic>::Constant(Board::data.numRows(), Stod(token)));
                }
                else
                {
                    stack.push(Board::data[token]);
                }
            }
            else if (is_unary(token)) // Unary operator
            {
                if (token == "cos")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().cos());
                }
                else if (token == "exp")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().exp());
                }
                else if (token == "sqrt")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().sqrt());
                }
                else if (token == "sin")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().sin());
                }
                else if (token == "asin" || token == "arcsin")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().asin());
                }
                else if (token == "log" || token == "ln")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().log());
                }
                else if (token == "tanh")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().tanh());
                }
                else if (token == "sech")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(1/temp.array().cosh());
                }
                else if (token == "acos" || token == "arccos")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().acos());
                }
                else if (token == "~") //unary minus
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(-temp.array());
                }
                else if (token == "abs") //unary minus
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().cwiseAbs());
                }
            }
            else // binary operator
            {
                Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic> left_operand = stack.top();
                stack.pop();
                Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic> right_operand = stack.top();
                stack.pop();

                if (token == "+")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand.array() + left_operand.array()) : (left_operand.array() + right_operand.array())));
                }
                else if (token == "-")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand.array() - left_operand.array()) : (left_operand.array() - right_operand.array())));
                }
                else if (token == "*")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand.array() * left_operand.array()) : (left_operand.array() * right_operand.array())));
                }
                else if (token == "/")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand.array() / left_operand.array()) : (left_operand.array() / right_operand.array())));
                }
                else if (token == "^")
                {
                    stack.push(((expression_type == "postfix") ? ((left_operand.array()*(right_operand.array().log())).exp()) : ((right_operand.array()*(left_operand.array().log())).exp())));
                }
            }
        }
        return stack.top();
    }

    std::vector<Eigen::VectorXd> expression_evaluator(const Eigen::VectorXd& params, const std::vector<std::vector<std::string>>& pieces) const
    {
        size_t sz = pieces.size();
        std::vector<Eigen::VectorXd> temp(sz);

        for (size_t idx = 0; idx < sz; idx++) //looping over each equation
        {
            if (this->graph_eval)
            {
                ExprDag dag = pieces_to_dag(pieces[idx]);
                temp[idx] = evaluate_dag(params, dag);
            }
            else
            {
                temp[idx] = expression_evaluator(params, pieces[idx]);
            }
        }
        return temp;
    }

    std::vector<Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic>> expression_evaluator(const std::vector<Eigen::AutoDiffScalar<Eigen::VectorXd>>& params, const std::vector<std::vector<std::string>>& pieces) const
    {
        std::vector<Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXd>, Eigen::Dynamic>> temp;
        size_t sz = pieces.size();
        temp.reserve(sz);
        for (size_t idx = 0; idx < sz; idx++)
        {
            temp.push_back(expression_evaluator(params, pieces[idx]));
        }
        return temp;
    }

    Eigen::AutoDiffScalar<Eigen::VectorXd> grad_func(std::vector<Eigen::AutoDiffScalar<Eigen::VectorXd>>& inputs)
    {
        return SNE(expression_evaluator(inputs, this->diffeq_result));
    }

    /*
     x: parameter vector: (x_0, x_1, ..., x_{x.size()-1})
     g: gradient evaluated at x: (g_0(x_0), g_1(x_1), ..., g_{g.size()-1}(x_{x.size()-1}))
     */
    double operator()(Eigen::VectorXd& x, Eigen::VectorXd& grad)
    {
        if (this->fit_method == "LBFGS" || this->fit_method == "LBFGSB")
        {
            assert(grad.size() == x.size());
            double grad_piece_prefactor = (this->isConstTol) ? (this->isConstTol/(Board::data.numRows()*this->num_objectives)) : 0.0;
            double sne = SNE(expression_evaluator(x, this->diffeq_result));
            if (this->fit_grad_method == "naive_numerical")
            {
                double low_b, temp, low_inv_var_b;
                for (int i = 0; i < x.size(); i++) //finite differences wrt x evaluated at the current values x(i)
                {
                    //https://stackoverflow.com/a/38855586/18255427
                    temp = x(i);
                    x(i) -= 0.00001;
                    low_inv_var_b = grad_piece_prefactor;
                    if (low_inv_var_b)
                    {
                        low_inv_var_b /= VarianceSum(this->expression_evaluator(x, this->pieces)); //larger variance in SR expressions -> smaller penalty
                    }
                    low_b = SNE(expression_evaluator(x, this->diffeq_result)) + low_inv_var_b;
                    x(i) = temp + 0.00001;
                    low_inv_var_b = grad_piece_prefactor;
                    if (low_inv_var_b)
                    {
                        low_inv_var_b /= VarianceSum(this->expression_evaluator(x, this->pieces)); //larger variance in SR expressions -> smaller penalty
                    }
                    grad(i) = ((SNE(expression_evaluator(x, this->diffeq_result)) + low_inv_var_b) - low_b) / 0.00002;
                    x(i) = temp;
                }
            }

            else if (this->fit_grad_method == "autodiff")
            {
                size_t sz = x.size();
                std::vector<Eigen::AutoDiffScalar<Eigen::VectorXd>> inputs(sz);
                inputs.reserve(sz);
                for (size_t i = 0; i < sz; i++)
                {
                    inputs[i].value() = x(i);
                    inputs[i].derivatives() = Eigen::VectorXd::Unit(sz, i);
                }
                grad = grad_func(inputs).derivatives();
            }
            return sne;
        }
        else if (this->fit_method == "LevenbergMarquardt")
        {
//            if (Board::expression_dict.size() >= Board::max_expression_dict_sz)
//            {
//                std::scoped_lock str_lock(Board::thread_locker);
//                puts(("NOW 3326 Board::expression_dict.size() >= Board::max_expression_dict_sz\n(params.size() == num_consts) is "
//                     +std::to_string(this->params.size() == this->__num_consts())).c_str());
//            }
            auto temp = this->expression_evaluator(x, this->diffeq_result); //std::vector<Eigen::VectorXd>
            std::vector<Eigen::VectorXd> expr_eval_var;
            if (this->isConstTol)
            {
                expr_eval_var = Variance(this->expression_evaluator(x, this->pieces)); //std::vector<Eigen::VectorXd>
            }
            decltype(temp.size()) num_cols = temp.size();
            decltype(temp[0].size()) num_rows = temp[0].size();
            decltype(expr_eval_var.size()) num_piece_cols = expr_eval_var.size();
            auto num_piece_vals = num_piece_cols*num_rows;
            double grad_piece_prefactor = (this->isConstTol) ? (this->isConstTol/(num_piece_vals)) : 0.0;
            auto total_cols = num_cols + num_piece_cols;
            assert((!expr_eval_var.size()) || (num_rows == expr_eval_var[0].size()));
            assert(num_piece_cols == static_cast<decltype(num_piece_cols)>((this->isConstTol) ? this->num_objectives : 0));
            assert(grad.size() == static_cast<decltype(grad.size())>(num_rows*total_cols));
            for (decltype(num_rows) kdx = 0; kdx < num_rows; kdx++) //for each row
            {
                for (decltype(num_cols) ldx = 0; ldx < num_cols; ldx++) //first loop over each differential equation value
                {
                    grad(kdx*total_cols + ldx) = temp[ldx][kdx]; // Assign values directly
                }
                if (this->isConstTol)
                {
                    for (decltype(total_cols) ldx = num_cols, mdx = 0; ldx < total_cols; ldx++, mdx++) //then loop over each SR-expression value
                    {
                        grad(kdx*total_cols + ldx) = ((10.0 * grad_piece_prefactor) / expr_eval_var[mdx][kdx]);
                    }
                }
            }

            /*
                e.g. Imagine below is data, where first num_cols = 3 columns are temp and last num_piece_cols = 2 columns are expr_eval_var
                Then num_rows = 7, total_cols = 3+2 = 5

                    1 3 5 1 1
                    2 1 4 2 1
                    3 2 2 1 2
                    4 2 1 3 1
                    5 2 5 4 1
                    3 6 4 5 2
                    1 2 0 2 3

                Then we have
                grad[(0*5 + 0) = 0] = temp[0][0]
                grad[(0*5 + 1) = 1] = temp[1][0]
                grad[(0*5 + 2) = 2] = temp[2][0]
                grad[(0*5 + 3) = 3] = grad_piece_prefactor/expr_eval_var[0][0]
                grad[(0*5 + 4) = 4] = grad_piece_prefactor/expr_eval_var[1][0]
                grad[(1*5 + 0) = 5] = temp[0][1]
                grad[(1*5 + 1) = 6] = temp[1][1]
                grad[(1*5 + 2) = 7] = temp[2][1]
                grad[(1*5 + 3) = 8] = grad_piece_prefactor/expr_eval_var[0][1]
                grad[(1*5 + 4) = 9] = grad_piece_prefactor/expr_eval_var[1][1]
            */
        }
        return 0.0;
    }
    bool LBFGS()
    {
        bool improved = false;
        auto start_time = Clock::now();
        LBFGSpp::LBFGSParam<double> param;
        param.epsilon = 1e-6;
        param.max_iterations = this->num_fit_iter;
        //https://lbfgspp.statr.me/doc/LineSearchBacktracking_8h_source.html
        LBFGSpp::LBFGSSolver<double, LBFGSpp::LineSearchMoreThuente> solver(param); //LineSearchBacktracking, LineSearchBracketing, LineSearchMoreThuente, LineSearchNocedalWright
        double fx;

        Eigen::VectorXd eigenVec = this->params;
        double sne = SNE(expression_evaluator(this->params, this->diffeq_result));
        try
        {
            solver.minimize((*this), eigenVec, fx);
        }
        catch (std::runtime_error& e){}
        catch (std::invalid_argument& e){}

        //printf("sne = %f -> fx = %f\n", sne, fx);
        if (fx < sne)
        {
            //printf("sne = %f -> fx = %f\n", sne, fx);
            this->params = eigenVec;
            improved = true;
        }
        Board::fit_time = Board::fit_time + (timeElapsedSince(start_time));
        return improved;
    }

    bool LBFGSB()
    {
        bool improved = false;
        auto start_time = Clock::now();
        LBFGSpp::LBFGSBParam<double> param;
        param.epsilon = 1e-6;
        param.max_iterations = this->num_fit_iter;
        //https://lbfgspp.statr.me/doc/LineSearchBacktracking_8h_source.html
        LBFGSpp::LBFGSBSolver<double> solver(param); //LineSearchBacktracking, LineSearchBracketing, LineSearchMoreThuente, LineSearchNocedalWright
        double fx;

        Eigen::VectorXd eigenVec = this->params;
        double sne = SNE(expression_evaluator(this->params, this->diffeq_result));
        try
        {
            solver.minimize((*this), eigenVec, fx, Eigen::VectorXd::Constant(eigenVec.size(), -std::numeric_limits<double>::infinity()), Eigen::VectorXd::Constant(eigenVec.size(), std::numeric_limits<double>::infinity()));
            //solver.minimize((*this), eigenVec, fx, Eigen::VectorXd::Constant(eigenVec.size(), -10.f), Eigen::VectorXd::Constant(eigenVec.size(), 10.f));
        }
        catch (std::runtime_error& e){}
        catch (std::invalid_argument& e){}
        catch (std::logic_error& e){}

        //printf("sne = %f -> fx = %f\n", sne, fx);
        if (fx < sne)
        {
            //printf("sne = %f -> fx = %f\n", sne, fx);
            this->params = eigenVec;
            improved = true;
        }
        Board::fit_time = Board::fit_time + (timeElapsedSince(start_time));
        return improved;
    }

    int values() const
    {
        return Board::data.numRows()*(this->num_diff_eqns + (this->isConstTol ? this->num_objectives : 0));
    }

    int df(Eigen::VectorXd &x, Eigen::MatrixXd &fjac)
    {
        double epsilon, temp;
        epsilon = 1e-5;

        for (int i = 0; i < x.size(); i++)
        {
            //Eigen::VectorXd xPlus(x);
            //xPlus(i) += epsilon;
            //
            //Eigen::VectorXd xMinus(x);
            //xMinus(i) -= epsilon;
            //x(i) -= epsilon;

            temp = x(i);

            x(i) = temp + epsilon;
            Eigen::VectorXd fvecPlus(values());
            operator()(x, fvecPlus);

            x(i) = temp - epsilon;
            Eigen::VectorXd fvecMinus(values());
            operator()(x, fvecMinus);

            fjac.block(0, i, values(), 1) = std::move((fvecPlus - fvecMinus) / (2.0 * epsilon));

            x(i) = temp;
        }
        return 0;
    }

    bool LevenbergMarquardt()
    {
        bool improved = false;
        auto start_time = Clock::now();
        Eigen::LevenbergMarquardt<decltype(*this), double> lm(*this);
        double score_before = SNE(expression_evaluator(this->params, this->diffeq_result));
        lm.parameters.maxfev = this->num_fit_iter;
        //std::cout << "ftol (Cost function change) = " << lm.parameters.ftol << '\n';
        //std::cout << "xtol (Parameters change) = " << lm.parameters.xtol << '\n';
        lm.minimize(this->params);
        double score_after = SNE(expression_evaluator(this->params, this->diffeq_result));
        if (score_after < score_before)
        {
//            printf("score_before = %f -> score_after = %f\n", score_before, score_after);
            improved = true;
        }
        //std::cout << "Iterations = " << lm.nfev << '\n';
        Board::fit_time = Board::fit_time + (timeElapsedSince(start_time));
        return improved;
    }

    //Returns `true` if each expression in `this->pieces` with parameters `this->params`
    //has variance greater than or equal to `this->isConstTol`.
    //Otherwise it returns `false`.
    bool passesConstantThreshold()
    {
        for (decltype(this->pieces.size()) jdx = 0; jdx < this->pieces.size(); jdx++) //loops over each generated symbolic expression
        {
            if (isConstant(expression_evaluator(this->params, this->pieces[jdx]), this->isConstTol))
            {
                return false;
            }
        }
        return true;
    }

    double fitFunctionToData()
    {
        double score = 0.0;
        bool checkMaxSize = (this->maxSize.size() == this->pieces.size());
        for (decltype(this->pieces.size()) jdx = 0; jdx < this->pieces.size(); jdx++) //loop over each generated symbolic expression
        {
            //This block checks if the max-size constraint is violated by any of the `this->n.size()` expressions
            if (checkMaxSize && this->pieces[jdx].size() > this->maxSize[jdx])
            {
                this->SNE_curr = DBL_MAX;
                return score;
            }
            
            //This block below checks if `this->pieces[jdx]` has nans or infs or bad_ops
            for (const auto& piece: this->pieces[jdx])
            {
                assert(piece.size());
                if (this->is_bad_op(piece))
                {
                    this->SNE_curr = DBL_MAX;
                    return score;
                }
                for (int i = 0; i < static_cast<int>(piece.size())-2; i++)
                {
                    //checks if next 3 characters are 'n', 'a', 'n' or 'i', 'n', 'f'
                    if (((piece[i] == 'n') && (piece[i+1] == 'a') && (piece[i+2] == 'n')) ||
                        ((piece[i] == 'i') && (piece[i+1] == 'n') && (piece[i+2] == 'f')))
                    {
                        this->SNE_curr = DBL_MAX;
                        return score;
                    }
                }
                
            }
            //If all of the features must be non-trivially in each expression in the vector of expressions `pieces`
            if (this->mustHaveAllFeatures)
            {
                std::vector<int> grasp;
                //MARK: Might want to add `passesConstantThreshold` here in the future...
                //Eigen::VectorXd expression_eval = expression_evaluator(this->params, this->pieces[jdx]);
                for (const std::string& i: Board::__input_vars)
                {
                    //If the independent variable `i` is NOT present in the expression `this->pieces[jdx]`...
                    if (std::find(this->pieces[jdx].begin(), this->pieces[jdx].end(), i) == this->pieces[jdx].end())
                    {
                        //then `this->pieces[jdx]` does not depend on `i`, so this is a trivial expression -> get out of dodge!
                        this->SNE_curr = DBL_MAX;
                        return score;
                    }
                    //If the variable `i` is found, we then test the derivative wrt, `i` to check if it's 0 within `this->isConstTol` tolerance.
                    if (this->isConstTol > 0)
                    {
                        if (this->expression_type == "prefix")
                        {
                            this->derivePrefix(0, this->pieces[jdx].size() - 1, i, this->pieces[jdx], grasp);
                        }
                        else //postfix
                        {
                            this->derivePostfix(0, this->pieces[jdx].size() - 1, i, this->pieces[jdx], grasp);
                        }
                        if (isZero(expression_evaluator(this->params, this->derivat), this->isConstTol)) //Ignore the trivial solution (N-d functions)!
                        {
                            this->SNE_curr = DBL_MAX;
                            return score;
                        }
                    }
                }
            }
            else if ((this->customFeatures.size() > jdx) && (this->customFeatures[jdx].size()))
            {
                std::vector<int> grasp;
                //MARK: Might want to add `passesConstantThreshold` here in the future...
                //Eigen::VectorXd expression_eval = expression_evaluator(this->params, this->pieces[jdx]);
                for (const std::string& i: this->customFeatures[jdx])
                {
                    //If the independent variable `i` is NOT present in the expression `this->pieces[jdx]`...
                    if (std::find(this->pieces[jdx].begin(), this->pieces[jdx].end(), i) == this->pieces[jdx].end())
                    {
                        //then `this->pieces[jdx]` does not depend on `i`, so this is a trivial expression -> get out of dodge!
                        this->SNE_curr = DBL_MAX;
                        return score;
                    }
                    //If the variable `i` is found, we then test the derivative wrt, `i` to check if it's 0 within `this->isConstTol` tolerance.
                    if (this->isConstTol > 0)
                    {
                        if (this->expression_type == "prefix")
                        {
                            this->derivePrefix(0, this->pieces[jdx].size() - 1, i, this->pieces[jdx], grasp);
                        }
                        else //postfix
                        {
                            this->derivePostfix(0, this->pieces[jdx].size() - 1, i, this->pieces[jdx], grasp);
                        }
                        if (isZero(expression_evaluator(this->params, this->derivat), this->isConstTol)) //Ignore the trivial solution (N-d functions)!
                        {
                            this->SNE_curr = DBL_MAX;
                            return score;
                        }
                    }
                }
            }
        }
        //Now that we've weeded out bad expressions, we fit/evaluate
        if (this->params.size()) //fit and evaluate
        {
            assert(this->num_consts_diff || this->use_const_pieces);
            this->diffeq_result = diffeq(*this, true);
            if (this->SNE_curr_vec.size() != this->diffeq_result.size())
            {
                this->SNE_curr_vec.assign(this->diffeq_result.size(), 0); //one sne value initialized to 0 per equation in the system we're trying to solve.
            }
            assert(this->diffeq_result.size() == static_cast<decltype(this->diffeq_result.size())>(this->num_diff_eqns));
            for (decltype(this->diffeq_result.size()) jdx = 0; jdx < this->diffeq_result.size(); jdx++)
            {
                ((this->expression_type == "prefix") ? simplifyPN(this->diffeq_result[jdx]) : simplifyRPN(this->diffeq_result[jdx]));
            }
            bool improved = true;
            if (this->fit_method == "LBFGS")
            {
                improved = LBFGS();
            }
            else if (this->fit_method == "LBFGSB")
            {
                improved = LBFGSB();
            }
            else if (this->fit_method == "LevenbergMarquardt")
            {
                improved = LevenbergMarquardt();
            }
            Eigen::VectorXd temp_vec; //need to have a back-up vector in case `improved == false` so we can get the score of the expression we just built.

            if (improved && this->passesConstantThreshold()) //If improved, update the expression_dict with this->params
            {
                //If the `Board::max_expression_dict_sz` hasn't been exceeded, add it to `Board::expression_dict`
                if (Board::expression_dict.contains(this->expression_string)) //If the expression has been visited before (it's already in `Board::expression_dict`)
                {
                    Board::expression_dict.visit(this->expression_string, [&](auto& x) //simply update the corresponding parameter vector with
                    {
                        x.second = this->params;
                    });
                }
                else if (Board::expression_dict.size() < Board::max_expression_dict_sz) //Else if there's capacity to add the new expression-params pair to `Board::expression_dict`
                {
                    Board::expression_dict.insert_or_assign(this->expression_string, this->params);
                }
            }
            if (Board::expression_dict.contains(this->expression_string))
            {
                Board::expression_dict.cvisit(this->expression_string, [&](const auto& x)
                {
                    temp_vec = x.second;
                });
            }
            else //Once `Board::expression_dict.size() >= Board::max_expression_dict_sz`, this can happen
            {
                temp_vec.setOnes(this->params.size());
            }
            auto expected = this->__num_consts();
            if (temp_vec.size() != expected)
            {
                #ifndef NDEBUG
                    std::cerr
                        << "[SR DEBUG] Param size mismatch — "
                        << "expected " << expected
                        << ", got " << temp_vec.size()
                        << " | expr: " << this->expression_string
                        << '\n';
                #endif // !NDEBUG
                temp_vec.setOnes(expected);
                if (this->params.size() != expected)
                {
                    this->params.setOnes(expected);
                    Board::expression_dict.insert_or_assign(this->expression_string, this->params);
                }
            }
            std::vector<Eigen::VectorXd> expression_eval = expression_evaluator(temp_vec, this->diffeq_result);

            score = 0.0;
            this->SNE_curr = SNE(expression_eval[0]);
            if (isInvalid(this->SNE_curr))
            {
                this->SNE_curr = DBL_MAX;
                return 0.0;
            }
            this->SNE_curr_vec[0] = this->SNE_curr;

            for (size_t jdx = 1; jdx < expression_eval.size(); ++jdx)
            {
                this->SNE_curr_vec[jdx] = SNE(expression_eval[jdx]);
                if (isInvalid(this->SNE_curr_vec[jdx]))
                {
                    this->SNE_curr = DBL_MAX;
                    return 0.0;
                }
                this->SNE_curr += this->SNE_curr_vec[jdx];

            }
            score = 1.0/(1.0+this->SNE_curr);
            this->params = temp_vec; //copy `temp_vec` back into `this->params` for displaying purposes
        }
        else //just evaluate
        {
            this->diffeq_result = diffeq(*this, false);
            assert(all_checks(this->diffeq_result));
            if (this->SNE_curr_vec.size() != this->diffeq_result.size())
            {
                this->SNE_curr_vec.assign(this->diffeq_result.size(), 0);
            }
            assert(this->diffeq_result.size() == static_cast<decltype(this->diffeq_result.size())>(this->num_diff_eqns));
            score = 0.0;
            this->SNE_curr = 0.0;
            for (decltype(this->diffeq_result.size()) jdx = 0; jdx < this->diffeq_result.size(); jdx++)
            {
                ((this->expression_type == "prefix") ? simplifyPN(this->diffeq_result[jdx]) : simplifyRPN(this->diffeq_result[jdx]));
                auto temp_data = expression_evaluator(this->params, this->diffeq_result[jdx]);

                this->SNE_curr_vec[jdx] = SNE(temp_data);
                if (isInvalid(this->SNE_curr_vec[jdx]))
                {
                    this->SNE_curr = DBL_MAX;
                    return 0.0;
                }
                this->SNE_curr += this->SNE_curr_vec[jdx];
            }
            score = 1.0/(1.0+this->SNE_curr);
        }
        return score;
    }

    /*
     Check whether the given player has created a
     complete (depth self.n) expression (again), and
     checks if it is a complete PN/RPN expression.
     Returns the score of the expression if complete,
     where 0 <= score <= 1 and -1 if not complete or if
     the desired depth has not been reached.
     */
    //TODO: maybe only fit the constants of the ones that yield, when set to 1, some percentage of the best thus-far (like maybe 80%)...?
    double complete_status(int idx, bool cache = true)
    {
        assert(this->stack.size() > static_cast<decltype(this->stack.size())>(idx));
        assert(n.size() > static_cast<decltype(n.size())>(idx));
        if (this->pieces[idx].empty())
        {
            this->stack[idx] = std::vector<int>();
            this->idx[idx] = 0;
            if (this->expression_type == "prefix")
            {
                this->depth[idx] = 0, this->num_binary[idx] = 0, this->num_leaves[idx] = 0;
            }
        }
        assert(this->stack.size() > static_cast<decltype(this->stack.size())>(idx));
        auto [depth, complete] =  ((this->expression_type == "prefix") ? getPNdepth(pieces[idx], idx, 0 /*start*/, 0 /*stop*/, this->cache && cache /*cache*/, true /*modify*/) : getRPNdepth(pieces[idx], idx, 0 /*start*/, 0 /*stop*/, this->cache && cache /*cache*/, true /*modify*/)); //structured binding :)
        if (!complete || depth < this->n[idx]) //Expression not complete
        {
            return -1.0;
        }
        else if (idx < (static_cast<long long>(this->pieces.size()) - 1))
        {
            return 0.0;
        }
        else
        {
            if (visualize_exploration)
            {
                //whenever. TODO: call some plotting function, e.g. ROOT CERN plotting API, Matplotlib from the Python-C API, Plotly if we want a web application for this, etc. The plotting function could also have the fitted constants (rounded of course), but then this if statement would need to be moved down to below the fitFunctionToData call in this `complete_status` method.
            }
            if (is_primary)
            {
                if (this->simplify_original)
                {
                    for (decltype(this->pieces.size()) jdx = 0; jdx < this->pieces.size(); jdx++)
                    {
                        ((this->expression_type == "prefix") ? simplifyPN(this->pieces[jdx]) : simplifyRPN(this->pieces[jdx])); //simplify expression
                    }
                }
                else
                {
                    this->temp_pieces = this->pieces;
                    for (decltype(this->pieces.size()) jdx = 0; jdx < this->pieces.size(); jdx++)
                    {
                        ((this->expression_type == "prefix") ? simplifyPN(this->pieces[jdx]) : simplifyRPN(this->pieces[jdx])); //simplify expression
                    }
                }
                if (this->num_consts_diff || this->use_const_pieces) //If I have tokens that need to be optimized
                {
                    this->expression_string.clear();
                    this->expression_string.reserve(8*pieces.size());

                    for (decltype(this->pieces.size()) jdx = 0; jdx < this->pieces.size(); jdx++)
                    {
                        for (const std::string& token: this->pieces[jdx])
                        {
                            this->expression_string += token+" ";
                        }
                        this->expression_string += ((jdx < this->pieces.size() - 1) ? ", " : "");
                    }

                    if (!Board::expression_dict.contains(this->expression_string)) //If the generated expression has NOT been generated before...
                    {
                        //insert it into the shared dictionary of `{expressions: best_fit_params}` key-value pairs...
                        try //MARK: Might be able to remove this try-catch block itf.
                        {
                            if (Board::expression_dict.size() < Board::max_expression_dict_sz) //if the capacity of the shared dict has not been exceeded.
                            {
                                Board::expression_dict.insert_or_assign(this->expression_string, Eigen::VectorXd());
                            }
                        }
                        catch (const std::bad_alloc& e)
                        {
                            std::scoped_lock str_lock(Board::thread_locker);
                            std::cerr << "Inserting into Board::expression_dict failed, Board::expression_dict.size() = "
                                      << Board::expression_dict.size() << '\n';
                            exit(1);
                        }
                    }
                    if (Board::expression_dict.contains(this->expression_string))
                    {
                        Board::expression_dict.cvisit(this->expression_string, [&](const auto& x)
                        {
                            this->params = x.second;
                        });
                    }

                    int piece_const_counter = this->num_consts_diff;
                    if (this->use_const_pieces)
                    {
                        for (decltype(this->pieces.size()) jdx = 0; jdx < this->pieces.size(); jdx++)
                        {
                            for (std::string& token: this->pieces[jdx])
                            {
                                if (token.compare(0, 5, "const") == 0)
                                {
                                    if (token.size() == 5)
                                    {
                                        token += std::to_string(piece_const_counter++);
                                    }
                                    else //MAYBE: Might be able to remove this else-statement itf.
                                    {
                                        assert((token.size() > 5));
                                        std::string int_suffix = token.substr(5);
                                        int temp_idx = std::stoi(int_suffix);
                                        //the differential equation constants are const0, const1, ..., const{this->num_consts_diff-1},
                                        //and the constant tokens in the candidate solutions are simply const without an integer suffix,
                                        //so if we find something like const{(val>=this->num_consts_diff)} then that's unexpected
                                        assert((static_cast<decltype(this->num_consts_diff)>(temp_idx) < this->num_consts_diff));
                                        if (static_cast<decltype(this->num_consts_diff)>(temp_idx) >= this->num_consts_diff)
                                        {
                                            throw std::runtime_error(("Somehow, there's a const token with a suffix integer equal to " + std::to_string(temp_idx)));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if (this->params.size() != piece_const_counter)
                    {
                        this->params.setOnes(piece_const_counter);
                        try //MAYBE: Might be able to remove this try-catch block itf.
                        {
                            if (Board::expression_dict.size() < Board::max_expression_dict_sz)
                            {
                                Board::expression_dict.insert_or_assign(this->expression_string, this->params);
                            }
                        }
                        catch (const std::bad_alloc& e)
                        {
                            std::scoped_lock str_lock(Board::thread_locker);
                            std::cerr << "Inserting into Board::expression_dict failed, Board::expression_dict.size() = "
                                      << Board::expression_dict.size() << '\n';
                            exit(1);
                        }
                    }
                    assert((this->params.size() == piece_const_counter));
                }
                double res = fitFunctionToData();
                if (!this->simplify_original)
                {
                    //this->pieces = this->temp_pieces;
                    this->pieces.swap(this->temp_pieces);
                }
                return res;
            }
            return 0.0;
        }
    }
    const Eigen::VectorXd& operator[] (int i)
    {
        return Board::data[i];
    }
    const Eigen::VectorXd& operator[] (const std::string& i)
    {
        return Board::data[i];
    }

    friend std::ostream& operator<<(std::ostream& os, const Board& b)
    {
        return (os << b.data);
    }

    //Function to compute the LGB or RGB, from https://www.jstor.org/stable/43998756
    //(top of pg. 165)
    void GB(size_t z, int& ind, const std::vector<std::string>& individual)
    {
        do
        {
            ind = ((expression_type == "prefix") ? std::min(static_cast<int>(individual.size()) - 1, ind+1) : std::max(0, ind-1));
            if (is_unary(individual[ind]))
            {
                GB(1, ind, individual);
            }
            else if (is_binary(individual[ind]))
            {
                GB(2, ind, individual);
            }
            --z;
        } while (z);
    }

    //Computes the grasp of an arbitrary element pieces[i],
    //from https://www.jstor.org/stable/43998756 (bottom of pg. 165)
    int GR(int i, const std::vector<std::string>& individual)
    {
        int start = i;
        int& ptr_lgb = start;
        if (is_unary(individual[i]))
        {
            GB(1, ptr_lgb, individual);
        }
        else if (is_binary(individual[i]))
        {
            GB(2, ptr_lgb, individual);
        }
        return ((expression_type == "prefix") ? ( ptr_lgb - i) : (i - ptr_lgb));
    }

    //Adds pairs containing the starting and stopping indices for each
    //depth `this->n[idx]` sub-expression in the expression individual
    void get_indices(std::vector<std::pair<int, int>>& sub_exprs, std::vector<std::string>& individual, int idx)
    {
        assert(individual.size());
        int temp;
        int sz = static_cast<int>(individual.size());
        for (int k = 0; k < sz; k++)
        {
            temp = k; //we don't want to change k
            int& ptr_GB = temp;

            if (is_unary(individual[k]))
            {
                GB(1, ptr_GB, individual);
            }
            else if (is_binary(individual[k]))
            {
                GB(2, ptr_GB, individual);
            }
            else if (this->n[idx] == 0) //then, in this case, individual[k] is a leaf node (since it's not a unary or binary operator as tested above), and depth-0 sub-trees are leaf-nodes
            {
                sub_exprs.push_back(std::make_pair(k, k));
                continue;
            }

            auto [start, stop] = std::make_pair( std::min(k, ptr_GB), std::max(k, ptr_GB));
            auto [depth, complete] =  ((expression_type == "prefix") ? getPNdepth(individual, idx, start, stop+1, false /*cache*/) : getRPNdepth(individual, idx, start, stop+1, false /*cache*/));

            if (complete && (depth == this->n[idx])) //if the subexpression starting this index `k` corresponds to a complete depth `this->n` sub-expression, then push it (the starting and stopping indices) back into the container `sub_exprs`
            {
                sub_exprs.push_back(std::make_pair(start, stop));
            }
        }
    }

    bool areDerivatRangesEqual(int start_idx_1, int start_idx_2, int num_steps)
    {
        int stop_idx_1 = start_idx_1 + num_steps;

        for (int i = start_idx_1, j = start_idx_2; i < stop_idx_1; i++, j++)
        {
            if (derivat[i] != derivat[j])
            {
                return false;
            }
        }
        return true;
    }

    bool areExpressionRangesEqual(int start_idx_1, int start_idx_2, int num_steps, const std::vector<std::string>& expression)
    {
        int stop_idx_1 = start_idx_1 + num_steps;

        for (int i = start_idx_1, j = start_idx_2; i < stop_idx_1; i++, j++)
        {
            if (expression[i] != expression[j])
            {
                return false;
            }
        }
        return true;
    }

    void setPrefixGR(const std::vector<std::string>& prefix, std::vector<int>& grasp)
    {
        grasp.reserve(prefix.size());
        for (size_t k = 0; k < prefix.size(); ++k)
        {
            grasp.push_back(GR(k, prefix));
        }
    }

    /*
     low and up: lower and upper Index bounds, respectively, for the piece of the array prefix which is to be the subject of the processing.
     dx: string representing the variable by which the derivation is to be made. (The derivative is made wrt dx)
     */
    void derivePrefixHelper(int low, int up, const std::string& dx, const std::vector<std::string>& prefix, std::vector<int>& grasp, bool setGRvar = false)
    {
        if (!setGRvar)
        {
            grasp.clear();
            this->derivat.clear();
            // std::cout << this->derivat.size();
            this->derivat.reserve(1000);
            //        Index = 0;
            setPrefixGR(prefix, grasp);
        }
        //allowed ops: +, -, *, /, ^, unary +, unary -, sin(), cos(), tan(), ctg(), log(), sqrt(), const, x0, x1, ..., x_numFeatures
        //Define `grasp` of prefix[i], i.e., the number of elements forming operands of prefix[i] (grasp(operand) = 0)
        //The grasped elements of prefix[i] are the elements forming operands of prefix[i]
        //The left-grasp-bound (LGB) of prefix[i] is the Index of the left-most grasped element of prefix[i] in the array prefix
        //For the expression formed by prefix[i] and its grasped elements, the element prefix[i] is termed the `head` or `main element`
        //ANY element prefix[i] is an N-arity operator acting on operands arg1, ..., argN, the heads of which are op1, ..., opN, where op1 is the left-neighest operator of prefix[i] in the array prefix (so operands are 0-arity operators)
        //For an N-arity operator prefix[i]:
        //The `grasp` of prefix[i]  is equal to i - LGB(prefix[i])
        //grasp(prefix[i]) = N + sum(grasp(op1), ..., grasp(op_k), ..., grasp(opN))
        //grasp(prefix[i]) = N + sum( (1 - grasp(op1)), ..., (k - grasp(op_k)), ..., (N - grasp(opN)))
        //LGB(prefix[i]) = i - N - sum( (1 - grasp(op1)), ..., (k - grasp(op_k)), ..., (N - grasp(opN)))
        //op_(N-j) = prefix[i - sum(grasp(op_(N-1)), ..., grasp(op_(N-j-1))) - j - 1], where j = 0, ..., N-1
        //If the grasp of an arbitrary prefix[i] is greater than N, then at least one of its argument heads is also an operator.
        //Example: If the grasp of any binary operator prefix[i] is greater than 2, then at least one of the two preceding elements in the RPN of the expression (prefix[i-1] and prefix[i-2]) is also an operator (unary or binary).
        //prefix[numElements] is certainly an operator (unary or binary)

        //if prefix[up] is a binary operator, then:
        //the head of its second argument (let's call it op2) is equal to prefix[up-1]
        //then the grasped elements of op2 are the elements from prefix[up-1-grasp[up-1]] to prefix[up-1]
        //e.g. prefix = {"x", "x", "*", "x", "cos", "x", "*", "+"}, up = 7 -> prefix[up] = "+" is binary
        //so prefix[up-1] = "*" is the head of the second argument of "+" and so the grasped elements
        //of prefix[up-1] are the elements [(prefix[up-1-grasp[up-1]] = prefix[6-3] = prefix[3]), prefix[up-1] = prefix[6]]
        //i.e., the elements {"x", "cos", "x", "*"}
        //the head of its first argument (lets call it op1) is equal to prefix[up-grasp(op2)-2] which is equal to prefix[up-2-grasp[up-1]].
        //then the grasped elements of op1 are the elements from prefix[low = 0] to prefix[up-2-grasp[up-1]]
        //e.g. prefix = {"x", "x", "*", "x", "cos", "x", "x", "*", "*", "+"}, up = 9 ->prefix[up] = "+" is binary
        //so prefix[up-grasp(op2)-2] = prefix[9-5-2] = prefix[2] = "*" is the head of the first argument of "+" and so the grasped elements
        //of prefix[up-grasp(op2)-2] are the elements [prefix[low] = prefix[0], prefix[up-grasp(op2)-2] = prefix[9-5-2] = prefix[2]]
        //i.e., the elements {"x", "x", "*"}

        if (std::find(prefix.begin(), prefix.end(), dx) == prefix.end())
        {
            this->derivat.push_back("0");
            return;
        }

        if (prefix[low] == "+" || prefix[low] == "-")
        {
            int op_idx = this->derivat.size();
            this->derivat.push_back(prefix[low]); //+/-
            int temp = low+1+grasp[low+1];
            int x_prime_low = this->derivat.size();
            derivePrefixHelper(low+1, temp, dx, prefix, grasp, true);  /* +/- x' */
            int x_prime_high = this->derivat.size();
            derivePrefixHelper(temp+1, temp+1+grasp[temp+1], dx, prefix, grasp, true); /* +/- x' y' */
            int y_prime_high = derivat.size();
            int step;

            /*
             Simplification cases:

             1.) y' == 0, +/-, x'
             2.) x' == 0,   +, y'
             3.) x' == 0,   -, ~ y'

             */

            if (derivat[x_prime_high] == "0") //1.) +/- x' 0 -> x'
            {
                //            puts("hi 147");
                //remove y'
                if (x_prime_high == static_cast<int>(derivat.size()) - 1)
                {
                    derivat.pop_back();
                }
                else
                {
                    derivat.erase(derivat.begin() + x_prime_high, derivat.end());
                }
                derivat.erase(derivat.begin() + op_idx); //remove +/- operator at beginning
            }

            else if (derivat[x_prime_low] == "0") //2.) and 3.)
            {
                //            puts("hi 162");
                if (prefix[low] == "+") //2.) + 0 y' -> y'
                {
                    derivat.erase(derivat.begin() + op_idx, derivat.begin() + x_prime_high); //remove "+" and "x'"
                }
                else //3.) prefix[low] == "-", - 0 y' -> ~ y'
                {
                    //                puts("hi 170");
                    derivat[op_idx] = "~"; //change binary minus to unary minus
                    derivat.erase(derivat.begin() + x_prime_low); //remove x'
                }
            }
            else if ((prefix[low] == "-") && ((step = (y_prime_high - x_prime_high)) == (x_prime_high - x_prime_low)) && (areDerivatRangesEqual(x_prime_low, x_prime_high, step)))
            {
                //                puts("hi 194");
                assert(derivat[op_idx] == prefix[low]);
                derivat[op_idx] = "0"; //change "-" to "0";
                derivat.erase(derivat.begin() + op_idx + 1, derivat.begin() + y_prime_high);
            }
        }
        else if (prefix[low] == "*")
        {
            derivat.push_back("+"); /* +  */
            derivat.push_back("*"); /* + * */
            int x_low = derivat.size();
            int temp = low+1+grasp[low+1];
            for (int k = low+1; k <= temp; k++) /* + * x */
            {
                derivat.push_back(prefix[k]);
            }
            if (derivat[x_low] == "0") //* 0 y' -> 0
            {
                //            puts("hi 187");
                derivat[x_low - 1] = "0"; //change "*" to "0"
                derivat.erase(derivat.begin() + x_low); //erase x
            }
            else
            {
                int y_prime_low = derivat.size();
                derivePrefixHelper(temp+1, temp+1+grasp[temp+1], dx, prefix, grasp, true); /* + * x y' */
                if (derivat[y_prime_low] == "0") //* x 0 -> 0
                {
                    //                puts("hi 197");
                    derivat[x_low - 1] = "0"; //change "*" to "0"
                    derivat.erase(derivat.begin() + x_low, derivat.end()); //erase x and y'
                }
                else if (derivat[x_low] == "1") //* 1 y' -> y'
                {
                    //                puts("hi 203");
                    derivat.erase(derivat.begin() + x_low - 1, derivat.begin() + x_low + 1); //erase "*" and "1"
                }
                else if (derivat[y_prime_low] == "1") //* x 1 -> x
                {
                    //                puts("hi 208");
                    derivat.pop_back(); //remove "1"
                    derivat.erase(derivat.begin() + x_low - 1); //remove "*"
                }
            }
            derivat.push_back("*"); /* + * x y' * */
            int x_prime_low = derivat.size();
            derivePrefixHelper(low+1, temp, dx, prefix, grasp, true); /* + * x y' * x' */
            if (derivat[x_prime_low] == "0") //* 0 y -> 0
            {
                //            puts("hi 218");
                derivat[x_prime_low - 1] = "0"; //change "*" to "0"
                derivat.erase(derivat.begin() + x_prime_low); //erase x'
            }
            else
            {
                int y_low = derivat.size();
                for (int k = temp+1; k <= temp+1+grasp[temp+1]; k++)
                {
                    derivat.push_back(prefix[k]); /* + * x y' * x' y */
                }
                if (derivat[y_low] == "0") //* x' 0 -> 0
                {
                    //                puts("hi 231");
                    derivat[x_prime_low - 1] = "0"; //change "*" to "0"
                    derivat.erase(derivat.begin() + x_prime_low, derivat.end()); //erase x' and y
                }
                else if (derivat[x_prime_low] == "1") //* 1 y -> y
                {
                    //                puts("hi 237");
                    derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
                }
                else if (derivat[y_low] == "1") //* x' 1 -> x'
                {
                    //                puts("hi 242");
                    derivat.pop_back(); //remove "1"
                    assert(derivat[x_prime_low - 1] == "*");
                    derivat.erase(derivat.begin() + x_prime_low - 1); //remove "*"
                }
            }
            if (derivat[x_low - 1] == "0") //+ 0 * x' y -> * x' y
            {
                //            puts("hi 249");
                derivat.erase(derivat.begin() + x_low - 2, derivat.begin() + x_low); //remove "+" and "0"
            }
            else if (derivat[x_prime_low - 1] == "0") //+ * x y' 0 -> * x y'
            {
                //            puts("hi 254");
                assert(static_cast<int>(derivat.size()) == x_prime_low);
                derivat.erase(derivat.begin() + x_low - 2); //erase "+"
                derivat.pop_back(); //remove "0"
            }
        }

        else if (prefix[low] == "/")
        {
            int div_idx = derivat.size();
            derivat.push_back("/"); /* / */
            derivat.push_back("-"); /* / - */
            derivat.push_back("*"); /* / - * */
            int temp = low+1+grasp[low+1];
            int x_prime_low = derivat.size();
            int k;
            derivePrefixHelper(low+1, temp, dx, prefix, grasp, true); /* / - * x' */
            if (derivat[x_prime_low] == "0") //* 0 y -> 0
            {
                //            puts("hi 297");
                derivat[x_prime_low - 1] = "0"; //change "*" to "0"
                assert(x_prime_low + 1 == static_cast<int>(derivat.size()));
                derivat.pop_back(); //remove x', which is 0
            }
            else
            {
                int y_low = derivat.size();
                for (k = temp+1; k <= temp+1+grasp[temp+1]; k++) /* / - * x' y */
                {
                    derivat.push_back(prefix[k]);
                }
                if (derivat[y_low] == "0") //* x' 0 -> 0
                {
                    //                puts("hi 312");
                    derivat[x_prime_low - 1] = "0"; //change "*" to "0"
                    derivat.erase(derivat.begin() + x_prime_low, derivat.end()); //remove x' and 0
                }
                else if (derivat[y_low] == "1") //* x' 1 -> x'
                {
                    //                puts("hi 318");
                    assert(y_low == static_cast<int>(derivat.size()) - 1);
                    derivat.erase(derivat.begin() + x_prime_low - 1); //erase "*"
                    derivat.pop_back(); //erase the "1"
                }
                else if (derivat[x_prime_low] == "1") //* 1 y -> y
                {
                    //                    puts("hi 326");
                    derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
                }
            }
            derivat.push_back("*"); /* / - * x' y * */
            int x_low = derivat.size();
            for (k = low+1; k <= temp; k++) /* / - * x' y * x */
            {
                derivat.push_back(prefix[k]);
            }
            if (derivat[x_low] == "0") //* 0 y' -> 0
            {
                //            puts("hi 338");
                derivat.erase(derivat.begin() + x_low - 1); //erase "*"
            }
            else
            {
                int y_prime_low = derivat.size();
                derivePrefixHelper(temp+1, temp+1+grasp[temp+1], dx, prefix, grasp, true); /* / - * x' y * x y' */
                if (derivat[y_prime_low] == "0") //* x 0 -> 0
                {
                    //                puts("hi 347");
                    assert(y_prime_low == static_cast<int>(derivat.size()) - 1);
                    derivat.erase(derivat.begin() + x_low - 1, derivat.begin() + y_prime_low); //erase * and x
                }
                else if (derivat[x_low] == "1") //* 1 y' -> y'
                {
                    //                puts("hi 352");
                    derivat.erase(derivat.begin() + x_low - 1, derivat.begin() + y_prime_low); //erase * and 1
                }
                else if (derivat[y_prime_low] == "1") //* x 1 -> x
                {
                    //                puts("hi 357");
                    assert(y_prime_low == static_cast<int>(derivat.size()) - 1);
                    derivat.erase(derivat.begin() + x_low - 1); //erase "*"
                    derivat.pop_back(); //remove the "1"
                }
            }

            if (((k = (x_low - x_prime_low)) == (static_cast<int>(derivat.size()) - (x_low - 1))) && (areDerivatRangesEqual(x_prime_low - 1, x_low - 1, k))) //- thing1 thing1 -> 0
            {
                //            puts("hi 367");
                derivat[div_idx] = "0";
                derivat.erase(derivat.begin() + div_idx + 1, derivat.end()); //erase everything else
            }
            else
            {
                if (derivat[x_prime_low - 1] == "0") //- 0 * x y' -> ~ * x y'
                {
                    //                puts("hi 375");
                    derivat[x_prime_low - 2] = "~"; //change "-" to "~"
                    derivat.erase(derivat.begin() + x_prime_low - 1); //erase "0"
                }
                else if (derivat[x_low - 1] == "0") //- * x' y 0 -> * x' y
                {
                    //                    puts("hi 381");
                    assert(static_cast<int>(derivat.size()) == x_low);
                    derivat.erase(derivat.begin() + x_prime_low - 2); //erase the "-"
                    derivat.pop_back(); //erase the "0"
                }
                derivat.push_back("*"); /* / - * x' y * x y' * */
                int y_low = derivat.size();
                for (k = temp+1; k <= temp+1+grasp[temp+1]; k++) /* / - * x' y * x y' * y */
                {
                    derivat.push_back(prefix[k]);
                }
                if (derivat[y_low] == "1") // / - * x' y * x y' * 1 1 ->  - * x' y * x y'
                {
                    //                puts("hi 381");
                    assert(y_low == static_cast<int>(derivat.size()) - 1);
                    derivat.erase(derivat.begin() + y_low - 1); //erase "*"
                    derivat.erase(derivat.begin() + div_idx); //erase "/"
                    derivat.pop_back(); //erase "1"
                }
                else
                {
                    for (k = temp+1; k <= temp+1+grasp[temp+1]; k++) /* / - * x' y * x y' * y y */
                    {
                        derivat.push_back(prefix[k]);
                    }
                }
            }
        }

        else if (prefix[low] == "^")
        {
            derivat.push_back("*"); /* * */
            derivat.push_back("^"); /* * ^ */
            int temp = low+1+grasp[low+1];
            int k;
            int x_low = derivat.size();
            for (k = low+1; k <= temp; k++) /* * ^ x */
            {
                derivat.push_back(prefix[k]);
            }
            if (derivat[x_low] == "0") //* ^ 0 y (* ln 0 y)' -> 0 (maybe problematic for y < 0, but oh well 😮‍💨)
            {
                //            puts("hi 454");
                assert(x_low == static_cast<int>(derivat.size()) - 1);
                derivat.erase(derivat.begin() + x_low - 2, derivat.begin() + x_low); //erase "*" and "^"
                return;
            }
            else if (derivat[x_low] == "1") //* ^ 1 y (* ln 1 y)' -> 0 (because ln(1) is 0)
            {
                //            puts("hi 461");
                assert(x_low == static_cast<int>(derivat.size()) - 1);
                derivat[x_low] = "0"; //change "1" to "0"
                derivat.erase(derivat.begin() + x_low - 2, derivat.begin() + x_low); //erase "*" and "^"
                return;
            }
            int y_low = derivat.size();
            for (k = temp+1; k <= temp+1+grasp[temp+1]; k++) /* * ^ x y */
            {
                derivat.push_back(prefix[k]);
            }
            if (derivat[y_low] == "0") //* ^ x 0 (* ln x 0)' -> 0
            {
                assert(y_low == static_cast<int>(derivat.size()) - 1);
                //            puts("hi 474");
                derivat[x_low - 2] = "0"; //change "*" to "0)
                derivat.erase(derivat.begin() + x_low - 1, derivat.end()); //erase the rest
                return;
            }
            else if (derivat[y_low] == "1") //^ x 1 -> x
            {
                assert(y_low == static_cast<int>(derivat.size()) - 1);
                derivat.pop_back(); //erase the "1"
                derivat.erase(derivat.begin() + x_low - 1); //erase the "*"
                //            puts("hi 485");
            }
            std::vector<std::string> prefix_temp;
            std::vector<int> grasp_temp;
            size_t reserve_amount = up+2-low; //up-low -> x and y, 2 -> ln and *, => up+2-low -> * ln x y
            prefix_temp.reserve(reserve_amount);
            grasp_temp.reserve(reserve_amount);
            prefix_temp.push_back("*"); /* * */
            prefix_temp.push_back("ln"); /* * ln */
            int x_temp_low = prefix_temp.size();
            for (k = low+1; k <= temp; k++) /* * ln x */
            {
                prefix_temp.push_back(prefix[k]);
            }
            y_low = prefix_temp.size();
            for (k = temp+1; k <= temp+1+grasp[temp+1]; k++) /* * ln x y */
            {
                prefix_temp.push_back(prefix[k]);
            }
            if (prefix_temp[y_low] == "1") //* ln x 1 -> ln x
            {
                //            puts("hi 506");
                assert(y_low == static_cast<int>(prefix_temp.size()) - 1);
                prefix_temp.pop_back(); //remove the "1"
                prefix_temp.erase(prefix_temp.begin() + x_temp_low - 2); //erase the "*"
            }
            setPrefixGR(prefix_temp, grasp_temp);
            int temp_term_low = derivat.size();
            //        derivat.push_back("1");
            derivePrefixHelper(0, prefix_temp.size() - 1, dx, prefix_temp, grasp_temp, true); /* * ^ x y (* ln x y)' */
            if (derivat[temp_term_low] == "0") //* ^ x y 0 -> 0
            {
                //            puts("hi 516");
                derivat[x_low - 2] = "0"; //changing "*" to "0"
                derivat.erase(derivat.begin() + x_low - 1, derivat.end()); //erase the rest
            }
            else if (derivat[temp_term_low] == "1") //* ^ x y 1 -> ^ x y
            {
                //            puts("hi 522");
                assert(temp_term_low == static_cast<int>(derivat.size()) - 1);
                derivat.erase(derivat.begin() + x_low - 2); //erasing "*"
                derivat.pop_back(); //erasing the "1"
            }
        }

        else if (prefix[low] == "cos")
        {
            derivat.push_back("*"); /* * */
            int x_prime_low = derivat.size();
            int temp = low+1;
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* * x' */
            if (derivat[x_prime_low] == "0") //* 0 ~ sin x -> 0
            {
                //                puts("hi 538");
                assert(static_cast<int>(derivat.size() - 1) == x_prime_low);
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase "*"
                return;
            }
            derivat.push_back("~"); /* * x' ~ */
            derivat.push_back("sin"); /* * x' ~ sin */
            for (int k = temp; k <= temp+grasp[temp]; k++)
            {
                derivat.push_back(prefix[k]); /* * x' ~ sin x */
            }
            if (derivat[x_prime_low] == "1") //* 1 ~ sin x -> ~ sin x
            {
                //                puts("hi 551");
                derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
            }
        }

        else if (prefix[low] == "sin")
        {
            derivat.push_back("*"); /* * */
            int x_prime_low = derivat.size();
            int temp = low+1;
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* * x' */
            if (derivat[x_prime_low] == "0") //* 0 cos x -> 0
            {
                //                puts("hi 565");
                assert(static_cast<int>(derivat.size() - 1) == x_prime_low);
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase "*"
                return;
            }
            derivat.push_back("cos"); /* * x' cos */
            for (int k = temp; k <= temp+grasp[temp]; k++)
            {
                derivat.push_back(prefix[k]); /* * x' cos x */
            }
            if (derivat[x_prime_low] == "1") //* 1 cos x -> cos x
            {
                //                puts("hi 577");
                derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
            }
        }

        else if (prefix[low] == "sqrt")
        {
            derivat.push_back("/");         /* / */
            int temp = low+1;
            int x_prime_low = derivat.size();
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* / x' */
            if (derivat[x_prime_low] == "0")
            {
                //            puts("hi 590");
                assert(x_prime_low == static_cast<int>(derivat.size() - 1));
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase the "/"
                return;
            }
            derivat.push_back("*");         /* / x' * */
            derivat.push_back("2");         /* / x' * 2 */
            derivat.push_back("sqrt");      /* / x' * 2 sqrt */
            for (int k = temp; k <= temp+grasp[temp]; k++) /* / x' * 2 sqrt x */
            {
                derivat.push_back(prefix[k]);
            }
        }

        else if (prefix[low] == "log" || prefix[low] == "ln")
        {
            derivat.push_back("/");               /* / */
            int temp = low+1;
            int x_prime_low = derivat.size();
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* / x' */
            if (derivat[x_prime_low] == "0") // / 0 x -> 0
            {
                //                puts("hi 578");
                assert(static_cast<int>(derivat.size()) - 1 == x_prime_low);
                derivat[x_prime_low - 1] = "0"; //change "/" to 0
                derivat.erase(derivat.begin() + x_prime_low, derivat.end()); //delete the rest
                return;
            }
            int x_low = derivat.size();
            for (int k = temp; k <= temp+grasp[temp]; k++)
            {
                derivat.push_back(prefix[k]);      /* / x' x */
            }
            int step = derivat.size() - x_low;
            if ((step == (x_low - x_prime_low)) && areDerivatRangesEqual(x_prime_low, x_low, step)) // / something something -> 1
            {
                //                puts("hi 591");
                derivat[x_prime_low - 1] = "1"; //change "/" to 0
                derivat.erase(derivat.begin() + x_prime_low, derivat.end()); //delete the rest
            }
        }

        else if (prefix[low] == "asin" || prefix[low] == "arcsin")
        {
            derivat.push_back("/");   /* / */
            int temp = low+1;
            int x_prime_low = derivat.size();
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* / x' */
            if (derivat[x_prime_low] == "0")
            {
                //                puts("hi 640");
                assert(x_prime_low == static_cast<int>(derivat.size()) - 1);
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase "/"
                return;
            }
            derivat.push_back("sqrt"); /* / x' sqrt */
            derivat.push_back("-");    /* / x' sqrt - */
            derivat.push_back("1");    /* / x' sqrt - 1 */
            derivat.push_back("*");    /* / x' sqrt - 1 * */
            for (int k = temp; k <= temp+grasp[temp]; k++) /* / x' sqrt - 1 * x */
            {
                derivat.push_back(prefix[k]);
            }
            for (int k = temp; k <= temp+grasp[temp]; k++) /* / x' sqrt - 1 * x x */
            {
                derivat.push_back(prefix[k]);
            }
        }

        else if (prefix[low] == "acos" || prefix[low] == "arccos")
        {
            derivat.push_back("~");   /* ~ */
            derivat.push_back("/");   /* ~ / */
            int temp = low+1;
            int x_prime_low = derivat.size();
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* ~ / x' */
            if (derivat[x_prime_low] == "0")
            {
                //            puts("hi 668");
                assert(x_prime_low == static_cast<int>(derivat.size()) - 1);
                derivat.erase(derivat.begin() + x_prime_low - 2, derivat.begin() + x_prime_low); //erase "~" and "/"
                return;
            }
            derivat.push_back("sqrt"); /* ~ / x' sqrt */
            derivat.push_back("-");    /* ~ / x' sqrt - */
            derivat.push_back("1");    /* ~ / x' sqrt - 1 */
            derivat.push_back("*");    /* ~ / x' sqrt - 1 * */
            for (int k = temp; k <= temp+grasp[temp]; k++) /* ~ / x' sqrt - 1 * x */
            {
                derivat.push_back(prefix[k]);
            }
            for (int k = temp; k <= temp+grasp[temp]; k++) /* ~ / x' sqrt - 1 * x x */
            {
                derivat.push_back(prefix[k]);
            }
        }

        else if (prefix[low] == "tanh")
        {
            derivat.push_back("*");      //*
            int x_prime_low = derivat.size();
            int temp = low+1;
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); //* x'
            if (derivat[x_prime_low] == "0")
            {
                //                puts("hi 696");
                assert(x_prime_low == static_cast<int>(derivat.size()) - 1);
                derivat.erase(derivat.begin() + x_prime_low - 1); //delete the "*"
                return;
            }
            derivat.push_back("*");      //* x' *
            derivat.push_back("sech");   //* x' * sech
            for (int k = temp; k <= temp+grasp[temp]; k++) //* x' * sech x
            {
                derivat.push_back(prefix[k]);
            }
            derivat.push_back("sech");   //* x' * sech x sech
            for (int k = temp; k <= temp+grasp[temp]; k++) //* x' * sech x sech x
            {
                derivat.push_back(prefix[k]);
            }
            if (derivat[x_prime_low] == "1") //* 1 * sech x sech x -> * sech x sech x
            {
                //                puts("hi 715");
                derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
            }
        }

        else if (prefix[low] == "sech")
        {
            derivat.push_back("*"); //*
            int x_prime_low = derivat.size();
            int temp = low+1;
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); //* x'
            if (derivat[x_prime_low] == "0") //* 0 * ~ sech x tanh x -> 0
            {
                //                puts("hi 722");
                assert(x_prime_low == static_cast<int>(derivat.size()) - 1);
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase the "*"
                return;
            }
            derivat.push_back("*");      //* x' *
            derivat.push_back("~");      //* x' * ~
            derivat.push_back("sech");   //* x' * ~ sech
            for (int k = temp; k <= temp+grasp[temp]; k++) //* x' * ~ sech x
            {
                derivat.push_back(prefix[k]);
            }
            derivat.push_back("tanh");   //* x' * ~ sech x tanh
            for (int k = temp; k <= temp+grasp[temp]; k++) //* x' * ~ sech x tanh x
            {
                derivat.push_back(prefix[k]);
            }
            if (derivat[x_prime_low] == "1") //* 1 exp x -> exp x
            {
                //                puts("hi 742");
                derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
            }
        }

        else if (prefix[low] == "exp")
        {
            derivat.push_back("*");               //*
            int temp = low+1;
            int x_prime_low = derivat.size();
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); //* x'
            if (derivat[x_prime_low] == "0") //* 0 exp x -> 0
            {
                //            puts("hi 682");
                assert(static_cast<int>(derivat.size() - 1) == x_prime_low);
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase "*"
                return;
            }
            derivat.push_back("exp");           //* x' exp
            for (int k = temp; k <= temp+grasp[temp]; k++)
            {
                derivat.push_back(prefix[k]);      //* x' exp x
            }
            if (derivat[x_prime_low] == "1") //* 1 exp x -> exp x
            {
                //                puts("hi 694");
                derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
            }
        }

        else if (prefix[low] == "~")
        {
            int temp = low+1;
            int un_minus_idx = derivat.size();
            derivat.push_back(prefix[low]); /* ~ */
            int x_prime_low = derivat.size();
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* ~ x' */
            if (derivat[x_prime_low] == "~")
            {
                //                puts("hi 590");
                derivat.erase(derivat.begin() + un_minus_idx, derivat.begin() + x_prime_low + 1); //erase the two "~"
            }
        }

        else
        {
            if (prefix[low] == dx)
            {
                this->derivat.push_back("1");
            }
            else
            {
                this->derivat.push_back("0");
            }
        }
    }

    void derivePrefix(int low, int up, const std::string& dx, const std::vector<std::string>& prefix, std::vector<int>& grasp)
    {
        derivePrefixHelper(low, up, dx, prefix, grasp, false);
    }

    void setPostfixGR(const std::vector<std::string>& postfix, std::vector<int>& grasp)
    {
        grasp.reserve(postfix.size()); //grasp[k] = GR( postfix[k]), k = 1, ... ,i.
        //In the paper they do `k = 1;` instead of `k = 0;`, presumably because GR(postfix[0]) always is 0, but it works
        //if you set k = 0 too.
        for (size_t k = 0; k < postfix.size(); ++k)
        {
            grasp.push_back(GR(k, postfix));
        }
    }

    /*
     low and up: lower and upper Index bounds, respectively, for the piece of the array postfix which is to be the subject of the processing.
     dx: string representing the variable by which the derivation is to be made. (The derivative is made wrt dx)
     */
    void derivePostfixHelper(int low, int up, const std::string& dx, const std::vector<std::string>& postfix, std::vector<int>& grasp, bool setGRvar = false)
    {
        if (!setGRvar)
        {
            grasp.clear();
            this->derivat.clear();
            // std::cout << this->derivat.size();
            this->derivat.reserve(1000);
            //        Index = 0;
            setPostfixGR(postfix, grasp);
        }
        //allowed ops: +, -, *, /, ^, unary +, unary -, sin(), cos(), tan(), ctg(), log(), sqrt(), const, x0, x1, ..., x_numFeatures
        //Define `grasp` of postfix[i], i.e., the number of elements forming operands of postfix[i] (grasp(operand) = 0)
        //The grasped elements of postfix[i] are the elements forming operands of postfix[i]
        //The left-grasp-bound (LGB) of postfix[i] is the Index of the left-most grasped element of postfix[i] in the array postfix
        //For the expression formed by postfix[i] and its grasped elements, the element postfix[i] is termed the `head` or `main element`
        //ANY element postfix[i] is an N-arity operator acting on operands arg1, ..., argN, the heads of which are op1, ..., opN, where op1 is the left-neighest operator of postfix[i] in the array postfix (so operands are 0-arity operators)
        //For an N-arity operator postfix[i]:
        //The `grasp` of postfix[i]  is equal to i - LGB(postfix[i])
        //grasp(postfix[i]) = N + sum(grasp(op1), ..., grasp(op_k), ..., grasp(opN))
        //grasp(postfix[i]) = N + sum( (1 - grasp(op1)), ..., (k - grasp(op_k)), ..., (N - grasp(opN)))
        //LGB(postfix[i]) = i - N - sum( (1 - grasp(op1)), ..., (k - grasp(op_k)), ..., (N - grasp(opN)))
        //op_(N-j) = postfix[i - sum(grasp(op_(N-1)), ..., grasp(op_(N-j-1))) - j - 1], where j = 0, ..., N-1
        //If the grasp of an arbitrary postfix[i] is greater than N, then at least one of its argument heads is also an operator.
        //Example: If the grasp of any binary operator postfix[i] is greater than 2, then at least one of the two preceding elements in the RPN of the expression (postfix[i-1] and postfix[i-2]) is also an operator (unary or binary).
        //postfix[numElements] is certainly an operator (unary or binary)

        //if postfix[up] is a binary operator, then:
        //the head of its second argument (let's call it op2) is equal to postfix[up-1]
        //then the grasped elements of op2 are the elements from postfix[up-1-grasp[up-1]] to postfix[up-1]
        //e.g. postfix = {"x", "x", "*", "x", "cos", "x", "*", "+"}, up = 7 -> postfix[up] = "+" is binary
        //so postfix[up-1] = "*" is the head of the second argument of "+" and so the grasped elements
        //of postfix[up-1] are the elements [(postfix[up-1-grasp[up-1]] = postfix[6-3] = postfix[3]), postfix[up-1] = postfix[6]]
        //i.e., the elements {"x", "cos", "x", "*"}
        //the head of its first argument (lets call it op1) is equal to postfix[up-grasp(op2)-2] which is equal to postfix[up-2-grasp[up-1]].
        //then the grasped elements of op1 are the elements from postfix[low = 0] to postfix[up-2-grasp[up-1]]
        //e.g. postfix = {"x", "x", "*", "x", "cos", "x", "x", "*", "*", "+"}, up = 9 ->postfix[up] = "+" is binary
        //so postfix[up-grasp(op2)-2] = postfix[9-5-2] = postfix[2] = "*" is the head of the first argument of "+" and so the grasped elements
        //of postfix[up-grasp(op2)-2] are the elements [(postfix[low] = postfix[0], postfix[up-grasp(op2)-2] = postfix[9-5-2] = postfix[2]]
        //i.e., the elements {"x", "x", "*"}

        if (std::find(postfix.begin(), postfix.end(), dx) == postfix.end())
        {
            this->derivat.push_back("0");
            return;
        }

        if (postfix[up] == "+" || postfix[up] == "-")
        {
            int x_prime_low = derivat.size();
            derivePostfixHelper(low, up-2-grasp[up-1], dx, postfix, grasp, true);  /*Putting x'*/
            int x_prime_high = derivat.size();
            derivePostfixHelper(up-1-grasp[up-1], up-1, dx, postfix, grasp, true); /*Putting y'*/
            int y_prime_high = derivat.size();
            int step;

            /*
             Simplification cases:

             1.) y' == 0, +/-, x'
             2.) x' == 0,   +, y'
             3.) x' == 0,   -, y' ~

             */

            if (derivat.back() == "0") //1.) x' 0 + -> x'
            {
                //            puts("hi 145");
                derivat.pop_back();
            }

            else if (derivat[x_prime_high - 1] == "0")
            {
                //            puts("hi 151");
                //erase elements from derivat[x_prime_low] to derivat[x_prime_high-1] inclusive
                derivat.erase(derivat.begin() + x_prime_low, derivat.begin() + x_prime_high); //0 y + -> y
                if (postfix[up] == "-") //3.)
                {
                    //                puts("hi 156");
                    derivat.push_back("~"); //0 y - -> y ~
                }
            }

            else if ((postfix[up] == "-") && ((step = (x_prime_high - x_prime_low)) == (y_prime_high - x_prime_high)) && (areDerivatRangesEqual(x_prime_low, x_prime_high, step)))
            {
                //                puts("hi 180");
                derivat[x_prime_low] = "0"; //change first symbol of x' to 0
                derivat.erase(derivat.begin() + x_prime_low + 1, derivat.begin() + y_prime_high); //erase the rest of x' and y'
            }

            else
            {
                derivat.push_back(postfix[up]);
            }
        }
        else if (postfix[up] == "*")
        {
            int x_low = derivat.size();
            for (int k = low; k <= up-2-grasp[up-1]; k++) /* x */
            {
                derivat.push_back(postfix[k]);
            }
            if (derivat.back() == "0") //0 y' * -> 0
            {
                //            puts("hi 176");
            }
            else
            {
                int x_high = derivat.size();
                derivePostfixHelper(up-1-grasp[up-1], up-1, dx, postfix, grasp, true); /* x y' */
                if (derivat.back() == "0") //x 0 * -> 0
                {
                    //                puts("hi 184");
                    derivat[x_low] = "0"; //change first symbol of x to 0
                    derivat.erase(derivat.begin() + x_low + 1, derivat.end()); //erase rest of x and y'
                }
                else if (derivat[x_high - 1] == "1") //1 y' * -> y'
                {
                    //                puts("hi 190");
                    assert(x_low == x_high - 1);
                    derivat.erase(derivat.begin() + x_low); //erase the x since it's 1
                }
                else if (derivat.back() == "1") //x 1 * -> x
                {
                    //                puts("hi 196");
                    derivat.pop_back(); //remove the y' since it's 1
                }
                else
                {
                    derivat.push_back("*"); /* x y' "*" */
                }
            }

            int x_prime_low = derivat.size();
            derivePostfixHelper(low, up-2-grasp[up-1], dx, postfix, grasp, true); /* x y' "*" x' */
            if (derivat.back() == "0") //0 y * -> 0
            {
                //            puts("hi 209");
            }
            else
            {
                int y_low = derivat.size();
                for (int k = up-1-grasp[up-1]; k <= up - 1; k++)
                {
                    derivat.push_back(postfix[k]); /* x y' "*" x' y */
                }
                if (derivat.back() == "0") //x' 0 * -> 0
                {
                    //                puts("hi 220");
                    derivat.erase(derivat.begin() + x_prime_low, derivat.begin() + y_low); //erase x'
                }
                else if (derivat[y_low - 1] == "1") //1 y * -> y
                {
                    //                puts("hi 225");
                    assert(y_low - 1 == x_prime_low);
                    derivat.erase(derivat.begin() + x_prime_low); //remove the 1
                }
                else if (derivat.back() == "1") //x' 1 * -> x'
                {
                    //                puts("hi 231");
                    derivat.pop_back(); //remove the "1"
                }
                else
                {
                    derivat.push_back("*"); /* x y' "*" x' y "*" */
                }
            }
            if (derivat[x_prime_low - 1] == "0") // 0 x' y "*" + -> x' y "*"
            {
                //                puts("hi 236");
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase 0
            }
            else if (derivat.back() == "0") //x y' "*" 0 + -> x y' "*"
            {
                //                puts("hi 241");
                derivat.pop_back();
            }
            else
            {
                derivat.push_back("+"); /* x y' "*" x' y "*" + */
            }
        }

        else if (postfix[up] == "/")
        {
            int x_prime_low = derivat.size();
            derivePostfixHelper(low, up-2-grasp[up-1], dx, postfix, grasp, true); /* x' */
            int k;
            if (derivat.back() == "0") //0 y * -> 0
            {
                //            puts("hi 286");
            }
            else
            {
                int y_low = derivat.size();
                for (k = up-1-grasp[up-1]; k <= up-1; k++) /* x' y */
                {
                    derivat.push_back(postfix[k]);
                }
                if (derivat.back() == "0") //x' 0 * -> 0
                {
                    //                puts("hi 297");
                    derivat.erase(derivat.begin() + x_prime_low, derivat.end() - 1); //erase x'
                }
                else if (derivat.back() == "1") //x' 1 * -> x'
                {
                    //                puts("hi 302");
                    derivat.pop_back(); //remove the "1"
                }
                else if (derivat[y_low-1] == "1") //1 y * -> y
                {
                    //                puts("hi 307");
                    derivat.erase(derivat.begin() + y_low - 1); //erase the "1"
                }
                else
                {
                    derivat.push_back("*"); /* x' y *  */
                }
            }
            int x_low = derivat.size();
            for (k = low; k <= up-2-grasp[up-1]; k++) /* x' y * x */
            {
                derivat.push_back(postfix[k]);
            }
            if (derivat.back() == "0") //0 y' * -> 0
            {
                //            puts("hi 322");
            }
            else
            {
                int y_prime_low = derivat.size();
                derivePostfixHelper(up-1-grasp[up-1], up-1, dx, postfix, grasp, true); /* x' y * x y' */
                if (derivat.back() == "0") //x 0 * -> 0
                {
                    //                puts("hi 330");
                    derivat.erase(derivat.begin() + x_low, derivat.begin() + y_prime_low); //erase x
                }
                else if (derivat.back() == "1") //x 1 * -> x
                {
                    //                puts("hi 335");
                    derivat.pop_back(); //erase the 1
                }
                else if (derivat[y_prime_low - 1] == "1") //1 y' * -> y'
                {
                    //                puts("hi 340");
                    derivat.erase(derivat.begin() + y_prime_low - 1); //erase the "1"
                }
                else
                {
                    derivat.push_back("*"); /* x' y * x y' * */
                }
            }
            if (((k = (x_low - x_prime_low)) == (static_cast<int>(derivat.size()) - x_low)) && (areDerivatRangesEqual(x_prime_low, x_low, k))) //thing1 thing1 - -> 0
            {
                //            puts("hi 350");
                derivat[x_prime_low] = "0"; //change first symbol of x' to 0
                derivat.erase(derivat.begin() + x_prime_low + 1, derivat.end()); //erase the rest of x' y * and x y' *
            }
            else
            {
                if (derivat[x_low - 1] == "0") //0 x y' * - -> x y' * ~
                {
                    //                puts("hi 358");
                    derivat.erase(derivat.begin() + x_low - 1); //remove "0"
                    derivat.push_back("~"); //add "~" at the end
                }
                else if (derivat.back() == "0") //x' y * 0 - -> x' y *
                {
                    //                puts("hi 364");
                    derivat.pop_back(); //remove "0"
                }
                else
                {
                    derivat.push_back("-"); /* x' y * x y' * - */
                }
                for (k = up-1-grasp[up-1]; k <= up-1; k++)      /* x' y * x y' * - y */
                {
                    derivat.push_back(postfix[k]);
                }
                if (derivat.back() == "1") //"1 1 * /" -> ""
                {
                    //                puts("hi 377");
                    derivat.pop_back(); //remove the "1"
                }
                else
                {
                    for (k = up-1-grasp[up-1]; k <= up-1; k++)      /* x' y * x y' * - y y */
                    {
                        derivat.push_back(postfix[k]);
                    }
                    derivat.push_back("*"); /* x' y * x y' * - y y * */
                    derivat.push_back("/"); /* x' y * x y' * - y y * / */
                }
            }
        }

        else if (postfix[up] == "^")
        {
            int k;
            int x_low = derivat.size();
            for (k = low; k <= up-2-grasp[up-1]; k++) /* x */
            {
                derivat.push_back(postfix[k]);
            }
            if (derivat.back() == "0") //0 y ^ (0 ln y *)' * -> 0 (maybe problematic for y < 0, but oh well 😮‍💨)
            {
                //            puts("hi 402");
                return;
            }
            else if (derivat.back() == "1") //1 y ^ (1 ln y *)' * -> 0 (because ln(1) is 0)
            {
                derivat.back() = "0";
                //            puts("hi 407");
                return;
            }
            else
            {
                for (k = up-1-grasp[up-1]; k <= up-1; k++) /* x y */
                {
                    derivat.push_back(postfix[k]);
                }
                if (derivat.back() == "0") //x 0 ^ (x ln 0 *)' * -> 0
                {
                    //                puts("hi 419");
                    derivat[x_low] = "0"; //change the first symbol of x to "0"
                    derivat.erase(derivat.begin() + x_low + 1, derivat.end()); //erase the rest
                    return;
                }
                else if (derivat.back() == "1") //x 1 ^ -> x
                {
                    //                    puts("hi 426");
                    derivat.pop_back(); //erase the 1
                }
                else
                {
                    derivat.push_back("^"); /* x y ^ */
                }
            }

            std::vector<std::string> postfix_temp;
            std::vector<int> grasp_temp;
            size_t reserve_amount = up+2-low; //up-low -> x and y, 2 -> ln and *, => up+2-low -> x ln y *
            postfix_temp.reserve(reserve_amount);
            grasp_temp.reserve(reserve_amount);
            for (k = low; k <= up-2-grasp[up-1]; k++) /* x */
            {
                postfix_temp.push_back(postfix[k]);
            }
            postfix_temp.push_back("ln"); /* x ln  */
            for (k = up-1-grasp[up-1]; k <= up-1; k++) /* x ln y */
            {
                postfix_temp.push_back(postfix[k]);
            }
            if (postfix_temp.back() == "1") //x ln 1 * -> x ln
            {
                //            puts("hi 452");
                postfix_temp.pop_back();
            }
            else
            {
                postfix_temp.push_back("*"); /* x ln y * */
            }
            setPostfixGR(postfix_temp, grasp_temp);
            derivePostfixHelper(0, postfix_temp.size() - 1, dx, postfix_temp, grasp_temp, true); /* x y ^ (x ln y *)' */
            if (derivat.back() == "0") //x y ^ 0 * -> 0
            {
                //            puts("hi 455");
                derivat[x_low] = "0"; //change the first symbol of x to "0"
                derivat.erase(derivat.begin() + x_low + 1, derivat.end()); //erase the rest
            }
            else if (derivat.back() == "1") //x y ^ 1 * -> x y ^
            {
                //            puts("hi 460");
                derivat.pop_back(); //erase (x ln y *)'
            }
            else
            {
                derivat.push_back("*"); /* x y ^ (x ln y *)' * */
            }
        }

        else if (postfix[up] == "cos")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == "0") //0 x sin ~ * -> 0
            {
                //            puts("hi 514");
                return;
            }
            int x_low = derivat.size();
            for (int k = low; k <= up-1; k++)
            {
                derivat.push_back(postfix[k]); /* x' x */
            }
            derivat.push_back("sin"); /* x' x sin */
            derivat.push_back("~"); /* x' x sin ~ */
            if (derivat[x_low - 1] == "1") //1 x sin ~ * -> x sin ~
            {
                //            puts("hi 526");
                derivat.erase(derivat.begin() + x_low - 1); //erase the "1"
            }
            else
            {
                derivat.push_back("*"); /* x' x sin ~ * */
            }
        }

        else if (postfix[up] == "sin")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == "0") //0 x cos * -> 0
            {
                //                puts("hi 540");
                return;
            }
            int x_low = derivat.size();
            for (int k = low; k <= up-1; k++)
            {
                derivat.push_back(postfix[k]); /* x' x */
            }
            derivat.push_back("cos"); /* x' x cos */
            if (derivat[x_low - 1] == "1") //1 x cos * -> x cos
            {
                //                puts("hi 551");
                derivat.erase(derivat.begin() + x_low - 1); //erase the "1"
            }
            else
            {
                derivat.push_back("*"); /* x' x cos * */
            }
        }

        else if (postfix[up] == "sqrt")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == "0") //0 2 x sqrt * / -> 0
            {
                //                puts("hi 565");
                return;
            }
            derivat.push_back("2"); /* x' 2 */
            for (int k = low; k <= up-1; k++) /* x' 2 x */
            {
                derivat.push_back(postfix[k]);
            }
            derivat.push_back("sqrt");    /* x' 2 x sqrt */
            derivat.push_back("*");       /* x' 2 x sqrt * */
            derivat.push_back("/");       /* x' 2 x sqrt * / */
        }

        else if (postfix[up] == "log" || postfix[up] == "ln")
        {
            int x_prime_low = derivat.size();
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == "0") //0 x / -> 0
            {
                //            puts("hi 551");
                assert(x_prime_low == static_cast<int>(derivat.size()) - 1);
                return;
            }
            int x_low = derivat.size();
            for (int k = low; k <= up-1; k++)
            {
                derivat.push_back(postfix[k]);      /* x' x */
            }
            int step = derivat.size() - x_low;
            if ((step == (x_low - x_prime_low)) && areDerivatRangesEqual(x_prime_low, x_low, step)) //something something / -> 1
            {
                //                puts("hi 563");
                derivat[x_prime_low] = "1"; //replace first symbol of x' with "1"
                derivat.erase(derivat.begin() + x_prime_low + 1, derivat.end()); //erase the rest
                return;
            }

            derivat.push_back("/");               /* x' x / */
        }

        else if (postfix[up] == "asin" || postfix[up] == "arcsin")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == "0") //0 1 x x * - sqrt / -> 0
            {
                //                puts("hi 610");
                return;
            }
            derivat.push_back("1"); /* x' 1 */
            for (int k = low; k <= up-1; k++) /* x' 1 x */
            {
                derivat.push_back(postfix[k]);
            }
            for (int k = low; k <= up-1; k++) /* x' 1 x x */
            {
                derivat.push_back(postfix[k]);
            }
            derivat.push_back("*");   /* x' 1 x x * */
            derivat.push_back("-");   /* x' 1 x x * - */
            derivat.push_back("sqrt");   /* x' 1 x x * - sqrt */
            derivat.push_back("/");   /* x' 1 x x * - sqrt / */
        }

        else if (postfix[up] == "acos" || postfix[up] == "arccos")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == "0") //0 1 x x * - sqrt / ~ -> 0
            {
                //            puts("hi 633");
                return;
            }
            derivat.push_back("1"); /* x' 1 */
            for (int k = low; k <= up-1; k++) /* x' 1 x */
            {
                derivat.push_back(postfix[k]);
            }
            for (int k = low; k <= up-1; k++) /* x' 1 x x */
            {
                derivat.push_back(postfix[k]);
            }
            derivat.push_back("*");   /* x' 1 x x * */
            derivat.push_back("-");   /* x' 1 x x * - */
            derivat.push_back("sqrt");   /* x' 1 x x * - sqrt */
            derivat.push_back("/");   /* x' 1 x x * - sqrt / */
            derivat.push_back("~");   /* x' 1 x x * - sqrt / ~ */
        }

        else if (postfix[up] == "tanh")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); //x'
            if (derivat.back() == "0") //0 x sech x sech * * -> 0
            {
                //                puts("hi 657");
                return;
            }
            int x_low = derivat.size();
            for (int k = low; k <= up-1; k++) //x' x
            {
                derivat.push_back(postfix[k]);
            }
            derivat.push_back("sech"); //x' x sech
            for (int k = low; k <= up-1; k++) //x' x sech x
            {
                derivat.push_back(postfix[k]);
            }
            derivat.push_back("sech"); //x' x sech x sech
            derivat.push_back("*"); //x' x sech x sech *
            if (derivat[x_low - 1] == "1") //1 x sech x sech * * -> x sech x sech * *
            {
                //                puts("hi 676");
                derivat.erase(derivat.begin() + x_low - 1); //erase the "1"
            }
            else
            {
                derivat.push_back("*");                 //x' x sech ~ x tanh * *
            }
        }

        else if (postfix[up] == "sech")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); //x'
            if (derivat.back() == "0") //0 x sech ~ x tanh * * -> 0
            {
                //                puts("hi 681");
                return;
            }
            int x_low = derivat.size();
            for (int k = low; k <= up-1; k++) //x' x
            {
                derivat.push_back(postfix[k]);
            }
            derivat.push_back("sech");   //x' x sech
            derivat.push_back("~");      //x' x sech ~
            for (int k = low; k <= up-1; k++) //x' x sech ~ x
            {
                derivat.push_back(postfix[k]);
            }
            derivat.push_back("tanh");   //x' x sech ~ x tanh
            derivat.push_back("*");      //x' x sech ~ x tanh *
            if (derivat[x_low - 1] == "1") //1 x sech ~ x tanh * * -> x sech ~ x tanh *
            {
                //                puts("hi 699");
                derivat.erase(derivat.begin() + x_low - 1); //erase the "1"
            }
            else
            {
                derivat.push_back("*");                 //x' x sech ~ x tanh * *
            }
        }

        else if (postfix[up] == "exp")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == "0") //0 x exp * -> 0
            {
                //            puts("hi 649");
                return;
            }
            int x_low = derivat.size();
            for (int k = low; k <= up-1; k++)
            {
                derivat.push_back(postfix[k]);      /* x' x */
            }
            derivat.push_back("exp");               /* x' x exp */
            if (derivat[x_low - 1] == "1") //1 x exp * -> x exp
            {
                //                puts("hi 660");
                derivat.erase(derivat.begin() + x_low - 1); //erase the "1"
            }
            else
            {
                derivat.push_back("*");               /* x' x exp * */
            }
        }

        else if (postfix[up] == "~")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == "~")
            {
                derivat.pop_back(); //two unary minuses cancel each-other
            }
            else
            {
                derivat.push_back(postfix[up]); /* x' ~ */
            }
        }

        else
        {
            if (postfix[up] == dx)
            {
                this->derivat.push_back("1");
            }
            else
            {
                this->derivat.push_back("0");
            }
        }
    }

    void derivePostfix(int low, int up, const std::string& dx, const std::vector<std::string>& postfix, std::vector<int>& grasp)
    {
        derivePostfixHelper(low, up, dx, postfix, grasp, false);
    }
};

/*
 Infix: abs(f - x1)
 Postfix: f x1 - abs
 Prefix: abs - f x1
 */

std::vector<std::vector<std::string>> WierdTrackFitter(Board& x, bool fit)
{
    /*
     '''
     x='((((((((sin(((x0 + sin(sqrt(x0))) * 0.13599420224810638)) * (27.317713 - ((x0 * 0.000007) * (x0 * x0)))) + ((-0.323435 * x0) * sin(sin((1.092289 * sqrt(x0)))))) - (8.034637 * cos((x0 * 0.18450196567500599)))) + arcsin(cos((0.2787780894020867 * (x0 * 0.898195))))) + (5.075383 * sin((x0 * -0.36787944117144233)))) + (7.763419 * ((x0 + (x0 + 1)) * (-0.033530 + sqrt(sech(x0)))))) - ((0.059060 * x0) * sin((x0 * 0.2627831798005332)))) - arcsin(cos(((x0 * 0.5175124998053154) + -0.862172))))'.replace("^","**").replace("~", "-").replace("x0","s"); from sympy import *; import sympy as sp; y = x.replace("arccos","acos").replace("arcsin","asin"); s = sp.symbols("s"); from sympy.printing.pycode import pycode; print('\n',x:=pycode(eval(y)).replace('math','sp'), end = ""); print(" if fitPlotFunc else lambda s: ", end = ""); print(x.replace("sp","np")); round_floats = lambda expr, ndigits: expr.xreplace({f: sp.Float(round(float(f), ndigits)) for f in expr.atoms(sp.Float)}); func_sym_r = round_floats(eval(y), 2); print(sp.latex(func_sym_r));
     '''
     
     track_idx = 0:
        depth = 4:
            Best score = 0.00451617, SNE = 220.427
            Squared-norm error for each equation: 220.427
            Best expression = (((12.785252 / (x0 + -0.061502)) * ((0.985492 - x0) - (x0 ^ 0.285370))) * (((x0 * 0.962481) ^ (x0 ^ -0.798924)) ^ cos((8.423473 ^ x0))))
            Best expression (original format) = * * / 12.785252 + x0 -0.061502 - - 0.985492 x0 ^ x0 0.285370 ^ ^ * x0 0.962481 ^ x0 -0.798924 cos ^ 8.423473 x0
        depth = 4, maxsize = 9:
            Best score = 0.00110916, SNE = 900.583
            Squared-norm error for each equation: 900.583
            Best expression = (-34.520199 * sech((sqrt(9.030267) - (x0 ^ -1.453420))))
            Best expression (original format) = -34.520199 9.030267 sqrt x0 -1.453420 ^ - sech *
        depth = 5, maxsize = 17:
            Best score = 0.00844909, SNE = 117.356
            Squared-norm error for each equation: 117.356
            Best expression = ((32.963733 / (2.007435 ^ cos((-7.438650 * x0)))) * (-0.629287 + sech(((x0 + -0.192839) / -0.089423))))
            Best expression (original format) = 32.963733 2.007435 -7.438650 x0 * cos ^ / -0.629287 x0 -0.192839 + -0.089423 / sech + *
        depth = 6, maxsize = 24:
            Best score = 0.0164764, SNE = 59.6928
            Squared-norm error for each equation: 59.6928
            Best expression = (((-32.328441 * (0.499611 ^ cos((7.372799 * x0)))) * (0.628265 - sech((-11.271965 * (x0 - 0.192670))))) + cos((-31.669971 / (4 ^ x0))))
            Best expression (original format) = + * * -32.328441 ^ 0.499611 cos * 7.372799 x0 - 0.628265 sech * -11.271965 - x0 0.192670 cos / -31.669971 ^ 4 x0
        depth = 7, maxsize = 30:
            Best score = 0.0224682, SNE = 43.5074
            Squared-norm error for each equation: 43.5074
            Best expression = ((((0.507194 ^ (-5.130599 + cos((x0 * -7.361629)))) * (-0.622545 + sech((-2.263896 - (-11.663861 * x0))))) + cos((4 ^ (2.492631 - x0)))) + ((0.981453 + x0) ^ -72.499292))
            Best expression (original format) = 0.507194 -5.130599 x0 -7.361629 * cos + ^ -0.622545 -2.263896 -11.663861 x0 * - sech + * 4 2.492631 x0 - ^ cos + 0.981453 x0 + -72.499292 ^ +
        depth = 8, maxsize = 36:
            Best score = 0.0275609, SNE = 35.2833
            Squared-norm error for each equation: 35.2833
            Best expression = (((((0.509134 ^ (-5.152372 + cos((x0 * -7.364836)))) * (-0.621656 + sech((-2.278212 + (11.746326 * x0))))) + cos((4 ^ (2.492631 - x0)))) + ((x0 + 0.981491) ^ -72.822877)) - (0.010065 / (-0.481231 + x0)))
            Best expression (original format) = 0.509134 -5.152372 x0 -7.364836 * cos + ^ -0.621656 -2.278212 11.746326 x0 * + sech + * 4 2.492631 x0 - ^ cos + x0 0.981491 + -72.822877 ^ + 0.010065 -0.481231 x0 + / -
        depth = 8, maxsize = 37:
            Best score = 0.0372662, SNE = 25.834
            Squared-norm error for each equation: 25.834
            Best expression = (((((0.508292 ^ (-5.130599 + cos((x0 * 7.306336)))) * (-0.622545 + sech((-2.263896 - (-11.663861 * x0))))) + cos((4 ^ (2.492631 - x0)))) + ((0.987333 + x0) ^ -72.499292)) + cos((sin(x0) / exp(-3.515159))))
            Best expression (original format) = 0.508292 -5.130599 x0 7.306336 * cos + ^ -0.622545 -2.263896 -11.663861 x0 * - sech + * 4 2.492631 x0 - ^ cos + 0.987333 x0 + -72.499292 ^ + x0 sin -3.515159 exp / cos +

    track_idx = 1:
        depth = 5, maxsize = 9:
            Best score = 0.000360526, SNE = 2772.72
            Squared-norm error for each equation: 2772.72
            Best expression = (31.554215 - (18.997395 ^ cos((16 * x0))))
            Best expression (original format) = 31.554215 18.997395 16 x0 * cos ^ -
        depth = 6, maxsize = 18:
            Best score = 0.00163711, SNE = 609.831
            Squared-norm error for each equation: 609.831
            Best expression = ((x0 + -0.450211) / ((arccos(x0) * (x0 ^ 0.476227)) - sech((0.785077 - (x0 ^ 7.38905609893065)))))
            Best expression (original format) = x0 -0.450211 + x0 arccos x0 0.476227 ^ * 0.785077 x0 7.38905609893065 ^ - sech - /

     track_idx = 2:
        depth = 5, maxsize = 9:
            Best score = 1.57292e-05, SNE = 63574.9
            Squared-norm error for each equation: 63574.9
            Best expression = (2.345027 + (x0 / (cos(sqrt(x0)) - 2.266180070913597)))
            Best expression (original format) = 2.345027 x0 x0 sqrt cos 2.266180070913597 - / +
        depth = 6, maxsize = 19:
            Best score = 3.57377e-05, SNE = 27980.7
            Squared-norm error for each equation: 27980.7
            Best expression = ((0.832126 * (x0 / (cos(sqrt(x0)) - 2))) - (-21.615306 * cos((4.207354924039483 * sqrt((168.000000 + x0))))))
            Best expression (original format) = 0.832126 x0 x0 sqrt cos 2 - / * -21.615306 4.207354924039483 168.000000 x0 + sqrt * cos * -
        depth = 7, maxsize = 29:
            Best score = 8.01385e-05, SNE = 12477.4
            Squared-norm error for each equation: 12477.4
            Best expression = (((-0.452316 * (x0 / (0.558241 ^ cos(sqrt(x0))))) + (23.928732 * cos((4.207354924039483 * sqrt((168.000000 + x0)))))) + (3.467233 - (-21.559683 * tanh(cos((1.014850 ^ x0))))))
            Best expression (original format) = -0.452316 x0 0.558241 x0 sqrt cos ^ / * 23.928732 4.207354924039483 168.000000 x0 + sqrt * cos * + 3.467233 -21.559683 1.014850 x0 ^ cos tanh * - +
        depth = 8, maxsize = 39:
            Best score = 0.000128622, SNE = 7773.74
            Squared-norm error for each equation: 7773.74
            Best expression = ((((-0.432626 * (x0 * (1.882588 ^ cos(sqrt(x0))))) + (25.974640 * cos((4.207354924039483 * sqrt((168.220938 + x0)))))) + (2.565390 - (-22.862463 * tanh(cos((1.014421 ^ x0)))))) + (7.407921 * sin((-21.205194 - (2 ^ ln(x0))))))
            Best expression (original format) = -0.432626 x0 1.882588 x0 sqrt cos ^ * * 25.974640 4.207354924039483 168.220938 x0 + sqrt * cos * + 2.565390 -22.862463 1.014421 x0 ^ cos tanh * - + 7.407921 -21.205194 2 x0 ln ^ - sin * +
        depth = 9, maxsize = 49:
            Best score = 0.000213868, SNE = 4674.79
            Squared-norm error for each equation: 4674.79
            Best expression = (((((-0.436897 * (x0 / (0.542974 ^ cos(sqrt(x0))))) - (-25.464032 * cos((4.207354924039483 * sqrt((168.220938 + x0)))))) + (2.526539 - (-23.049172 * tanh(cos((1.014421 ^ x0)))))) - (-7.157784 * cos((-2.448741 + (2 ^ ln(x0)))))) + ((x0 ^ 0.595393) / (118.95633426995997 - (x0 ^ 1.000996))))
            Best expression (original format) = -0.436897 x0 0.542974 x0 sqrt cos ^ / * -25.464032 4.207354924039483 168.220938 x0 + sqrt * cos * - 2.526539 -23.049172 1.014421 x0 ^ cos tanh * - + -7.157784 -2.448741 2 x0 ln ^ + cos * - x0 0.595393 ^ 118.95633426995997 x0 1.000996 ^ - / +
        depth = 10, maxsize = 59:
            Best score = 0.00032041, SNE = 3120
            Squared-norm error for each equation: 3120
            Best expression = ((((((-0.440023 * (x0 / (0.542656 ^ cos(sqrt(x0))))) - (-25.646620 * cos((4.207354924039483 * sqrt((168.220938 + x0)))))) + (2.850168 + (23.229391 * tanh(cos((1.014421 ^ x0)))))) - (7.688486 * cos((-49.540434 + (2 ^ ln(x0)))))) + ((x0 ^ 0.576049) / (118.95633426995997 - (x0 ^ 1.000996)))) - (cos((5.056729 - (x0 / exp(0.988960)))) * -4.366545))
            Best expression (original format) = -0.440023 x0 0.542656 x0 sqrt cos ^ / * -25.646620 4.207354924039483 168.220938 x0 + sqrt * cos * - 2.850168 23.229391 1.014421 x0 ^ cos tanh * + + 7.688486 -49.540434 2 x0 ln ^ + cos * - x0 0.576049 ^ 118.95633426995997 x0 1.000996 ^ - / + 5.056729 x0 0.988960 exp / - cos -4.366545 * -
        depth = 11, maxsize = 69:
            Best score = 0.000392308, SNE = 2548.01
            Squared-norm error for each equation: 2548.01
            Best expression = (((((((-0.435995 * (x0 / (0.537501 ^ cos(sqrt(x0))))) + (25.633881 * cos((4.207354924039483 * sqrt((168.220938 + x0)))))) + (2.659658 + (23.229391 * tanh(cos((1.014344 ^ x0)))))) + (7.991940 * sin((-0.867100 + (2 ^ ln(x0)))))) + ((x0 ^ 0.579037) / (118.95633426995997 - (x0 ^ 1.000973)))) + (cos((-1.180394 - (x0 / 2.685784865116654))) * 4.122059)) + arcsin(cos((1.502449 * (x0 ^ 0.7371027432716666)))))
            Best expression (original format) = -0.435995 x0 0.537501 x0 sqrt cos ^ / * 25.633881 4.207354924039483 168.220938 x0 + sqrt * cos * + 2.659658 23.229391 1.014344 x0 ^ cos tanh * + + 7.991940 -0.867100 2 x0 ln ^ + sin * + x0 0.579037 ^ 118.95633426995997 x0 1.000973 ^ - / + -1.180394 x0 2.685784865116654 / - cos 4.122059 * + 1.502449 x0 0.7371027432716666 ^ * cos arcsin +
        depth = 12, maxsize = 76:
            Best score = 0.000635363, SNE = 1572.9
            Squared-norm error for each equation: 1572.9
            Best expression = ((((((((-1.435159 * (x0 ^ (1.044739 ^ cos(sqrt(x0))))) - (-26.235216 * cos((4.207354924039483 * sqrt((168.000000 + x0)))))) + (x0 - (-25.311785 * tanh(cos((1.014267 ^ x0)))))) - (8.292410 * sin((8.438624 + (x0 ^ 0.6931471805599453))))) + ((-103.794895 + x0) / (118.95633426995997 - (x0 ^ 1.000946)))) - (cos((1.126893 + (x0 / 2.685784865116654))) * -4.426079)) + arcsin(cos((1.502449 * (x0 ^ 0.7371027432716666))))) + exp((-1.940061 * cos(((103.614820 - x0) * 0.2658022288340797)))))
            Best expression (original format) = -1.435159 x0 1.044739 x0 sqrt cos ^ ^ * -26.235216 4.207354924039483 168.000000 x0 + sqrt * cos * - x0 -25.311785 1.014267 x0 ^ cos tanh * - + 8.292410 8.438624 x0 0.6931471805599453 ^ + sin * - -103.794895 x0 + 118.95633426995997 x0 1.000946 ^ - / + 1.126893 x0 2.685784865116654 / + cos -4.426079 * - 1.502449 x0 0.7371027432716666 ^ * cos arcsin + -1.940061 103.614820 x0 - 0.2658022288340797 * cos * exp +
        depth = 13: maxsize = 85:
            Best score = 0.0008257, SNE = 1210.09
            Squared-norm error for each equation: 1210.09
            Best expression = (((((((((-1.435127 * (x0 ^ (1.044635 ^ cos(sqrt(x0))))) + (26.253885 * cos((4.207354924039483 * sqrt((168.000000 + x0)))))) + (x0 - (-25.143824 * tanh(cos((1.014278 ^ x0)))))) - (8.212401 * sin((1.004750 - (x0 ^ 0.6931471805599453))))) + ((-104.539086 + x0) / (118.95633426995997 - (x0 / 0.995440)))) - (cos((1.137694 + (x0 / 2.685784865116654))) * -4.486179)) + arcsin(cos((1.502449 * (x0 ^ 0.7371027432716666))))) + exp((1.946541 * cos(((-2.721564 - x0) * 0.2658022288340797))))) + (sin((x0 * -0.45018598229727835)) / (0.5 - tanh(x0))))
            Best expression (original format) = -1.435127 x0 1.044635 x0 sqrt cos ^ ^ * 26.253885 4.207354924039483 168.000000 x0 + sqrt * cos * + x0 -25.143824 1.014278 x0 ^ cos tanh * - + 8.212401 1.004750 x0 0.6931471805599453 ^ - sin * - -104.539086 x0 + 118.95633426995997 x0 0.995440 / - / + 1.137694 x0 2.685784865116654 / + cos -4.486179 * - 1.502449 x0 0.7371027432716666 ^ * cos arcsin + 1.946541 -2.721564 x0 - 0.2658022288340797 * cos * exp + x0 -0.45018598229727835 * sin 0.5 x0 tanh - / +
        depth = 14, maxsize = 95:
            Best score = 0.00105224, SNE = 949.357
            Squared-norm error for each equation: 949.357
            Best expression = ((((((((((-1.434598 * (x0 ^ (1.044655 ^ cos(sqrt(x0))))) + (26.204516 * cos((4.207354924039483 * sqrt((168.000000 + x0)))))) + (x0 + (25.178137 * tanh(cos((1.014267 ^ x0)))))) + (-8.318077 * sin((-4.128074 + (x0 ^ 0.6931471805599453))))) + ((-104.910201 + x0) / (118.95633426995997 - (x0 / 0.995676)))) - (sin((0.985580 * (x0 / 2.685784865116654))) * 4.600156)) + arcsin(cos((1.502449 * (x0 ^ 0.7371027432716666))))) + exp((1.967990 * sin(((3.355986 - x0) * 0.2658022288340797))))) - (sin((x0 * -0.45018598229727835)) / sech(1.502677))) + ((1.003001 ^ 168.000000) * cos((x0 / (1.030905 - 2.718281828459045)))))
            Best expression (original format) = -1.434598 x0 1.044655 x0 sqrt cos ^ ^ * 26.204516 4.207354924039483 168.000000 x0 + sqrt * cos * + x0 25.178137 1.014267 x0 ^ cos tanh * + + -8.318077 -4.128074 x0 0.6931471805599453 ^ + sin * + -104.910201 x0 + 118.95633426995997 x0 0.995676 / - / + 0.985580 x0 2.685784865116654 / * sin 4.600156 * - 1.502449 x0 0.7371027432716666 ^ * cos arcsin + 1.967990 3.355986 x0 - 0.2658022288340797 * sin * exp + x0 -0.45018598229727835 * sin 1.502677 sech / - 1.003001 168.000000 ^ x0 1.030905 2.718281828459045 - / cos * +
        depth = 15, maxsize = 104:
            Best score = 0.00123659, SNE = 807.673
            Squared-norm error for each equation: 807.673
            Best expression = (((((((((((-1.434557 * (x0 ^ (1.044505 ^ cos(sqrt(x0))))) - (-26.210051 * cos((4.207354924039483 * sqrt((168.000000 + x0)))))) + (x0 + (25.172898 * tanh(cos((1.014294 ^ x0)))))) + (-8.254968 * sin((0.983557 - (x0 ^ 0.6931471805599453))))) + ((-105.189336 + x0) / (118.95633426995997 - (x0 * 1.004240)))) + (sin((0.985180 * (x0 / 2.685784865116654))) * -4.586794)) + arcsin(cos((1.502449 * (x0 ^ 0.7371027432716666))))) + exp((1.949297 * sin(((97.939188 - x0) * 0.2658022288340797))))) + (sin((x0 * -0.45018598229727835)) / -0.4107263935558514)) + (1.722899 * cos((x0 / -1.6880058284590451)))) + (sech(x0) + cos((0.696976831813758 * x0))))
            Best expression (original format) = -1.434557 x0 1.044505 x0 sqrt cos ^ ^ * -26.210051 4.207354924039483 168.000000 x0 + sqrt * cos * - x0 25.172898 1.014294 x0 ^ cos tanh * + + -8.254968 0.983557 x0 0.6931471805599453 ^ - sin * + -105.189336 x0 + 118.95633426995997 x0 1.004240 * - / + 0.985180 x0 2.685784865116654 / * sin -4.586794 * + 1.502449 x0 0.7371027432716666 ^ * cos arcsin + 1.949297 97.939188 x0 - 0.2658022288340797 * sin * exp + x0 -0.45018598229727835 * sin -0.4107263935558514 / + 1.722899 x0 -1.6880058284590451 / cos * + x0 sech 0.696976831813758 x0 * cos + +
        depth = 16, maxsize = 107:
            Best score = 0.00167321, SNE = 596.653
            Squared-norm error for each equation: 596.653
            Best expression = ((((((((((((-1.438588 * (x0 ^ (1.043900 ^ cos(sqrt(x0))))) - (-26.218935 * cos((4.207354924039483 * sqrt((168.000000 + x0)))))) + (x0 + (24.750491 * tanh(cos((1.014294 ^ x0)))))) - (8.151847 * sin((1.023804 - (x0 ^ 0.6931471805599453))))) + ((-106.088260 + x0) / (118.95633426995997 - (x0 / 0.995287)))) + (sin((0.985199 * (x0 / 2.685784865116654))) * -4.586794)) + cos(log((1.502449 ^ (x0 ^ 0.997872))))) + exp((-1.995201 * sin(((44.248750 + x0) * 0.2658022288340797))))) + (sin((x0 * -0.45018598229727835)) / -0.4107263935558514)) + (1.497965 * cos((x0 / -1.6880058284590451)))) + (sech(x0) + cos((0.696976831813758 * x0)))) - (tanh(0.031806) / sin((4 * (x0 - 0.7615941559557649)))))
            Best expression (original format) = -1.438588 x0 1.043900 x0 sqrt cos ^ ^ * -26.218935 4.207354924039483 168.000000 x0 + sqrt * cos * - x0 24.750491 1.014294 x0 ^ cos tanh * + + 8.151847 1.023804 x0 0.6931471805599453 ^ - sin * - -106.088260 x0 + 118.95633426995997 x0 0.995287 / - / + 0.985199 x0 2.685784865116654 / * sin -4.586794 * + 1.502449 x0 0.997872 ^ ^ log cos + -1.995201 44.248750 x0 + 0.2658022288340797 * sin * exp + x0 -0.45018598229727835 * sin -0.4107263935558514 / + 1.497965 x0 -1.6880058284590451 / cos * + x0 sech 0.696976831813758 x0 * cos + + 0.031806 tanh 4 x0 0.7615941559557649 - * sin / -
        depth = 17, maxsize = 117:
            Best score = 0.00224077, SNE = 445.276
            Squared-norm error for each equation: 445.276
            Best expression = (((((((((((((-2.446553 * (x0 ^ (1.026091 ^ cos(sqrt(x0))))) - (-26.239310 * cos((4.207354924039483 * sqrt((168.000000 + x0)))))) + (x0 + (25.089302 * tanh(cos((1.014294 ^ x0)))))) - (-7.997338 * sin((-1.057546 + (x0 ^ 0.6931471805599453))))) + ((-106.322846 + x0) / (118.95633426995997 - (x0 / 0.995131)))) + (sin((0.985374 * (x0 / 2.685784865116654))) * -4.973426)) + cos(log((1.502449 ^ (x0 ^ 0.997305))))) + exp((-2.066466 * sin(((20.589308 + x0) * 0.2658022288340797))))) + (sin((x0 * -0.45018598229727835)) / -0.4107263935558514)) + (1.609716 * cos((x0 / -1.6880058284590451)))) + (x0 + cos((0.696976831813758 * x0)))) - (0.033090 / sin((4 * (x0 - 0.7615941559557649))))) + sin(sqrt((3.751510 + (1.052177 ^ (x0 ^ 0.993021))))))
            Best expression (original format) = -2.446553 x0 1.026091 x0 sqrt cos ^ ^ * -26.239310 4.207354924039483 168.000000 x0 + sqrt * cos * - x0 25.089302 1.014294 x0 ^ cos tanh * + + -7.997338 -1.057546 x0 0.6931471805599453 ^ + sin * - -106.322846 x0 + 118.95633426995997 x0 0.995131 / - / + 0.985374 x0 2.685784865116654 / * sin -4.973426 * + 1.502449 x0 0.997305 ^ ^ log cos + -2.066466 20.589308 x0 + 0.2658022288340797 * sin * exp + x0 -0.45018598229727835 * sin -0.4107263935558514 / + 1.609716 x0 -1.6880058284590451 / cos * + x0 0.696976831813758 x0 * cos + + 0.033090 4 x0 0.7615941559557649 - * sin / - 3.751510 1.052177 x0 0.993021 ^ ^ + sqrt sin +
        depth = 18, maxsize = 125:
            Best score = 0.002901, SNE = 343.709
            Squared-norm error for each equation: 343.709
            Best expression = ((((((((((((((-2.446317 * (x0 ^ (1.026171 ^ cos(sqrt(x0))))) - (-26.267023 * cos((4.207354924039483 * sqrt((168.000000 + x0)))))) + (x0 + (25.114484 * tanh(cos((1.014280 ^ x0)))))) - (-8.094348 * cos((-3.652141 - (x0 ^ 0.6931471805599453))))) + ((-106.639258 + x0) / (118.95633426995997 - (x0 / 0.995137)))) - (sin((0.985554 * (x0 / 2.685784865116654))) * 5.048220)) + cos(log((1.502449 ^ (x0 ^ 0.997721))))) + exp((2.081938 * sin(((2.962750 - x0) * 0.2658022288340797))))) + (sin((x0 * -0.45018598229727835)) * -2.744077)) - (-1.635574 * cos((x0 / -1.6880058284590451)))) + (x0 + cos((0.696976831813758 * x0)))) - (0.031400 / sin((4 * (x0 - 0.7615941559557649))))) + sin(sqrt((4.063501 + (1.052177 ^ (x0 ^ 0.992804)))))) + cos((~(1.027196) * (2 * (x0 ^ 0.8414709848078965)))))
            Best expression (original format) = -2.446317 x0 1.026171 x0 sqrt cos ^ ^ * -26.267023 4.207354924039483 168.000000 x0 + sqrt * cos * - x0 25.114484 1.014280 x0 ^ cos tanh * + + -8.094348 -3.652141 x0 0.6931471805599453 ^ - cos * - -106.639258 x0 + 118.95633426995997 x0 0.995137 / - / + 0.985554 x0 2.685784865116654 / * sin 5.048220 * - 1.502449 x0 0.997721 ^ ^ log cos + 2.081938 2.962750 x0 - 0.2658022288340797 * sin * exp + x0 -0.45018598229727835 * sin -2.744077 * + -1.635574 x0 -1.6880058284590451 / cos * - x0 0.696976831813758 x0 * cos + + 0.031400 4 x0 0.7615941559557649 - * sin / - 4.063501 1.052177 x0 0.992804 ^ ^ + sqrt sin + 1.027196 ~ 2 x0 0.8414709848078965 ^ * * cos +
        depth = 19, maxsize = 135:
            Best score = 0.00327277, SNE = 304.552
            Squared-norm error for each equation: 304.552
            Best expression = (((((((((((((((-2.446627 * (x0 ^ (1.026127 ^ cos(sqrt(x0))))) - (-26.155498 * cos((4.207354924039483 * sqrt((168.000000 + x0)))))) + (x0 - (-25.093806 * tanh(cos((1.014280 ^ x0)))))) - (8.001965 * cos((-0.515339 - (x0 ^ 0.6931471805599453))))) + ((-106.697598 + x0) / (118.95633426995997 - (x0 * 1.004942)))) - (sin((x0 * (0.985907 / 2.685784865116654))) * 5.048220)) + cos(log((1.502449 ^ (x0 ^ 0.997906))))) + exp((2.070110 * sin(((2.852405 - x0) * 0.2658022288340797))))) - (sin((x0 * -0.45018598229727835)) * 2.698598)) - (-1.678895 * cos((x0 / -1.6880058284590451)))) + (x0 + cos((0.696976831813758 * x0)))) - (0.032015 / sin((4 * (x0 - 0.7615941559557649))))) + sin(sqrt((3.218607 + (1.052177 ^ (x0 ^ 0.992804)))))) + cos((~(1.027706) * (2 * (x0 ^ 0.8414709848078965))))) + (168.000000 * (0.012165 / (11.175638 + (x0 - 54.598150033144236)))))
            Best expression (original format) = -2.446627 x0 1.026127 x0 sqrt cos ^ ^ * -26.155498 4.207354924039483 168.000000 x0 + sqrt * cos * - x0 -25.093806 1.014280 x0 ^ cos tanh * - + 8.001965 -0.515339 x0 0.6931471805599453 ^ - cos * - -106.697598 x0 + 118.95633426995997 x0 1.004942 * - / + x0 0.985907 2.685784865116654 / * sin 5.048220 * - 1.502449 x0 0.997906 ^ ^ log cos + 2.070110 2.852405 x0 - 0.2658022288340797 * sin * exp + x0 -0.45018598229727835 * sin 2.698598 * - -1.678895 x0 -1.6880058284590451 / cos * - x0 0.696976831813758 x0 * cos + + 0.032015 4 x0 0.7615941559557649 - * sin / - 3.218607 1.052177 x0 0.992804 ^ ^ + sqrt sin + 1.027706 ~ 2 x0 0.8414709848078965 ^ * * cos + 168.000000 0.012165 11.175638 x0 54.598150033144236 - + / * +
        depth = 20, maxsize = 141:
            Best score = 0.00359927, SNE = 276.834
            Squared-norm error for each equation: 276.834
            Best expression = ((((((((((((((((-2.444296 * (x0 ^ (1.026203 ^ cos(sqrt(x0))))) - (-26.199496 * cos((4.207354924039483 * sqrt((168.000000 + x0)))))) + (x0 + (25.019477 * tanh(cos((1.014280 ^ x0)))))) - (-8.058567 * cos((-2.633602 + (x0 ^ 0.6931471805599453))))) + (11.696530 / (118.95633426995997 - (x0 / 0.995059)))) - (sin((x0 * 0.3670833851233197)) * 4.957468)) + cos(log((1.502449 ^ (x0 ^ 0.997889))))) + exp((-2.060603 * sin(((44.498356 + x0) * 0.2658022288340797))))) + (sin((x0 * -0.45018598229727835)) * -2.701240)) - (-1.599485 * cos((x0 / -1.6880058284590451)))) + (x0 + cos((0.696976831813758 * x0)))) - (0.032499 / sin((4 * (x0 - 0.7615941559557649))))) + sin(sqrt((2.702079 + (1.052177 ^ (x0 ^ 0.992888)))))) + cos((1.027621 * (2 * (x0 ^ 0.8414709848078965))))) - (1.225249 - (2.020931 / (11.175638 + (x0 - 54.598150033144236))))) - (-6.130043 / ((1.218107 + x0) / cos((x0 - 6.804936)))))
            Best expression (original format) = -2.444296 x0 1.026203 x0 sqrt cos ^ ^ * -26.199496 4.207354924039483 168.000000 x0 + sqrt * cos * - x0 25.019477 1.014280 x0 ^ cos tanh * + + -8.058567 -2.633602 x0 0.6931471805599453 ^ + cos * - 11.696530 118.95633426995997 x0 0.995059 / - / + x0 0.3670833851233197 * sin 4.957468 * - 1.502449 x0 0.997889 ^ ^ log cos + -2.060603 44.498356 x0 + 0.2658022288340797 * sin * exp + x0 -0.45018598229727835 * sin -2.701240 * + -1.599485 x0 -1.6880058284590451 / cos * - x0 0.696976831813758 x0 * cos + + 0.032499 4 x0 0.7615941559557649 - * sin / - 2.702079 1.052177 x0 0.992888 ^ ^ + sqrt sin + 1.027621 2 x0 0.8414709848078965 ^ * * cos + 1.225249 2.020931 11.175638 x0 54.598150033144236 - + / - - -6.130043 1.218107 x0 + x0 6.804936 - cos / / -
        depth = 21, maxsize = 150:
            Best score = 0.004133, SNE = 240.955
            Squared-norm error for each equation: 240.955
            Best expression = (((((((((((((((((-2.443340 * (x0 ^ (1.026195 ^ cos(sqrt(x0))))) - (-26.141684 * cos((4.207354924039483 * sqrt((168.000000 + x0)))))) + (x0 + (25.068585 * tanh(cos((1.014280 ^ x0)))))) - (-8.087407 * cos((-2.638495 + (x0 ^ 0.6931471805599453))))) - (-11.661316 / (118.95633426995997 - (x0 + 0.579601)))) + (sin((x0 * 0.3670833851233197)) * -4.949110)) + cos(log((1.502449 ^ (x0 ^ 0.998111))))) + exp((2.027208 * sin(((2.782294 - x0) * 0.2658022288340797))))) + (sin((x0 * -0.45018598229727835)) * -2.666797)) + (1.509721 * cos((x0 / -1.6880058284590451)))) + (x0 + cos((0.696976831813758 * x0)))) - (0.032277 / sin((4 * (x0 - 0.7615941559557649))))) + sin(sqrt((2.923343 + (1.052177 ^ (x0 ^ 0.992888)))))) + cos((1.027655 * (2 * (x0 ^ 0.8414709848078965))))) - (1.244610 - (2.026644 / (11.175638 + (x0 - 54.598150033144236))))) + (-6.551556 / ((1.214443 + x0) / cos((x0 - -2.647537))))) + (-0.036732 / asin(cos(sqrt((1.132285 + x0))))))
            Best expression (original format) = + + - + + - + + + + + + - - + - * -2.443340 ^ x0 ^ 1.026195 cos sqrt x0 * -26.141684 cos * 4.207354924039483 sqrt + 168.000000 x0 + x0 * 25.068585 tanh cos ^ 1.014280 x0 * -8.087407 cos + -2.638495 ^ x0 0.6931471805599453 / -11.661316 - 118.95633426995997 + x0 0.579601 * sin * x0 0.3670833851233197 -4.949110 cos log ^ 1.502449 ^ x0 0.998111 exp * 2.027208 sin * - 2.782294 x0 0.2658022288340797 * sin * x0 -0.45018598229727835 -2.666797 * 1.509721 cos / x0 -1.6880058284590451 + x0 cos * 0.696976831813758 x0 / 0.032277 sin * 4 - x0 0.7615941559557649 sin sqrt + 2.923343 ^ 1.052177 ^ x0 0.992888 cos * 1.027655 * 2 ^ x0 0.8414709848078965 - 1.244610 / 2.026644 + 11.175638 - x0 54.598150033144236 / -6.551556 / + 1.214443 x0 cos - x0 -2.647537 / -0.036732 asin cos sqrt + 1.132285 x0
        depth = 22, maxsize = 160:
            Best score = 0.00531365, SNE = 187.195
            Squared-norm error for each equation: 187.195
            Best expression = ((((((((((((((((((-2.443340 * (x0 ^ (1.026205 ^ cos(sqrt(x0))))) - (-26.134535 * cos((4.207354924039483 * sqrt((168.000000 + x0)))))) + (x0 + (25.152657 * tanh(cos((1.014280 ^ x0)))))) - (-8.162444 * cos((-2.638495 + (x0 ^ 0.6931471805599453))))) - (-11.891217 / (118.95633426995997 - (x0 + 0.567287)))) - (sin((x0 * 0.3670833851233197)) * 4.846034)) + cos(log((1.502449 ^ (x0 ^ 0.997729))))) + exp((1.982657 * sin(((2.514225 - x0) * 0.2658022288340797))))) + (sin((x0 * -0.45018598229727835)) * -2.453809)) + (1.545488 * cos((x0 / -1.6880058284590451)))) + (x0 + cos((0.696976831813758 * x0)))) - (0.031557 / sin((4 * (x0 - 0.7615941559557649))))) + sin(sqrt((3.212886 + (1.052177 ^ (x0 ^ 0.992888)))))) + cos((1.027655 * (2 * (x0 ^ 0.8414709848078965))))) - (1.143581 - (2.253225 / (11.175638 + (x0 - 54.598150033144236))))) + (-6.012152 / ((1.214443 + x0) / cos((x0 - 9.779302))))) + (-0.041021 / asin(cos(sqrt((1.133684 + x0)))))) + sin(cos(((x0 / 3.152574) ^ 1.113573))))
            Best expression (original format) = + + + - + + - + + + + + - - - + - * -2.443340 ^ x0 ^ 1.026205 cos sqrt x0 * -26.134535 cos * 4.207354924039483 sqrt + 168.000000 x0 + x0 * 25.152657 tanh cos ^ 1.014280 x0 * -8.162444 cos + -2.638495 ^ x0 0.6931471805599453 / -11.891217 - 118.95633426995997 + x0 0.567287 * sin * x0 0.3670833851233197 4.846034 cos log ^ 1.502449 ^ x0 0.997729 exp * 1.982657 sin * - 2.514225 x0 0.2658022288340797 * sin * x0 -0.45018598229727835 -2.453809 * 1.545488 cos / x0 -1.6880058284590451 + x0 cos * 0.696976831813758 x0 / 0.031557 sin * 4 - x0 0.7615941559557649 sin sqrt + 3.212886 ^ 1.052177 ^ x0 0.992888 cos * 1.027655 * 2 ^ x0 0.8414709848078965 - 1.143581 / 2.253225 + 11.175638 - x0 54.598150033144236 / -6.012152 / + 1.214443 x0 cos - x0 9.779302 / -0.041021 asin cos sqrt + 1.133684 x0 sin cos ^ / x0 3.152574 1.113573
        
        depth = 6, maxsize = 25, bad_ops = {"exp", "ln", "log", "^", "/"}
            Best score = 6.98285e-05, SNE = 14319.8
            Squared-norm error for each equation: 14319.8
            Best expression = ((cos(((x0 + 11.968373) * 0.13598386240567323)) * ~(((-0.231705 * x0) + 39.219198))) - ((x0 * -0.276514) * (-1.863160 - sin((1.094133 * sqrt(x0))))))
            Best expression (original format) = - * cos * + x0 11.968373 0.13598386240567323 ~ + * -0.231705 x0 39.219198 * * x0 -0.276514 - -1.863160 sin * 1.094133 sqrt x0
        depth = 7, maxsize = 27, bad_ops = {"exp", "ln", "log", "^", "/"}
            Best score = 8.19886e-05, SNE = 12195.8
            Squared-norm error for each equation: 12195.8
            Best expression = ((sin(((x0 + sin(sqrt(x0))) * 0.13599420224810638)) * ~(((0.223351 * x0) + -39.031218))) + ((x0 * 0.272925) * (-1.884260 - sin((1.096031 * sqrt(x0))))))
            Best expression (original format) = + * sin * + x0 sin sqrt x0 0.13599420224810638 ~ + * 0.223351 x0 -39.031218 * * x0 0.272925 - -1.884260 sin * 1.096031 sqrt x0
        depth = 8, maxsize = 36, bad_ops = {"exp", "ln", "log", "^", "/"}
            Best score = 0.000112852, SNE = 8860.18
            Squared-norm error for each equation: 8860.18
            Best expression = (((sin(((x0 + sin(sqrt(x0))) * 0.13599420224810638)) * (29.034195 + ((-0.000009 * x0) * (x0 * x0)))) + ((x0 * 0.270243) * (-1.900612 - sin((1.095518 * sqrt(x0)))))) - (7.504397 * cos((x0 * 0.18450196567500599))))
            Best expression (original format) = x0 x0 sqrt sin + 0.13599420224810638 * sin 29.034195 -0.000009 x0 * x0 x0 * * + * x0 0.270243 * -1.900612 1.095518 x0 sqrt * sin - * + 7.504397 x0 0.18450196567500599 * cos * -
        depth = 9, maxsize = 47, bad_ops = {"exp", "ln", "log", "^", "/"}
            Best score = 0.000124988, SNE = 7999.78
            Squared-norm error for each equation: 7999.78
            Best expression = ((((sin(((x0 + sin(sqrt(x0))) * 0.13599420224810638)) * (29.040990 + ((-0.000009 * x0) * (x0 * x0)))) + ((x0 * 0.271124) * (-1.899330 - sin((1.094607 * sqrt(x0)))))) - (7.605615 * cos((x0 * 0.18450196567500599)))) + asin(cos((0.2787780894020867 * (0.999693 * (x0 - 2))))))
            Best expression (original format) = x0 x0 sqrt sin + 0.13599420224810638 * sin 29.040990 -0.000009 x0 * x0 x0 * * + * x0 0.271124 * -1.899330 1.094607 x0 sqrt * sin - * + 7.605615 x0 0.18450196567500599 * cos * - 0.2787780894020867 0.999693 x0 2 - * * cos asin +
        depth = 10, maxsize = 56:
            Best score = 0.000149821, SNE = 6673.63
            Squared-norm error for each equation: 6673.63 290.8555248689591
            Best expression = (((((sin(((x0 + sin(sqrt(x0))) * 0.13599420224810638)) * (28.886312 - ((0.000008 * x0) * (x0 * x0)))) + ((x0 * 0.272853) * (-1.891370 - sin((1.094294 * sqrt(x0)))))) + (-7.617770 * cos((x0 * 0.18450196567500599)))) + asin(cos((0.2787780894020867 * (0.999693 * (x0 - 2)))))) + (sqrt(sqrt(x0)) * sin((x0 * -0.36787944117144233))))
            Best expression (original format) = x0 x0 sqrt sin + 0.13599420224810638 * sin 28.886312 0.000008 x0 * x0 x0 * * - * x0 0.272853 * -1.891370 1.094294 x0 sqrt * sin - * + -7.617770 x0 0.18450196567500599 * cos * + 0.2787780894020867 0.999693 x0 2 - * * cos asin + x0 sqrt sqrt x0 -0.36787944117144233 * sin * +
        depth = 11, maxsize = 65:
            Best score = 0.000175187, SNE = 5707.19
            Squared-norm error for each equation: 5690.95 558
            Best expression = ((((((sin(((x0 + sin(sqrt(x0))) * 0.13599420224810638)) * (27.854452 - ((0.000008 * x0) * (x0 * x0)))) + ((x0 * 0.272853) * ~(sin((1.094294 * sqrt(x0)))))) + (-8.396368 * cos((x0 * 0.18450196567500599)))) + asin(cos((0.2787780894020867 * (x0 - 2))))) + (4.530500 * sin((x0 * -0.36787944117144233)))) + (15.425325 * ((x0 + cos(x0)) * (~(0.033472) + sqrt(sech(x0))))))
            Best expression (original format) = x0 x0 sqrt sin + 0.13599420224810638 * sin 27.854452 0.000008 x0 * x0 x0 * * - * x0 0.272853 * 1.094294 x0 sqrt * sin ~ * + -8.396368 x0 0.18450196567500599 * cos * + 0.2787780894020867 x0 2 - * cos asin + 4.530500 x0 -0.36787944117144233 * sin * + 15.425325 x0 x0 cos + 0.033472 ~ x0 sech sqrt + * * +
        depth = 12, maxsize = 74:
            Best score = 0.000201647, SNE = 4958.17
            Squared-norm error for each equation: 4534.99 423.186
            Best expression = (((((((sin(((x0 + sin(sqrt(x0))) * 0.13599420224810638)) * (27.391414 + ((x0 * -0.000007) * (x0 * x0)))) - ((0.321608 * x0) * sin(sin((1.092289 * sqrt(x0)))))) - (8.015060 * cos((x0 * 0.18450196567500599)))) + arcsin(cos((0.2787780894020867 * (x0 * 0.898195))))) + (5.064484 * sin((x0 * -0.36787944117144233)))) + (7.763419 * ((x0 + (x0 + 0.879127)) * (-0.033499 + sqrt(sech(x0)))))) - ((0.058354 * x0) * sin((x0 * 0.2627831798005332))))
            Best expression (original format) = x0 x0 sqrt sin + 0.13599420224810638 * sin 27.391414 x0 -0.000007 * x0 x0 * * + * 0.321608 x0 * 1.092289 x0 sqrt * sin sin * - 8.015060 x0 0.18450196567500599 * cos * - 0.2787780894020867 x0 0.898195 * * cos arcsin + 5.064484 x0 -0.36787944117144233 * sin * + 7.763419 x0 x0 0.879127 + + -0.033499 x0 sech sqrt + * * + 0.058354 x0 * x0 0.2627831798005332 * sin * -
        depth = 13, maxsize = 82:
            Best score = 0.000222912, SNE = 4485.07
            Squared-norm error for each equation: 4038.1 446.974
            Best expression = ((((((((sin(((x0 + sin(sqrt(x0))) * 0.13599420224810638)) * (27.320094 - ((0.000007 * x0) * (x0 * x0)))) + ((-0.323439 * x0) * sin(sin((1.092289 * sqrt(x0)))))) - (8.034637 * cos((x0 * 0.18450196567500599)))) + arcsin(cos((0.2787780894020867 * (x0 * 0.898195))))) + (5.075383 * sin((x0 * -0.36787944117144233)))) + (7.763419 * ((x0 + (x0 + 1)) * (-0.033530 + sqrt(sech(x0)))))) + ((-0.059082 * x0) * sin((x0 * 0.2627831798005332)))) - arcsin(cos(((x0 * 0.5175124998053154) + -0.862119))))
            Best expression (original format) = x0 x0 sqrt sin + 0.13599420224810638 * sin 27.320094 0.000007 x0 * x0 x0 * * - * -0.323439 x0 * 1.092289 x0 sqrt * sin sin * + 8.034637 x0 0.18450196567500599 * cos * - 0.2787780894020867 x0 0.898195 * * cos arcsin + 5.075383 x0 -0.36787944117144233 * sin * + 7.763419 x0 x0 1 + + -0.033530 x0 sech sqrt + * * + -0.059082 x0 * x0 0.2627831798005332 * sin * + x0 0.5175124998053154 * -0.862119 + cos arcsin -
     */
    static bool added_additive = false;
    thread_local std::vector<std::vector<std::string>> results(x.num_diff_eqns);
    static double const_val = 0.0;
    assert(x.num_diff_eqns == 2);
    if (x.add_additive)
    {
        std::scoped_lock str_lock(Board::thread_locker);
        if (!added_additive)
        {
            if (x.expression_type == "prefix")
            {
                //- x1 additiveCorrections
                results[0].push_back("-"); // -
                results[0].push_back("x1"); // x1
                for (const std::string& i: x.additiveCorrections[0]) // additiveCorrections
                {
                    results[0].push_back(i);
                    if (isdouble(i))
                    {
                        const_val += abs(Stod(i));
                    }
                }
            }
            else if (x.expression_type == "postfix")
            {
                //x1 additiveCorrections -
                results[0].push_back("x1"); // x1
                for (const std::string& i: x.additiveCorrections[0]) // additiveCorrections
                {
                    results[0].push_back(i);
                    if (isdouble(i))
                    {
                        const_val += abs(Stod(i));
                    }
                }
                results[0].push_back("-"); // -
            }
            //Fit the residual `x1 - additiveCorrections`, instead of `x1`
            Board::data["x1"] = x.expression_evaluator(x.params, results[0]);
            added_additive = true;
            results[0].clear();
        }
    }

    for (std::vector<std::string>& res: results)
    {
        res.clear();
        res.reserve(100);
    }
    if (x.expression_type == "prefix")
    {
        //abs - f x1
        results[0].push_back("abs"); // abs
        results[0].push_back("-"); // -
        for (const std::string& i: x.pieces[0]) // f
        {
            if (x.pieces.size() > 1 && i == "x0")
            {
                //replace ("x0" with x.pieces[1])
                for (const std::string& i: x.pieces[1])
                {
                    results[0].push_back(i);
                }
            }
            else
            {
                results[0].push_back(i);
            }
        }
        results[0].push_back("x1"); // x1
    }
    else if (x.expression_type == "postfix")
    {
        //f x1 - abs
        for (const std::string& i: x.pieces[0]) // f
        {
            if (x.pieces.size() > 1 && i == "x0")
            {
                //replace ("x0" with x.pieces[1])
                for (const std::string& i: x.pieces[1])
                {
                    results[0].push_back(i);
                }
            }
            else
            {
                results[0].push_back(i);
            }
        }
        results[0].push_back("x1"); // x1
        results[0].push_back("-"); // -
        results[0].push_back("abs"); // abs
    }
    
    double val = const_val;
//    std::cout << "val = " << val << '\n';
    
    for (const std::string& i: x.pieces[0]) // f
    {
        if (isdouble(i))
        {
            val += abs(Stod(i));
//            std::cout << "i = " << i << ", val = " << val << '\n';
        }
        else if (i.compare(0, 5, "const") == 0)
        {
            if (results[1].empty())
            {
                results[1].push_back(i);
                results[1].push_back("abs");
            }
            else
            {
                results[1].push_back(i);
                results[1].push_back("abs");
                results[1].push_back("+");
            }
        }
    }
//    std::cout << "x.pieces[0] = " << x.pieces[0] << ", val = " << val << '\n';
    if (val > 0.0)
    {
        if (results[1].empty())
        {
            results[1].push_back(to_string_general(val));
            results[1].push_back("abs");
        }
        else
        {
            results[1].push_back(to_string_general(val));
            results[1].push_back("abs");
            results[1].push_back("+");
        }
        results[1].push_back("3e-2");
        results[1].push_back("*");
    }
    if (results[1].empty())
    {
        results[1].push_back("0");
    }
    return results;
}

/*
 Infix: abs(f - x102)
 Postfix: f x102 - abs
 Prefix: abs - f x102
 */
std::vector<std::vector<std::string>> InPaintWildfireSpreadTS(Board& x, bool fit)
{
    /*
     ```
        from sympy import symbols, cos, sin, tanh, sech, acos, log, sympify, latex, multiline_latex, Float
        import re
        replace_vars = lambda x: re.sub(r'\bx(\d+)\b', r'df["x\1"]', x)
        align_rep = lambda x: x.replace('align*','align').replace(r'\\',r'\nonumber \\').replace(r"\end{align}", r"\label{eq:best_sr_eq_inpaint_no_lap_fixed_depth_3}""\n"r"\end{align}")
        round_floats = lambda expr, ndigits: expr.xreplace({f: Float(round(float(f), ndigits)) for f in expr.atoms(Float)})
        f, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65, x66, x67, x68, x69, x70, x71, x72, x73, x74, x75, x76, x77, x78, x79, x80, x81, x82, x83, x84, x85, x86, x87, x88, x89, x90, x91, x92, x93, x94, x95, x96, x97, x98, x99, x100, x101 = symbols('f x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30 x31 x32 x33 x34 x35 x36 x37 x38 x39 x40 x41 x42 x43 x44 x45 x46 x47 x48 x49 x50 x51 x52 x53 x54 x55 x56 x57 x58 x59 x60 x61 x62 x63 x64 x65 x66 x67 x68 x69 x70 x71 x72 x73 x74 x75 x76 x77 x78 x79 x80 x81 x82 x83 x84 x85 x86 x87 x88 x89 x90 x91 x92 x93 x94 x95 x96 x97 x98 x99 x100 x101')
        func = '((((x71 * x69) + (0.82357 ^ (335 / (0.01872 + x31)))) + (((0.496312 + (x67 * -0.6901253057600001)) + 6.2216003257007526e-24) + ((-1.278025 / x14) + -2431.4844))) / (((2838.0127038380915 - (-0.286767 + x76)) + 144.49968) - ((x18 + 2.063746) * (x100 - ((12187.586 ^ x28) + (x101 + -14640.288932156))))))'
        func = func.replace("^","**").replace("~","-")
        func_sym = sympify(func)
        func_sym_r = round_floats(func_sym, 2)
        print(f"func_sym = {align_rep(multiline_latex(f, func_sym_r, 1))}")
        f_res = replace_vars(func)
        print(f"f = {f_res}")
     ```
     Without Laplacian Smoothing:
         Depth = 3:
            Training:
                 Best score = 2.28121e-09, SNE = 4.38364e+08
                 Squared-norm error for each equation: 4.38364e+08
                 Best expression = sech(((x3 + x101) - (x100 / 2.993659)))
                 Best expression (original format) = x3 x101 + x100 2.993659 / - sech
            Validation:
                 Best score = 6.74587e-11, SNE = 1.48239e+10
                 Squared-norm error for each equation: 1.48239e+10
                 Best expression = sech(((x3 + x101) - (x100 / 2.993659)))
                 Best expression (original format) = x3 x101 + x100 2.993659 / - sech
         Depth = 4:
            Training:
                Best score = 2.28159e-09, SNE = 4.38291e+08
                Squared-norm error for each equation: 4.38291e+08
                Best expression = sech((((0.398685 / x61) + (x47 + x101)) - ((x86 + x100) / (x94 - x23))))
                Best expression (original format) = 0.398685 x61 / x47 x101 + + x86 x100 + x94 x23 - / - sech
            Validation:
                Best score = 6.74587e-11, SNE = 1.48239e+10
                Squared-norm error for each equation: 1.48239e+10
                Best expression = sech((((0.398685 / x61) + (x47 + x101)) - ((x86 + x100) / (x94 - x23))))
                Best expression (original format) = 0.398685 x61 / x47 x101 + + x86 x100 + x94 x23 - / - sech
         Depth = 5:
            Training:
                Best score = 2.28171e-09, SNE = 4.38268e+08
                Squared-norm error for each equation: 4.38268e+08
                Best expression = sech(((((x21 + 0.394189) / (0.001595 + x61)) + ((x84 ^ x70) + (x73 + x101))) - ((cos(x101) + (x55 + x100)) / ((x21 ^ x58) - (x21 + x23)))))
                Best expression (original format) = x21 0.394189 + 0.001595 x61 + / x84 x70 ^ x73 x101 + + + x101 cos x55 x100 + + x21 x58 ^ x21 x23 + - / - sech
            Validation:
                Best score = 6.74587e-11, SNE = 1.48239e+10
                Squared-norm error for each equation: 1.48239e+10
                Best expression = sech(((((x21 + 0.394189) / (0.001595 + x61)) + ((x84 ^ x70) + (x73 + x101))) - ((cos(x101) + (x55 + x100)) / ((x21 ^ x58) - (x21 + x23)))))
                Best expression (original format) = x21 0.394189 + 0.001595 x61 + / x84 x70 ^ x73 x101 + + + x101 cos x55 x100 + + x21 x58 ^ x21 x23 + - / - sech
         Depth = 6:
            Training:
                Best score = 2.28184e-09, SNE = 4.38243e+08
                Squared-norm error for each equation: 4.38243e+08
                Best expression = sech((((((x20 + x71) * (x20 + -0.29813)) / ((x81 ^ 1414.708600) + x61)) + (((x89 + 0.001691) ^ x70) + ((x55 / x68) + (x21 + x101)))) - ((((x73 + x29) - (x89 + x71)) + ((x39 ^ x93) + (x53 + x100))) / (((x85 ^ x33) ^ (x3 + x87)) - ((x88 - x20) + (0.004457 + x23))))))
                Best expression (original format) = x20 x71 + x20 -0.29813 + * x81 1414.708600 ^ x61 + / x89 0.001691 + x70 ^ x55 x68 / x21 x101 + + + + x73 x29 + x89 x71 + - x39 x93 ^ x53 x100 + + + x85 x33 ^ x3 x87 + ^ x88 x20 - 0.004457 x23 + + - / - sech
            Validation:
                Best score = 0, SNE = 1.79769e+308
                Squared-norm error for each equation: nan
                Best expression = sech((((((x20 + x71) * (x20 + -0.29813)) / ((x81 ^ 1414.708600) + x61)) + (((x89 + 0.001691) ^ x70) + ((x55 / x68) + (x21 + x101)))) - ((((x73 + x29) - (x89 + x71)) + ((x39 ^ x93) + (x53 + x100))) / (((x85 ^ x33) ^ (x3 + x87)) - ((x88 - x20) + (0.004457 + x23))))))
                Best expression (original format) = x20 x71 + x20 -0.29813 + * x81 1414.708600 ^ x61 + / x89 0.001691 + x70 ^ x55 x68 / x21 x101 + + + + x73 x29 + x89 x71 + - x39 x93 ^ x53 x100 + + + x85 x33 ^ x3 x87 + ^ x88 x20 - 0.004457 x23 + + - / - sech
         Depth = 7:
            Training:
                Best score = 2.28184e-09, SNE = 4.38242e+08
                Squared-norm error for each equation: 4.38242e+08
                Best expression = sech(((((((x20 / x85) + (0.000062 + x71)) * ((x90 * 6.296075) + -0.29813)) / ((0.35284360433905765 / (154956.920000 - x40)) + x61)) + ((((0.601569 ^ x100) + 0.001691) ^ (-0.00016173816055613588 + x70)) + ((x55 / (0.288605 + x68)) + ((x9 ^ x29) + x101)))) - (((((0.001972 + x73) + (0.004457 + x29)) - ((x39 * 0.000062) + x71)) + ((x39 ^ x93) + ((0.011783 + x53) + (x88 + x100)))) / (((x79 ^ (x11 / 0.310902)) ^ ((x68 / 6.296075) + x87)) - ((0.004496 ^ (x66 * x101)) + (0.004023 + x23))))))
                Best expression (original format) = x20 x85 / 0.000062 x71 + + x90 6.296075 * -0.29813 + * 0.35284360433905765 154956.920000 x40 - / x61 + / 0.601569 x100 ^ 0.001691 + -0.00016173816055613588 x70 + ^ x55 0.288605 x68 + / x9 x29 ^ x101 + + + + 0.001972 x73 + 0.004457 x29 + + x39 0.000062 * x71 + - x39 x93 ^ 0.011783 x53 + x88 x100 + + + + x79 x11 0.310902 / ^ x68 6.296075 / x87 + ^ 0.004496 x66 x101 * ^ 0.004023 x23 + + - / - sech
            Validation:
                Best score = 0, SNE = 1.79769e+308
                Squared-norm error for each equation: nan
                Best expression = sech((((((x20 + x71) * (x20 + -0.29813)) / ((x81 ^ 1414.7086) + x61)) + (((x89 + 0.001691) ^ x70) + ((x55 / x68) + (x21 + x101)))) - ((((x73 + x29) - (x89 + x71)) + ((x39 ^ x93) + (x53 + x100))) / (((x85 ^ x33) ^ (x3 + x87)) - ((x88 - x20) + (0.004457 + x23))))))
                Best expression (original format) = x20 x71 + x20 -0.29813 + * x81 1414.7086 ^ x61 + / x89 0.001691 + x70 ^ x55 x68 / x21 x101 + + + + x73 x29 + x89 x71 + - x39 x93 ^ x53 x100 + + + x85 x33 ^ x3 x87 + ^ x88 x20 - 0.004457 x23 + + - / - sech
                Best diff result = abs((sech((((((x20 + x71) * (x20 + -0.29813)) / ((x81 ^ 1414.7086) + x61)) + (((x89 + 0.001691) ^ x70) + ((x55 / x68) + (x21 + x101)))) - ((((x73 + x29) - (x89 + x71)) + ((x39 ^ x93) + (x53 + x100))) / (((x85 ^ x33) ^ (x3 + x87)) - ((x88 - x20) + (0.004457 + x23)))))) - x102))
                Best expression (original format) = x20 x71 + x20 -0.29813 + * x81 1414.7086 ^ x61 + / x89 0.001691 + x70 ^ x55 x68 / x21 x101 + + + + x73 x29 + x89 x71 + - x39 x93 ^ x53 x100 + + + x85 x33 ^ x3 x87 + ^ x88 x20 - 0.004457 x23 + + - / - sech x102 - abs
     With Laplacian Smoothing:
        Training:
            Best score = 2.281e-09, SNE = 4.38404e+08
            Squared-norm error for each equation: 4.38403e+08 2.7307e-10 1182.15
            Best expression = (((x93 + 1) + ((x97 - (x57 ^ (-0.654108 * x18))) + -2395.18428)) / (((((-0.837968 * (x24 ^ -1.162915)) + 2838.0147548380914) - (x71 - (35.574783 - (374.000000 / x60)))) + (193.559570 - (2431.484400 ^ x73))) - ((((x63 / -1.386089) + x18) + 2.063746) * (((154956.920000 / (x93 / -1.878656)) + x100) - ((12187.586 ^ x28) + (x101 + -14640.288932156))))))
            Best expression (original format) = x93 1 + x97 x57 -0.654108 x18 * ^ - -2395.18428 + + -0.837968 x24 -1.162915 ^ * 2838.0147548380914 + x71 35.574783 374.000000 x60 / - - - 193.559570 2431.484400 x73 ^ - + x63 -1.386089 / x18 + 2.063746 + 154956.920000 x93 -1.878656 / / x100 + 12187.586 x28 ^ x101 -14640.288932156 + + - * - /
             Best differential equation parameters = {}
             Best expression parameters = {}
        Validation:
            Best score = 6.7406339e-11, SNE = 1.48354e+10
            Squared-norm error for each equation: 1.48354e+10 2.14516e-07
            Best expression = ((((((-4.186665 / (0.418842 * x35)) + ((x77 * x62) / (0.980140 / x41))) + (((292.000000 ^ x25) + (18.875944 - x3)) + (5.876488 / (x28 * 0.289593)))) * (6.730804900000001e-05 / ((-0.010155448521933261 + x91) - ((x22 + 5.886818) * (x61 - x70))))) + ((((3.2810369607244607 - (-1.279780 * x74)) + (-0.368644 - (x83 * 0.823570))) * (((x94 - x3) + 35.21565023391354) / ((1.620943 + x9) / (x2 + -1.257003)))) + 0.0031050203787341656)) * ((((((x6 - 1.486383) - x61) + ((0.002144 + x78) ^ (x82 - 4))) + (((x54 / x37) + x86) * ((x2 + 4) ^ (x67 + -0.995703)))) + ((((x73 + x85) + (0.398685 + x75)) / ((x84 + x61) - (-0.837968 / x78))) + (((0.001595 + x70) + (1.620244 / x12)) + ((-0.152212 + x47) + 3.910026)))) * (((((0.000684 + x21) + (x81 + -1.358924)) + (-1.186665 - (-1.358924 / x78))) + ((2.5738023662116807 + (6.912820 ^ x77)) + ((x79 + 3.991361) + (x75 / 4.300312)))) - (((1.5687453253569374 - ~(x101)) + 354.49598) * (((0.974754 * x65) - (x47 / 1.011502)) / ((x43 + 72.902766) - (x100 / 45.000000)))))))
            Best expression (original format) = -4.186665 0.418842 x35 * / x77 x62 * 0.980140 x41 / / + 292.000000 x25 ^ 18.875944 x3 - + 5.876488 x28 0.289593 * / + + 6.730804900000001e-05 -0.010155448521933261 x91 + x22 5.886818 + x61 x70 - * - / * 3.2810369607244607 -1.279780 x74 * - -0.368644 x83 0.823570 * - + x94 x3 - 35.21565023391354 + 1.620943 x9 + x2 -1.257003 + / / * 0.0031050203787341656 + + x6 1.486383 - x61 - 0.002144 x78 + x82 4 - ^ + x54 x37 / x86 + x2 4 + x67 -0.995703 + ^ * + x73 x85 + 0.398685 x75 + + x84 x61 + -0.837968 x78 / - / 0.001595 x70 + 1.620244 x12 / + -0.152212 x47 + 3.910026 + + + + 0.000684 x21 + x81 -1.358924 + + -1.186665 -1.358924 x78 / - + 2.5738023662116807 6.912820 x77 ^ + x79 3.991361 + x75 4.300312 / + + + 1.5687453253569374 x101 ~ - 354.49598 + 0.974754 x65 * x47 1.011502 / - x43 72.902766 + x100 45.000000 / - / * - * *
     With Laplacian Smoothing and Wind-Alignment:
        Training:
            Best score = 2.28088e-09, SNE = 4.38428e+08
            Squared-norm error for each equation: 4.38428e+08 2.04899e-28 0.68969
            Best expression = (((((ln(cos(x46)) + -3.4799761065034414) + ((x95 * 1.620943) + (x95 + ~(x48)))) * ~(((cos(x41) + 52.64009483497598) ^ (2.7907071011403315 - cos(x93))))) + ((((3.1585732538600397 * (x12 ^ x54)) + (cos(x21) + (0.365382 + x13))) ^ (~(x20) + 1.5729403267948965)) + (x3 * sqrt(x48)))) / ((((0.9867622178470573 - ((7169.463400 - x100) - (x17 * 11181230.000000))) / (acos(tanh(x23)) + 57.67636600070402)) + sin((-2.884980 + x50))) - (sqrt(((x17 ^ x59) ^ (x71 + ~(x87)))) - (0.9910929232006058 ^ (-4.0405169999999995 * ((x61 - x101) * -0.04344899097047564))))))
            Best expression (original format) = / + * + + ln cos x46 -3.4799761065034414 + * x95 1.620943 + x95 ~ x48 ~ ^ + cos x41 52.64009483497598 - 2.7907071011403315 cos x93 + ^ + * 3.1585732538600397 ^ x12 x54 + cos x21 + 0.365382 x13 + ~ x20 1.5729403267948965 * x3 sqrt x48 - + / - 0.9867622178470573 - - 7169.463400 x100 * x17 11181230.000000 + acos tanh x23 57.67636600070402 sin + -2.884980 x50 - sqrt ^ ^ x17 x59 + x71 ~ x87 ^ 0.9910929232006058 * -4.0405169999999995 * - x61 x101 -0.04344899097047564
            With wind-alignment factor from 1 -> 1e6:
                Best score = 2.28088e-09, SNE = 4.38428e+08
                Squared-norm error for each equation: 4.38428e+08 2.04899e-28 0.68969
                Best expression = (((((ln(cos(x46)) + -3.4799761065034414) + ((x95 * 1.620943) + (x95 + ~(x48)))) * ~(((cos(x41) + 52.64009483497598) ^ (2.7907071011403315 - cos(x93))))) + ((((3.1585732538600397 * (x12 ^ x54)) + (cos(x21) + (0.365382 + x13))) ^ (~(x20) + 1.5729403267948965)) + (x3 * sqrt(x48)))) / ((((0.9867622178470573 - ((7169.463400 - x100) - (x17 * 11181230.000000))) / (acos(tanh(x23)) + 57.67636600070402)) + sin((-2.884980 + x50))) - (sqrt(((x17 ^ x59) ^ (x71 + ~(x87)))) - (0.9910929232006058 ^ (-4.0405169999999995 * ((x61 - x101) * -0.04344899097047564))))))
                Best expression (original format) = / + * + + ln cos x46 -3.4799761065034414 + * x95 1.620943 + x95 ~ x48 ~ ^ + cos x41 52.64009483497598 - 2.7907071011403315 cos x93 + ^ + * 3.1585732538600397 ^ x12 x54 + cos x21 + 0.365382 x13 + ~ x20 1.5729403267948965 * x3 sqrt x48 - + / - 0.9867622178470573 - - 7169.463400 x100 * x17 11181230.000000 + acos tanh x23 57.67636600070402 sin + -2.884980 x50 - sqrt ^ ^ x17 x59 + x71 ~ x87 ^ 0.9910929232006058 * -4.0405169999999995 * - x61 x101 -0.04344899097047564
        Validation:
            Best score = 6.74666e-11, SNE = 1.48221e+10
            Squared-norm error for each equation: 1.48221e+10 6.38901e-08 22.6177
            Best expression = ((((x71 * x69) + (0.82357 ^ (335 / (0.01872 + x31)))) + (((0.496312 + (x67 * -0.6901253057600001)) + 6.2216003257007526e-24) + ((-1.278025 / x14) + -2431.4844))) / (((2838.0127038380915 - (-0.286767 + x76)) + 144.49968) - ((x18 + 2.063746) * (x100 - ((12187.586 ^ x28) + (x101 + -14640.288932156))))))
            Best expression (original format) = x71 x69 * 0.82357 335 0.01872 x31 + / ^ + 0.496312 x67 -0.6901253057600001 * + 6.2216003257007526e-24 + -1.278025 x14 / -2431.4844 + + + 2838.0127038380915 -0.286767 x76 + - 144.49968 + x18 2.063746 + x100 12187.586 x28 ^ x101 -14640.288932156 + + - * - /
     */
    std::vector<std::vector<std::string>> results(x.num_diff_eqns);
    assert(x.num_diff_eqns == 3);
    for (std::vector<std::string>& res: results)
    {
        res.reserve(100);
    }
    constexpr const char* s = "1";
    constexpr const char* tau = "0.1";
    
    thread_local bool prefactors_computed = false;
    thread_local std::vector<std::string> dfdx100, dfdx101, temp;
    thread_local std::vector<int> grasp;
    grasp.clear();
    grasp.reserve(100);
    temp.clear();
    temp.reserve(100);
    dfdx100.clear();
    dfdx100.reserve(100);
    dfdx101.clear();
    dfdx101.reserve(100);
    
    if (x.expression_type == "prefix")
    {
        //abs - f x102
        results[0].push_back("abs");
        results[0].push_back("-");
        for (const std::string& i: x.pieces[0])
        {
            results[0].push_back(i);
        }
        results[0].push_back("x102");
        
        //+ ∂^2f/∂(x100)^2 ∂^2f/∂(x101)^2
        results[1].push_back("+"); // +
        x.derivePrefix(0, x.pieces[0].size()-1, "x100", x.pieces[0], grasp);
        dfdx100 = x.derivat;
        x.derivePrefix(0, dfdx100.size()-1, "x100", dfdx100, grasp);
        for (const std::string& i: x.derivat) // ∂^2f/∂(x100)^2
        {
            results[1].push_back(i);
        }
        x.derivePrefix(0, x.pieces[0].size()-1, "x101", x.pieces[0], grasp);
        dfdx101 = x.derivat;
        x.derivePrefix(0, dfdx101.size()-1, "x101", dfdx101, grasp);
        for (const std::string& i: x.derivat) // ∂^2f/∂(x101)^2
        {
            results[1].push_back(i);
        }
        
        // / * x30 * x24 s + x30 tau
        if (!prefactors_computed)
        {
            x.subs_dict["prefac"] = x.expression_evaluator(x.params, std::vector<std::string>{"/", "*", "x30", "*", "x24", s, "+", "x30", tau});
        }
        // * * prefac * p - 1 p + * x28 ∂f/∂(x100) * x29 ∂f/∂(x101)
        results[2].push_back("*"); // *
        results[2].push_back("*"); // *
        results[2].push_back("prefac"); // prefac
        if (fit)
        {
            // * / 1 - 1 exp ~ f - 1 / 1 - 1 exp ~ f
            results[2].push_back("*"); // *
            results[2].push_back("/"); // /
            results[2].push_back("1"); // 1
            results[2].push_back("-"); // -
            results[2].push_back("1"); // 1
            results[2].push_back("exp"); // exp
            results[2].push_back("~"); // ~
            for (const std::string& i: x.pieces[0]) // f
            {
                results[2].push_back(i);
            }
            results[2].push_back("-"); // -
            results[2].push_back("1"); // 1
            results[2].push_back("/"); // /
            results[2].push_back("1"); // 1
            results[2].push_back("-"); // -
            results[2].push_back("1"); // 1
            results[2].push_back("exp"); // exp
            results[2].push_back("~"); // ~
            for (const std::string& i: x.pieces[0]) // f
            {
                results[2].push_back(i);
            }
        }
        else
        {
            //* p - 1 p
            //p = (/ 1 - 1 exp ~ f)
            temp = {"/", "1", "-", "1", "exp", "~"}; // / 1 - 1 exp ~
            for (const std::string& i: x.pieces[0]) //f
            {
                temp.push_back(i);
            }
            x.subs_dict["p"] = x.expression_evaluator(x.params, temp);
            results[2].push_back("*"); // *
            results[2].push_back("p"); // p
            results[2].push_back("-"); // -
            results[2].push_back("1"); // 1
            results[2].push_back("p"); // p
        }
        // + * x28 ∂f/∂(x100) * x29 ∂f/∂(x101)
        results[2].push_back("*"); // *
        results[2].push_back("1000000"); // 1000000
        
        results[2].push_back("+"); // +
        results[2].push_back("*"); // *
        results[2].push_back("x28"); // x28
        for (const std::string& i: dfdx100) //∂f/∂(x100)
        {
            results[2].push_back(i);
        }
        results[2].push_back("*"); // *
        results[2].push_back("x29"); // x29
        for (const std::string& i: dfdx101) //∂f/∂(x101)
        {
            results[2].push_back(i);
        }
    }
    else if (x.expression_type == "postfix")
    {
        //f x102 - abs
        for (const std::string& i: x.pieces[0])
        {
            results[0].push_back(i);
        }
        results[0].push_back("x102");
        results[0].push_back("-");
        results[0].push_back("abs");
        
        //∂^2f/∂(x100)^2 ∂^2f/∂(x101)^2 +
        x.derivePostfix(0, x.pieces[0].size()-1, "x100", x.pieces[0], grasp);
        dfdx100 = x.derivat;
        x.derivePostfix(0, dfdx100.size()-1, "x100", dfdx100, grasp);
        for (const std::string& i: x.derivat) // ∂^2f/∂(x100)^2
        {
            results[1].push_back(i);
        }
        x.derivePostfix(0, x.pieces[0].size()-1, "x101", x.pieces[0], grasp);
        dfdx101 = x.derivat;
        x.derivePostfix(0, dfdx101.size()-1, "x101", dfdx101, grasp);
        for (const std::string& i: x.derivat) // ∂^2f/∂(x101)^2
        {
            results[1].push_back(i);
        }
        results[1].push_back("+"); // +
        // x30 x24 s * * x30 tau + /
        if (!prefactors_computed)
        {
            x.subs_dict["prefac"] = x.expression_evaluator(x.params, std::vector<std::string>{"x30", "x24", s, "*", "*", "x30", tau, "+", "/"});
        }
        // prefac p 1 p - * * x28 ∂f/∂(x100) * x29 ∂f/∂(x101) * + *
        results[2].push_back("prefac"); // prefac
        if (fit)
        {
            // 1 1 f ~ exp - / 1 1 1 f ~ exp - / - *
            results[2].push_back("1"); // 1
            results[2].push_back("1"); // 1
            for (const std::string& i: x.pieces[0]) // f
            {
                results[2].push_back(i);
            }
            results[2].push_back("~"); // ~
            results[2].push_back("exp"); // exp
            results[2].push_back("-"); // -
            results[2].push_back("/"); // /
            results[2].push_back("1"); // 1
            results[2].push_back("1"); // 1
            results[2].push_back("1"); // 1
            for (const std::string& i: x.pieces[0]) // f
            {
                results[2].push_back(i);
            }
            results[2].push_back("~"); // ~
            results[2].push_back("exp"); // exp
            results[2].push_back("-"); // -
            results[2].push_back("/"); // /
            results[2].push_back("-"); // -
            results[2].push_back("*"); // *
        }
        else
        {
            // p 1 p - *
            
            // 1 1 f ~ exp - /
            temp = {"1", "1"}; // 1 1
            for (const std::string& i: x.pieces[0]) // f
            {
                temp.push_back(i);
            }
            temp.push_back("~"); // ~
            temp.push_back("exp"); // exp
            temp.push_back("-"); // -
            temp.push_back("/"); // /
            
            x.subs_dict["p"] = x.expression_evaluator(x.params, temp);
            results[2].push_back("p"); // p
            results[2].push_back("1"); // 1
            results[2].push_back("p"); // p
            results[2].push_back("-"); // -
            results[2].push_back("*"); // *
        }
        // * x28 ∂f/∂(x100) * x29 ∂f/∂(x101) * + *
        results[2].push_back("*"); // *
        results[2].push_back("x28"); // x28
        for (const std::string& i: dfdx100) // ∂f/∂(x100)
        {
            results[2].push_back(i);
        }
        results[2].push_back("*"); // *
        results[2].push_back("x29"); // x29
        for (const std::string& i: dfdx101) // ∂f/∂(x101)
        {
            results[2].push_back(i);
        }
        results[2].push_back("*"); // *
        results[2].push_back("+"); // +
        results[2].push_back("*"); // *
    }
    prefactors_computed = true;
    return results;
}

/*
 Infix: w1*x26*log(eps + 1/(1-exp(-f))) + w0*(1-x26)*log(eps+1-1/(1-exp(-f))), ∂^2f/∂(x23)^2 + ∂^2f/∂(x24)^2, p*(1-p)*(∂f/∂(x25) + u_x*∂f/∂(x23) + u_y*∂f/∂(x24))
 Prefix: + * * w1 x26 log + / 1 - 1 exp ~ f eps * * w0 - 1 x26 log + - 1 / 1 - 1 exp ~ f eps, + ∂^2f/∂(x23)^2 ∂^2f/∂(x24)^2, * * p - 1 p + ∂f/∂(x25) + * u_x ∂f/∂(x23) * u_y ∂f/∂(x24)
 Postfix: w1 x26 * 1 1 f ~ exp - / eps + log * w0 1 x26 - * 1 1 1 f ~ exp - / - eps + log * +, ∂^2f/∂(x23)^2 ∂^2f/∂(x24)^2 +, p 1 p - * ∂f/∂(x25) u_x ∂f/∂(x23) * u_y ∂f/∂(x24) * + + *
 
 // {x0: VIIRS_band_M11, x1: VIIRS_band_I2, x2: VIIRS_band_I1, x3: NDVI, x4: EVI2, x5: total_precipitation, x6: wind_speed, x7: wind_direction, x8: min_temperature, x9: max_temperature, x10: energy_release_component, x11: specific_humidity, x12: slope, x13: aspect, x14: elevation, x15: palmer_drought_severity_index, x16: landcover_class, x17: forecast_total_precipitation, x18: forecast_wind_speed, x19: forecast_wind_direction, x20: forecast_temperature, x21: forecast_specific_humidity, x22: active_fire, x23: row, x24: col, x25: date, x26: next_day_active_fire_bin}
 // Want to predict x26 given {x0, x1, ..., x25}
 // {x.pieces[0]: f}
 */
std::vector<std::vector<std::string>> WildfireSpreadTS(Board& x, bool fit)
{
    std::vector<std::vector<std::string>> results(x.num_diff_eqns); //For now, simply comparing 𝛔(f(\vec{x})) with x
    assert(x.num_diff_eqns == 3);
    for (std::vector<std::string>& res: results)
    {
        res.reserve(100);
    }
    thread_local std::vector<std::string> p_expr, dfdx23, dfdx24;
    thread_local std::vector<int> grasp;
    thread_local bool prefactors_computed = false;
    constexpr const char* eps = "1e-12";
    const thread_local double num_ones = x["x26"].sum(); //since x26 (i.e. `next_day_active_fire_bin`) is just a vector of 0's and 1's
    const thread_local double num_zeroes = Board::data.num_rows - num_ones;
    thread_local const std::string w1 = to_string_general(-Board::data.num_rows / (2.0 * num_ones));
    thread_local const std::string w0 = to_string_general(-Board::data.num_rows / (2.0 * num_zeroes));
    
    /*
     Without Laplacian Smoothing:
         Best score = 4.16439e-06, SNE = 240130
         Squared-norm error for each equation: 240130
         Best expression = (((((9744.788019575928 + (((-2.640000 / x13) ^ 783) / x13)) - ((((x9 ^ x20) / (15666.000000 ^ x6)) * x18) ^ cos((-0.6400000000000001 + x6)))) - ((0.002104996890902969 * ((~(x17) + x0) - (9.666708 + (8.800000 + x10)))) ^ ((59 + (log(x8) + x7)) - (x11 ^ ((x11 + x5) + x15))))) + ((x25 - 3.4288275429960554e+302) / (((((-3.225653 + x1) - 336) ^ -100) + (0.9184389999999991 ^ (x18 ^ (292.000000 - x8)))) ^ ((((x13 + x22) / 9736) + x18) - ((0.00621 / x20) + 2))))) + (((((((0.004735 / x20) + x6) / 2.176586002694007) ^ ((402.077009 + (38.000000 + x0)) - ((x22 * x24) + x2))) + ((364.72436670687574 ^ ((x17 + 36.293228) - x10)) + ((x5 * 6075) + x20))) + ((((9736 / exp(x23)) + x17) / ((127.669719 + (x22 + x22)) * x11)) ^ (((x19 / -12.04) + ((x22 / 292.000000) + 93.9525010000761)) - (((x7 - 38.000000) / (x19 + 9.666708)) + (-1.2400589896061356 + x23))))) * ((((x19 + (x20 + 0.00621)) * 28.90232091662481) + ((((x14 + x8) + 9.183585634667363) * (x22 / -100)) + ((log(x7) + 2.868271954674221) + -768.0736260000001))) + (((435 + x8) / (((x21 - x19) + x11) + (38 - (x23 + -88.959518)))) + ((x6 + acos(sin(x13))) + -18290.143616999998)))))
         Best expression (original format) = 9744.788019575928 -2.640000 x13 / 783 ^ x13 / + x9 x20 ^ 15666.000000 x6 ^ / x18 * -0.6400000000000001 x6 + cos ^ - 0.002104996890902969 x17 ~ x0 + 9.666708 8.800000 x10 + + - * 59 x8 log x7 + + x11 x11 x5 + x15 + ^ - ^ - x25 3.4288275429960554e+302 - -3.225653 x1 + 336 - -100 ^ 0.9184389999999991 x18 292.000000 x8 - ^ ^ + x13 x22 + 9736 / x18 + 0.00621 x20 / 2 + - ^ / + 0.004735 x20 / x6 + 2.176586002694007 / 402.077009 38.000000 x0 + + x22 x24 * x2 + - ^ 364.72436670687574 x17 36.293228 + x10 - ^ x5 6075 * x20 + + + 9736 x23 exp / x17 + 127.669719 x22 x22 + + x11 * / x19 -12.04 / x22 292.000000 / 93.9525010000761 + + x7 38.000000 - x19 9.666708 + / -1.2400589896061356 x23 + + - ^ + x19 x20 0.00621 + + 28.90232091662481 * x14 x8 + 9.183585634667363 + x22 -100 / * x7 log 2.868271954674221 + -768.0736260000001 + + + 435 x8 + x21 x19 - x11 + 38 x23 -88.959518 + - + / x6 x13 sin acos + -18290.143616999998 + + + * +
         Best diff result = (term1 + term2)
         Best expression (original format) = term1 term2 +
         ```
            from sympy import symbols, cos, sin, tanh, sech, acos, log, sympify, latex, multiline_latex, Float
            import re
            replace_vars = lambda x: re.sub(r'\bx(\d+)\b', r'df["x\1"]', x)
            align_rep = lambda x: x.replace('align*','align').replace(r'\\',r'\nonumber \\').replace(r"\end{align}", r"\label{eq:best_sr_eq_1}""\n"r"\end{align}")
            round_floats = lambda expr, ndigits: expr.xreplace({f: Float(round(float(f), ndigits)) for f in expr.atoms(Float)})
            f, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25 = symbols('f x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25')
            func = '((((-2.640000 / (0.051731 / (sqrt(x7) - (x15 - 8.000000)))) - ((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) / ((x6 / (x19 - (-100.000000 + x8))) + ((x14 / -211800) + x15)))) + ((x18 - x24) - (x2 + (x7 - (x0 - 1684.200012))))) * (((-7.446376466569234 + (((-3.225653 + (x6 - 8.800000)) + -508) + (-0.9081765689798138 * (38.000000 / sin(x16))))) + (((((x1 - x0) + 0.0007699998478223693) + -16.82119949898502) + (((x0 ^ x15) + -100) + (-279.200012 + (-2.640000 / x16)))) + -28.995355508740936)) + ((((-88.959518 * (1.000000 / (88.856491 / x1))) + (-2118 + ((x8 - 2118.000000) + ~(x22)))) * (0.00077 + x5)) - (((25.400000 ^ (x20 / 1.0021072170678698)) + ((15893.000000 + (x22 + x8)) + ((x1 - -1405.000000) + (15666.000000 + x24)))) / ~(((x0 + x25) / ((16.000000 + x23) - 9736)))))))'
            func = func.replace("^","**").replace("~","-")
            func_sym = sympify(func)
            func_sym_r = round_floats(func_sym, 3)
            print(f"func_sym = {align_rep(multiline_latex(f, func_sym_r, 1))}")
            f_res = replace_vars(func)
            print(f"f = {f_res}")
         ```
     With Laplacian Smoothing:
        Training:
            Best score = 8.21789e-08, SNE = 1.21686e+07
            Squared-norm error for each equation: 1.21686e+07 0
            Best expression = ((58 + ((((2.520000 * x2) + 5470.0596345781605) + (3303.5714033364225 ^ (1.320645006282e+07 - (ln(x17) * (x21 - -843.000000))))) + (x8 / (x17 * 0.0007869063691834137)))) * (((((x18 ^ (20200128.000000 - (x7 ^ x7))) + ((x24 + (x15 ^ 16.000000)) + (x16 + (15893.000000 ^ x20)))) + ((0.002105 ^ (-459.704071 - (359.704071 - x2))) + (58.000000 * (x7 + (0.006210 + x12))))) * ((((~(x22) + -88.959518) / (88.856491 * (x8 - 8.000000))) + ((x3 / -2.0200127e+07) + -1)) * (((x22 / (3316.000000 ^ x19)) + (sech(x19) + x11)) * ((0.12004407415282907 + (x14 / x8)) + ((x22 + 39.3125) ^ (0.000770 + x20)))))) - ((((-2.225653 + (x20 / 0.2068956670668029)) + ((0.051731 / (x20 / x6)) + (x20 + x23))) - (((x5 * (x0 + x22)) + ((0.002105 / x11) + 1.5707963267948966)) + (((256.500000 ^ x15) + x17) + (4.00621 + x24)))) * (((((x20 * 360.000000) + -15858.185171) + (360.000000 + (-2.640000 * x7))) / ((8.051731 + ln(x6)) + ((8.000000 - x18) + -781.0930331488324))) + (((x10 * (x16 / x14)) + ((x20 / x15) + 1.7112288116110355)) + (((58.000000 / x1) + 36.310283) + ((0.000770 + x18) + -73.0701029874863)))))))
            Best expression (original format) = 58 2.520000 x2 * 5470.0596345781605 + 3303.5714033364225 1.320645006282e+07 x17 ln x21 -843.000000 - * - ^ + x8 x17 0.0007869063691834137 * / + + x18 20200128.000000 x7 x7 ^ - ^ x24 x15 16.000000 ^ + x16 15893.000000 x20 ^ + + + 0.002105 -459.704071 359.704071 x2 - - ^ 58.000000 x7 0.006210 x12 + + * + + x22 ~ -88.959518 + 88.856491 x8 8.000000 - * / x3 -2.0200127e+07 / -1 + + x22 3316.000000 x19 ^ / x19 sech x11 + + 0.12004407415282907 x14 x8 / + x22 39.3125 + 0.000770 x20 + ^ + * * * -2.225653 x20 0.2068956670668029 / + 0.051731 x20 x6 / / x20 x23 + + + x5 x0 x22 + * 0.002105 x11 / 1.5707963267948966 + + 256.500000 x15 ^ x17 + 4.00621 x24 + + + - x20 360.000000 * -15858.185171 + 360.000000 -2.640000 x7 * + + 8.051731 x6 ln + 8.000000 x18 - -781.0930331488324 + + / x10 x16 x14 / * x20 x15 / 1.7112288116110355 + + 58.000000 x1 / 36.310283 + 0.000770 x18 + -73.0701029874863 + + + + * - *
            Best diff result = (term1 + term2), 0
            Best expression (original format) = term1 term2 +, 0
        Validation:
            Best score = 7.8752e-12, SNE = 1.26981e+11
            Squared-norm error for each equation: 1.26981e+11 0
            Best expression = ((58 + ((((2.520000 * x2) + 5470.0596345781605) + (3303.5714033364225 ^ (1.320645006282e+07 - (ln(x17) * (x21 - -843.000000))))) + (x8 / (x17 * 0.0007869063691834137)))) * (((((x18 ^ (20200128.000000 - (x7 ^ x7))) + ((x24 + (x15 ^ 16.000000)) + (x16 + (15893.000000 ^ x20)))) + ((0.002105 ^ (-459.704071 - (359.704071 - x2))) + (58.000000 * (x7 + (0.006210 + x12))))) * ((((~(x22) + -88.959518) / (88.856491 * (x8 - 8.000000))) + ((x3 / -2.0200127e+07) + -1)) * (((x22 / (3316.000000 ^ x19)) + (sech(x19) + x11)) * ((0.12004407415282907 + (x14 / x8)) + ((x22 + 39.3125) ^ (0.000770 + x20)))))) - ((((-2.225653 + (x20 / 0.2068956670668029)) + ((0.051731 / (x20 / x6)) + (x20 + x23))) - (((x5 * (x0 + x22)) + ((0.002105 / x11) + 1.5707963267948966)) + (((256.500000 ^ x15) + x17) + (4.00621 + x24)))) * (((((x20 * 360.000000) + -15858.185171) + (360.000000 + (-2.640000 * x7))) / ((8.051731 + ln(x6)) + ((8.000000 - x18) + -781.0930331488324))) + (((x10 * (x16 / x14)) + ((x20 / x15) + 1.7112288116110355)) + (((58.000000 / x1) + 36.310283) + ((0.000770 + x18) + -73.0701029874863)))))))
            Best expression (original format) = 58 2.520000 x2 * 5470.0596345781605 + 3303.5714033364225 1.320645006282e+07 x17 ln x21 -843.000000 - * - ^ + x8 x17 0.0007869063691834137 * / + + x18 20200128.000000 x7 x7 ^ - ^ x24 x15 16.000000 ^ + x16 15893.000000 x20 ^ + + + 0.002105 -459.704071 359.704071 x2 - - ^ 58.000000 x7 0.006210 x12 + + * + + x22 ~ -88.959518 + 88.856491 x8 8.000000 - * / x3 -2.0200127e+07 / -1 + + x22 3316.000000 x19 ^ / x19 sech x11 + + 0.12004407415282907 x14 x8 / + x22 39.3125 + 0.000770 x20 + ^ + * * * -2.225653 x20 0.2068956670668029 / + 0.051731 x20 x6 / / x20 x23 + + + x5 x0 x22 + * 0.002105 x11 / 1.5707963267948966 + + 256.500000 x15 ^ x17 + 4.00621 x24 + + + - x20 360.000000 * -15858.185171 + 360.000000 -2.640000 x7 * + + 8.051731 x6 ln + 8.000000 x18 - -781.0930331488324 + + / x10 x16 x14 / * x20 x15 / 1.7112288116110355 + + 58.000000 x1 / 36.310283 + 0.000770 x18 + -73.0701029874863 + + + + * - *
            Best diff result = (term1 + term2), 0
            Best expression (original format) = term1 term2 +, 0
     With Laplacian Smoothing and Wind-Alignment:
        Training:
            Best score = 8.22255e-09, SNE = 1.21617e+08
            Squared-norm error for each equation: 1.21617e+08 2.46555 4.6049e-24
            Best expression = ((((-2.640000 / (0.051731 / (sqrt(x7) - (x15 - 8.000000)))) - ((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) / ((x6 / (x19 - (-100.000000 + x8))) + ((x14 / -211800) + x15)))) + ((x18 - x24) - (x2 + (x7 - (x0 - 1684.200012))))) * (((-7.446376466569234 + (((-3.225653 + (x6 - 8.800000)) + -508) + (-0.9081765689798138 * (38.000000 / sin(x16))))) + (((((x1 - x0) + 0.0007699998478223693) + -16.82119949898502) + (((x0 ^ x15) + -100) + (-279.200012 + (-2.640000 / x16)))) + -28.995355508740936)) + ((((-88.959518 * (1.000000 / (88.856491 / x1))) + (-2118 + ((x8 - 2118.000000) + ~(x22)))) * (0.00077 + x5)) - (((25.400000 ^ (x20 / 1.0021072170678698)) + ((15893.000000 + (x22 + x8)) + ((x1 - -1405.000000) + (15666.000000 + x24)))) / ~(((x0 + x25) / ((16.000000 + x23) - 9736)))))))
            Best expression (original format) = -2.640000 0.051731 x7 sqrt x15 8.000000 - - / / -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 x19 -100.000000 x8 + - / x14 -211800 / x15 + + / - x18 x24 - x2 x7 x0 1684.200012 - - + - + -7.446376466569234 -3.225653 x6 8.800000 - + -508 + -0.9081765689798138 38.000000 x16 sin / * + + x1 x0 - 0.0007699998478223693 + -16.82119949898502 + x0 x15 ^ -100 + -279.200012 -2.640000 x16 / + + + -28.995355508740936 + + -88.959518 1.000000 88.856491 x1 / / * -2118 x8 2118.000000 - x22 ~ + + + 0.00077 x5 + * 25.400000 x20 1.0021072170678698 / ^ 15893.000000 x22 x8 + + x1 -1405.000000 - 15666.000000 x24 + + + + x0 x25 + 16.000000 x23 + 9736 - / ~ / - + *
            Best diff result = (term1 + term2), (((((-2.640000 / (0.051731 / (sqrt(x7) - (x15 - 8.000000)))) - ((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) / ((x6 / (x19 - (-100.000000 + x8))) + ((x14 / -211800) + x15)))) + ((x18 - x24) - (x2 + (x7 - (x0 - 1684.200012))))) * ~((((~((((25.400000 ^ (x20 / 1.0021072170678698)) + ((15893.000000 + (x22 + x8)) + ((x1 - -1405.000000) + (15666.000000 + x24)))) * ~((~((~((x0 + x25)) * (((16.000000 + x23) - 9736) + ((16.000000 + x23) - 9736)))) / ((((16.000000 + x23) - 9736) * ((16.000000 + x23) - 9736)) * (((16.000000 + x23) - 9736) * ((16.000000 + x23) - 9736))))))) * (~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736))))) - (~((((25.400000 ^ (x20 / 1.0021072170678698)) + ((15893.000000 + (x22 + x8)) + ((x1 - -1405.000000) + (15666.000000 + x24)))) * ~((~((x0 + x25)) / (((16.000000 + x23) - 9736) * ((16.000000 + x23) - 9736)))))) * ((~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~((~((x0 + x25)) / (((16.000000 + x23) - 9736) * ((16.000000 + x23) - 9736))))) + (~((~((x0 + x25)) / (((16.000000 + x23) - 9736) * ((16.000000 + x23) - 9736)))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736))))))) / ((~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736)))) * (~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736)))))))) + ((~(((x0 + x25) / ((16.000000 + x23) - 9736))) / (~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736))))) + (~(((x0 + x25) / ((16.000000 + x23) - 9736))) / (~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736))))))), ((p * (1 - p)) * (((((-2.640000 / (0.051731 / (sqrt(x7) - (x15 - 8.000000)))) - ((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) / ((x6 / (x19 - (-100.000000 + x8))) + ((x14 / -211800) + x15)))) + ((x18 - x24) - (x2 + (x7 - (x0 - 1684.200012))))) * ~((~((((25.400000 ^ (x20 / 1.0021072170678698)) + ((15893.000000 + (x22 + x8)) + ((x1 - -1405.000000) + (15666.000000 + x24)))) * ~((((16.000000 + x23) - 9736) / (((16.000000 + x23) - 9736) * ((16.000000 + x23) - 9736)))))) / (~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736))))))) + ((u_x * ((((-2.640000 / (0.051731 / (sqrt(x7) - (x15 - 8.000000)))) - ((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) / ((x6 / (x19 - (-100.000000 + x8))) + ((x14 / -211800) + x15)))) + ((x18 - x24) - (x2 + (x7 - (x0 - 1684.200012))))) * ~((~((((25.400000 ^ (x20 / 1.0021072170678698)) + ((15893.000000 + (x22 + x8)) + ((x1 - -1405.000000) + (15666.000000 + x24)))) * ~((~((x0 + x25)) / (((16.000000 + x23) - 9736) * ((16.000000 + x23) - 9736)))))) / (~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736)))))))) + (u_y * (((((-2.640000 / (0.051731 / (sqrt(x7) - (x15 - 8.000000)))) - ((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) / ((x6 / (x19 - (-100.000000 + x8))) + ((x14 / -211800) + x15)))) + ((x18 - x24) - (x2 + (x7 - (x0 - 1684.200012))))) * ~((~(((x0 + x25) / ((16.000000 + x23) - 9736))) / (~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736))))))) + ~((((-7.446376466569234 + (((-3.225653 + (x6 - 8.800000)) + -508) + (-0.9081765689798138 * (38.000000 / sin(x16))))) + (((((x1 - x0) + 0.0007699998478223693) + -16.82119949898502) + (((x0 ^ x15) + -100) + (-279.200012 + (-2.640000 / x16)))) + -28.995355508740936)) + ((((-88.959518 * (1.000000 / (88.856491 / x1))) + (-2118 + ((x8 - 2118.000000) + ~(x22)))) * (0.00077 + x5)) - (((25.400000 ^ (x20 / 1.0021072170678698)) + ((15893.000000 + (x22 + x8)) + ((x1 - -1405.000000) + (15666.000000 + x24)))) / ~(((x0 + x25) / ((16.000000 + x23) - 9736))))))))))))
            Best expression (original format) = term1 term2 +, -2.640000 0.051731 x7 sqrt x15 8.000000 - - / / -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 x19 -100.000000 x8 + - / x14 -211800 / x15 + + / - x18 x24 - x2 x7 x0 1684.200012 - - + - + 25.400000 x20 1.0021072170678698 / ^ 15893.000000 x22 x8 + + x1 -1405.000000 - 15666.000000 x24 + + + + x0 x25 + ~ 16.000000 x23 + 9736 - 16.000000 x23 + 9736 - + * ~ 16.000000 x23 + 9736 - 16.000000 x23 + 9736 - * 16.000000 x23 + 9736 - 16.000000 x23 + 9736 - * * / ~ * ~ x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * * 25.400000 x20 1.0021072170678698 / ^ 15893.000000 x22 x8 + + x1 -1405.000000 - 15666.000000 x24 + + + + x0 x25 + ~ 16.000000 x23 + 9736 - 16.000000 x23 + 9736 - * / ~ * ~ x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + ~ 16.000000 x23 + 9736 - 16.000000 x23 + 9736 - * / ~ * x0 x25 + ~ 16.000000 x23 + 9736 - 16.000000 x23 + 9736 - * / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * + * - x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * * / ~ * x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * / x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * / + +, p 1 p - * -2.640000 0.051731 x7 sqrt x15 8.000000 - - / / -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 x19 -100.000000 x8 + - / x14 -211800 / x15 + + / - x18 x24 - x2 x7 x0 1684.200012 - - + - + 25.400000 x20 1.0021072170678698 / ^ 15893.000000 x22 x8 + + x1 -1405.000000 - 15666.000000 x24 + + + + 16.000000 x23 + 9736 - 16.000000 x23 + 9736 - 16.000000 x23 + 9736 - * / ~ * ~ x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * / ~ * u_x -2.640000 0.051731 x7 sqrt x15 8.000000 - - / / -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 x19 -100.000000 x8 + - / x14 -211800 / x15 + + / - x18 x24 - x2 x7 x0 1684.200012 - - + - + 25.400000 x20 1.0021072170678698 / ^ 15893.000000 x22 x8 + + x1 -1405.000000 - 15666.000000 x24 + + + + x0 x25 + ~ 16.000000 x23 + 9736 - 16.000000 x23 + 9736 - * / ~ * ~ x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * / ~ * * u_y -2.640000 0.051731 x7 sqrt x15 8.000000 - - / / -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 x19 -100.000000 x8 + - / x14 -211800 / x15 + + / - x18 x24 - x2 x7 x0 1684.200012 - - + - + x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * / ~ * -7.446376466569234 -3.225653 x6 8.800000 - + -508 + -0.9081765689798138 38.000000 x16 sin / * + + x1 x0 - 0.0007699998478223693 + -16.82119949898502 + x0 x15 ^ -100 + -279.200012 -2.640000 x16 / + + + -28.995355508740936 + + -88.959518 1.000000 88.856491 x1 / / * -2118 x8 2118.000000 - x22 ~ + + + 0.00077 x5 + * 25.400000 x20 1.0021072170678698 / ^ 15893.000000 x22 x8 + + x1 -1405.000000 - 15666.000000 x24 + + + + x0 x25 + 16.000000 x23 + 9736 - / ~ / - + ~ + * + + *
            Best differential equation parameters = {}
            Best expression parameters = {}
        Validation:
            Best score = 0, SNE = 1.79769e+308
            Squared-norm error for each equation: nan 0 0
            Best expression = ((((-2.640000 / (0.051731 / (sqrt(x7) - (x15 - 8.000000)))) - ((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) / ((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)))) + ((x18 - x24) - (x2 + (x13 - (x0 - 1684.200012))))) * (((-0.997895 + (((x6 - 8.800000) + -508) + (-0.9081765689798138 * (38.000000 / sin(x16))))) + ((((x1 - x0) + -7.154491498985021) + (((x0 ^ x15) + -100) + -279.199242)) + -11.682563967704278)) + ((((-88.959518 * (1.000000 / (88.856491 / x1))) + (-2117 + ((x8 - 2118.000000) + ~(x22)))) * (0.00077 + x5)) - (((25.400000 ^ (x20 / 1.0021072170678698)) + ((x14 * 3.1091329981202156) + ((x1 - -1405.000000) + (15666.000000 + x24)))) / ~(((x0 + x25) / ((16.000000 + x23) - 9736)))))))
            Best expression (original format) = -2.640000 0.051731 x7 sqrt x15 8.000000 - - / / -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + / - x18 x24 - x2 x13 x0 1684.200012 - - + - + -0.997895 x6 8.800000 - -508 + -0.9081765689798138 38.000000 x16 sin / * + + x1 x0 - -7.154491498985021 + x0 x15 ^ -100 + -279.199242 + + -11.682563967704278 + + -88.959518 1.000000 88.856491 x1 / / * -2117 x8 2118.000000 - x22 ~ + + + 0.00077 x5 + * 25.400000 x20 1.0021072170678698 / ^ x14 3.1091329981202156 * x1 -1405.000000 - 15666.000000 x24 + + + + x0 x25 + 16.000000 x23 + 9736 - / ~ / - + *
            Best diff result = (term1 + term2), (((((((-2.640000 / (0.051731 / (sqrt(x7) - (x15 - 8.000000)))) - ((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) / ((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)))) + ((x18 - x24) - (x2 + (x13 - (x0 - 1684.200012))))) * ((~(0) * (0.00077 + x5)) - (((~((((25.400000 ^ (x20 / 1.0021072170678698)) + ((x14 * 3.1091329981202156) + ((x1 - -1405.000000) + (15666.000000 + x24)))) * ~((((~(0) * (((16.000000 + x23) - 9736) * ((16.000000 + x23) - 9736))) - (~((x0 + x25)) * (((16.000000 + x23) - 9736) + ((16.000000 + x23) - 9736)))) / ((((16.000000 + x23) - 9736) * ((16.000000 + x23) - 9736)) * (((16.000000 + x23) - 9736) * ((16.000000 + x23) - 9736))))))) * (~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736))))) - (~((((25.400000 ^ (x20 / 1.0021072170678698)) + ((x14 * 3.1091329981202156) + ((x1 - -1405.000000) + (15666.000000 + x24)))) * ~((~((x0 + x25)) / (((16.000000 + x23) - 9736) * ((16.000000 + x23) - 9736)))))) * ((~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~((~((x0 + x25)) / (((16.000000 + x23) - 9736) * ((16.000000 + x23) - 9736))))) + (~((~((x0 + x25)) / (((16.000000 + x23) - 9736) * ((16.000000 + x23) - 9736)))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736))))))) / ((~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736)))) * (~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736)))))))) + (~((~(((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) * (~((x6 * ~(0))) / (~((-100.000000 + x8)) * ~((-100.000000 + x8)))))) / (((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)) * ((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15))))) * ((~(0) * (0.00077 + x5)) - (~((((25.400000 ^ (x20 / 1.0021072170678698)) + ((x14 * 3.1091329981202156) + ((x1 - -1405.000000) + (15666.000000 + x24)))) * ~((~((x0 + x25)) / (((16.000000 + x23) - 9736) * ((16.000000 + x23) - 9736)))))) / (~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736)))))))) + ((~((~(((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) * (~((x6 * ~(0))) / (~((-100.000000 + x8)) * ~((-100.000000 + x8)))))) / (((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)) * ((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15))))) * ((~(0) * (0.00077 + x5)) - (~((((25.400000 ^ (x20 / 1.0021072170678698)) + ((x14 * 3.1091329981202156) + ((x1 - -1405.000000) + (15666.000000 + x24)))) * ~((~((x0 + x25)) / (((16.000000 + x23) - 9736) * ((16.000000 + x23) - 9736)))))) / (~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736))))))) + (~((((~(((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) * (((~((x6 * ~(0))) * (~((-100.000000 + x8)) * ~((-100.000000 + x8)))) - (~((x6 * ~(0))) * ((~((-100.000000 + x8)) * ~(0)) + (~(0) * ~((-100.000000 + x8)))))) / ((~((-100.000000 + x8)) * ~((-100.000000 + x8))) * (~((-100.000000 + x8)) * ~((-100.000000 + x8))))))) * (((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)) * ((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)))) - (~(((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) * (~((x6 * ~(0))) / (~((-100.000000 + x8)) * ~((-100.000000 + x8)))))) * ((((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)) * (~((x6 * ~(0))) / (~((-100.000000 + x8)) * ~((-100.000000 + x8))))) + ((~((x6 * ~(0))) / (~((-100.000000 + x8)) * ~((-100.000000 + x8)))) * ((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)))))) / ((((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)) * ((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15))) * (((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)) * ((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)))))) * (((-0.997895 + (((x6 - 8.800000) + -508) + (-0.9081765689798138 * (38.000000 / sin(x16))))) + ((((x1 - x0) + -7.154491498985021) + (((x0 ^ x15) + -100) + -279.199242)) + -11.682563967704278)) + ((((-88.959518 * (1.000000 / (88.856491 / x1))) + (-2117 + ((x8 - 2118.000000) + ~(x22)))) * (0.00077 + x5)) - (((25.400000 ^ (x20 / 1.0021072170678698)) + ((x14 * 3.1091329981202156) + ((x1 - -1405.000000) + (15666.000000 + x24)))) / ~(((x0 + x25) / ((16.000000 + x23) - 9736))))))))) + ((((((-2.640000 / (0.051731 / (sqrt(x7) - (x15 - 8.000000)))) - ((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) / ((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)))) + ((x18 - x24) - (x2 + (x13 - (x0 - 1684.200012))))) * ((~(0) * (0.00077 + x5)) - ((((~(0) - ((((25.400000 ^ (x20 / 1.0021072170678698)) + ((x14 * 3.1091329981202156) + ((x1 - -1405.000000) + (15666.000000 + x24)))) * ~(0)) + ~(0))) * (~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736))))) - ((~(((x0 + x25) / ((16.000000 + x23) - 9736))) - (((25.400000 ^ (x20 / 1.0021072170678698)) + ((x14 * 3.1091329981202156) + ((x1 - -1405.000000) + (15666.000000 + x24)))) * ~(0))) * ((~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~(0)) + (~(0) * ~(((x0 + x25) / ((16.000000 + x23) - 9736))))))) / ((~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736)))) * (~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736)))))))) + ((~((~(((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) * (~((x6 * ~(0))) / (~((-100.000000 + x8)) * ~((-100.000000 + x8)))))) / (((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)) * ((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15))))) + ~(1)) * ((~(0) * (0.00077 + x5)) - ((~(((x0 + x25) / ((16.000000 + x23) - 9736))) - (((25.400000 ^ (x20 / 1.0021072170678698)) + ((x14 * 3.1091329981202156) + ((x1 - -1405.000000) + (15666.000000 + x24)))) * ~(0))) / (~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736)))))))) + (((~((~(((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) * (~((x6 * ~(0))) / (~((-100.000000 + x8)) * ~((-100.000000 + x8)))))) / (((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)) * ((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15))))) + ~(1)) * ((~(0) * (0.00077 + x5)) - ((~(((x0 + x25) / ((16.000000 + x23) - 9736))) - (((25.400000 ^ (x20 / 1.0021072170678698)) + ((x14 * 3.1091329981202156) + ((x1 - -1405.000000) + (15666.000000 + x24)))) * ~(0))) / (~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736))))))) + ((~((((~(((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) * (((~((x6 * ~(0))) * (~((-100.000000 + x8)) * ~((-100.000000 + x8)))) - (~((x6 * ~(0))) * ((~((-100.000000 + x8)) * ~(0)) + (~(0) * ~((-100.000000 + x8)))))) / ((~((-100.000000 + x8)) * ~((-100.000000 + x8))) * (~((-100.000000 + x8)) * ~((-100.000000 + x8))))))) * (((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)) * ((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)))) - (~(((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) * (~((x6 * ~(0))) / (~((-100.000000 + x8)) * ~((-100.000000 + x8)))))) * ((((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)) * (~((x6 * ~(0))) / (~((-100.000000 + x8)) * ~((-100.000000 + x8))))) + ((~((x6 * ~(0))) / (~((-100.000000 + x8)) * ~((-100.000000 + x8)))) * ((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)))))) / ((((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)) * ((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15))) * (((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)) * ((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)))))) + ~(0)) * (((-0.997895 + (((x6 - 8.800000) + -508) + (-0.9081765689798138 * (38.000000 / sin(x16))))) + ((((x1 - x0) + -7.154491498985021) + (((x0 ^ x15) + -100) + -279.199242)) + -11.682563967704278)) + ((((-88.959518 * (1.000000 / (88.856491 / x1))) + (-2117 + ((x8 - 2118.000000) + ~(x22)))) * (0.00077 + x5)) - (((25.400000 ^ (x20 / 1.0021072170678698)) + ((x14 * 3.1091329981202156) + ((x1 - -1405.000000) + (15666.000000 + x24)))) / ~(((x0 + x25) / ((16.000000 + x23) - 9736)))))))))), ((p * (1 - p)) * ((((((-2.640000 / (0.051731 / (sqrt(x7) - (x15 - 8.000000)))) - ((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) / ((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)))) + ((x18 - x24) - (x2 + (x13 - (x0 - 1684.200012))))) * ((~(0) * (0.00077 + x5)) - (~((((25.400000 ^ (x20 / 1.0021072170678698)) + ((x14 * 3.1091329981202156) + ((x1 - -1405.000000) + (15666.000000 + x24)))) * ~((((16.000000 + x23) - 9736) / (((16.000000 + x23) - 9736) * ((16.000000 + x23) - 9736)))))) / (~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736))))))) + (~((~(((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) * (~((x6 * ~(0))) / (~((-100.000000 + x8)) * ~((-100.000000 + x8)))))) / (((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)) * ((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15))))) * (((-0.997895 + (((x6 - 8.800000) + -508) + (-0.9081765689798138 * (38.000000 / sin(x16))))) + ((((x1 - x0) + -7.154491498985021) + (((x0 ^ x15) + -100) + -279.199242)) + -11.682563967704278)) + ((((-88.959518 * (1.000000 / (88.856491 / x1))) + (-2117 + ((x8 - 2118.000000) + ~(x22)))) * (0.00077 + x5)) - (((25.400000 ^ (x20 / 1.0021072170678698)) + ((x14 * 3.1091329981202156) + ((x1 - -1405.000000) + (15666.000000 + x24)))) / ~(((x0 + x25) / ((16.000000 + x23) - 9736)))))))) + ((u_x * (((((-2.640000 / (0.051731 / (sqrt(x7) - (x15 - 8.000000)))) - ((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) / ((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)))) + ((x18 - x24) - (x2 + (x13 - (x0 - 1684.200012))))) * ((~(0) * (0.00077 + x5)) - (~((((25.400000 ^ (x20 / 1.0021072170678698)) + ((x14 * 3.1091329981202156) + ((x1 - -1405.000000) + (15666.000000 + x24)))) * ~((~((x0 + x25)) / (((16.000000 + x23) - 9736) * ((16.000000 + x23) - 9736)))))) / (~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736))))))) + (~((~(((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) * (~((x6 * ~(0))) / (~((-100.000000 + x8)) * ~((-100.000000 + x8)))))) / (((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)) * ((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15))))) * (((-0.997895 + (((x6 - 8.800000) + -508) + (-0.9081765689798138 * (38.000000 / sin(x16))))) + ((((x1 - x0) + -7.154491498985021) + (((x0 ^ x15) + -100) + -279.199242)) + -11.682563967704278)) + ((((-88.959518 * (1.000000 / (88.856491 / x1))) + (-2117 + ((x8 - 2118.000000) + ~(x22)))) * (0.00077 + x5)) - (((25.400000 ^ (x20 / 1.0021072170678698)) + ((x14 * 3.1091329981202156) + ((x1 - -1405.000000) + (15666.000000 + x24)))) / ~(((x0 + x25) / ((16.000000 + x23) - 9736))))))))) + (u_y * (((((-2.640000 / (0.051731 / (sqrt(x7) - (x15 - 8.000000)))) - ((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) / ((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)))) + ((x18 - x24) - (x2 + (x13 - (x0 - 1684.200012))))) * ((~(0) * (0.00077 + x5)) - ((~(((x0 + x25) / ((16.000000 + x23) - 9736))) - (((25.400000 ^ (x20 / 1.0021072170678698)) + ((x14 * 3.1091329981202156) + ((x1 - -1405.000000) + (15666.000000 + x24)))) * ~(0))) / (~(((x0 + x25) / ((16.000000 + x23) - 9736))) * ~(((x0 + x25) / ((16.000000 + x23) - 9736))))))) + ((~((~(((-1803.016571 + ((x21 * (36.293228 * x0)) + ((x17 + -843.000000) + (x15 + 15893.000000)))) * (~((x6 * ~(0))) / (~((-100.000000 + x8)) * ~((-100.000000 + x8)))))) / (((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15)) * ((x6 / ~((-100.000000 + x8))) + ((x14 / -211800) + x15))))) + ~(1)) * (((-0.997895 + (((x6 - 8.800000) + -508) + (-0.9081765689798138 * (38.000000 / sin(x16))))) + ((((x1 - x0) + -7.154491498985021) + (((x0 ^ x15) + -100) + -279.199242)) + -11.682563967704278)) + ((((-88.959518 * (1.000000 / (88.856491 / x1))) + (-2117 + ((x8 - 2118.000000) + ~(x22)))) * (0.00077 + x5)) - (((25.400000 ^ (x20 / 1.0021072170678698)) + ((x14 * 3.1091329981202156) + ((x1 - -1405.000000) + (15666.000000 + x24)))) / ~(((x0 + x25) / ((16.000000 + x23) - 9736))))))))))))
            Best expression (original format) = term1 term2 +, -2.640000 0.051731 x7 sqrt x15 8.000000 - - / / -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + / - x18 x24 - x2 x13 x0 1684.200012 - - + - + 0 ~ 0.00077 x5 + * 25.400000 x20 1.0021072170678698 / ^ x14 3.1091329981202156 * x1 -1405.000000 - 15666.000000 x24 + + + + 0 ~ 16.000000 x23 + 9736 - 16.000000 x23 + 9736 - * * x0 x25 + ~ 16.000000 x23 + 9736 - 16.000000 x23 + 9736 - + * - 16.000000 x23 + 9736 - 16.000000 x23 + 9736 - * 16.000000 x23 + 9736 - 16.000000 x23 + 9736 - * * / ~ * ~ x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * * 25.400000 x20 1.0021072170678698 / ^ x14 3.1091329981202156 * x1 -1405.000000 - 15666.000000 x24 + + + + x0 x25 + ~ 16.000000 x23 + 9736 - 16.000000 x23 + 9736 - * / ~ * ~ x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + ~ 16.000000 x23 + 9736 - 16.000000 x23 + 9736 - * / ~ * x0 x25 + ~ 16.000000 x23 + 9736 - 16.000000 x23 + 9736 - * / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * + * - x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * * / - * -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 0 ~ * ~ -100.000000 x8 + ~ -100.000000 x8 + ~ * / * ~ x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + * / ~ 0 ~ 0.00077 x5 + * 25.400000 x20 1.0021072170678698 / ^ x14 3.1091329981202156 * x1 -1405.000000 - 15666.000000 x24 + + + + x0 x25 + ~ 16.000000 x23 + 9736 - 16.000000 x23 + 9736 - * / ~ * ~ x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * / - * + -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 0 ~ * ~ -100.000000 x8 + ~ -100.000000 x8 + ~ * / * ~ x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + * / ~ 0 ~ 0.00077 x5 + * 25.400000 x20 1.0021072170678698 / ^ x14 3.1091329981202156 * x1 -1405.000000 - 15666.000000 x24 + + + + x0 x25 + ~ 16.000000 x23 + 9736 - 16.000000 x23 + 9736 - * / ~ * ~ x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * / - * -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 0 ~ * ~ -100.000000 x8 + ~ -100.000000 x8 + ~ * * x6 0 ~ * ~ -100.000000 x8 + ~ 0 ~ * 0 ~ -100.000000 x8 + ~ * + * - -100.000000 x8 + ~ -100.000000 x8 + ~ * -100.000000 x8 + ~ -100.000000 x8 + ~ * * / * ~ x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + * * -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 0 ~ * ~ -100.000000 x8 + ~ -100.000000 x8 + ~ * / * ~ x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + x6 0 ~ * ~ -100.000000 x8 + ~ -100.000000 x8 + ~ * / * x6 0 ~ * ~ -100.000000 x8 + ~ -100.000000 x8 + ~ * / x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + * + * - x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + * x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + * * / ~ -0.997895 x6 8.800000 - -508 + -0.9081765689798138 38.000000 x16 sin / * + + x1 x0 - -7.154491498985021 + x0 x15 ^ -100 + -279.199242 + + -11.682563967704278 + + -88.959518 1.000000 88.856491 x1 / / * -2117 x8 2118.000000 - x22 ~ + + + 0.00077 x5 + * 25.400000 x20 1.0021072170678698 / ^ x14 3.1091329981202156 * x1 -1405.000000 - 15666.000000 x24 + + + + x0 x25 + 16.000000 x23 + 9736 - / ~ / - + * + + -2.640000 0.051731 x7 sqrt x15 8.000000 - - / / -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + / - x18 x24 - x2 x13 x0 1684.200012 - - + - + 0 ~ 0.00077 x5 + * 0 ~ 25.400000 x20 1.0021072170678698 / ^ x14 3.1091329981202156 * x1 -1405.000000 - 15666.000000 x24 + + + + 0 ~ * 0 ~ + - x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * * x0 x25 + 16.000000 x23 + 9736 - / ~ 25.400000 x20 1.0021072170678698 / ^ x14 3.1091329981202156 * x1 -1405.000000 - 15666.000000 x24 + + + + 0 ~ * - x0 x25 + 16.000000 x23 + 9736 - / ~ 0 ~ * 0 ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * + * - x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * * / - * -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 0 ~ * ~ -100.000000 x8 + ~ -100.000000 x8 + ~ * / * ~ x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + * / ~ 1 ~ + 0 ~ 0.00077 x5 + * x0 x25 + 16.000000 x23 + 9736 - / ~ 25.400000 x20 1.0021072170678698 / ^ x14 3.1091329981202156 * x1 -1405.000000 - 15666.000000 x24 + + + + 0 ~ * - x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * / - * + -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 0 ~ * ~ -100.000000 x8 + ~ -100.000000 x8 + ~ * / * ~ x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + * / ~ 1 ~ + 0 ~ 0.00077 x5 + * x0 x25 + 16.000000 x23 + 9736 - / ~ 25.400000 x20 1.0021072170678698 / ^ x14 3.1091329981202156 * x1 -1405.000000 - 15666.000000 x24 + + + + 0 ~ * - x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * / - * -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 0 ~ * ~ -100.000000 x8 + ~ -100.000000 x8 + ~ * * x6 0 ~ * ~ -100.000000 x8 + ~ 0 ~ * 0 ~ -100.000000 x8 + ~ * + * - -100.000000 x8 + ~ -100.000000 x8 + ~ * -100.000000 x8 + ~ -100.000000 x8 + ~ * * / * ~ x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + * * -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 0 ~ * ~ -100.000000 x8 + ~ -100.000000 x8 + ~ * / * ~ x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + x6 0 ~ * ~ -100.000000 x8 + ~ -100.000000 x8 + ~ * / * x6 0 ~ * ~ -100.000000 x8 + ~ -100.000000 x8 + ~ * / x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + * + * - x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + * x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + * * / ~ 0 ~ + -0.997895 x6 8.800000 - -508 + -0.9081765689798138 38.000000 x16 sin / * + + x1 x0 - -7.154491498985021 + x0 x15 ^ -100 + -279.199242 + + -11.682563967704278 + + -88.959518 1.000000 88.856491 x1 / / * -2117 x8 2118.000000 - x22 ~ + + + 0.00077 x5 + * 25.400000 x20 1.0021072170678698 / ^ x14 3.1091329981202156 * x1 -1405.000000 - 15666.000000 x24 + + + + x0 x25 + 16.000000 x23 + 9736 - / ~ / - + * + + +, p 1 p - * -2.640000 0.051731 x7 sqrt x15 8.000000 - - / / -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + / - x18 x24 - x2 x13 x0 1684.200012 - - + - + 0 ~ 0.00077 x5 + * 25.400000 x20 1.0021072170678698 / ^ x14 3.1091329981202156 * x1 -1405.000000 - 15666.000000 x24 + + + + 16.000000 x23 + 9736 - 16.000000 x23 + 9736 - 16.000000 x23 + 9736 - * / ~ * ~ x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * / - * -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 0 ~ * ~ -100.000000 x8 + ~ -100.000000 x8 + ~ * / * ~ x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + * / ~ -0.997895 x6 8.800000 - -508 + -0.9081765689798138 38.000000 x16 sin / * + + x1 x0 - -7.154491498985021 + x0 x15 ^ -100 + -279.199242 + + -11.682563967704278 + + -88.959518 1.000000 88.856491 x1 / / * -2117 x8 2118.000000 - x22 ~ + + + 0.00077 x5 + * 25.400000 x20 1.0021072170678698 / ^ x14 3.1091329981202156 * x1 -1405.000000 - 15666.000000 x24 + + + + x0 x25 + 16.000000 x23 + 9736 - / ~ / - + * + u_x -2.640000 0.051731 x7 sqrt x15 8.000000 - - / / -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + / - x18 x24 - x2 x13 x0 1684.200012 - - + - + 0 ~ 0.00077 x5 + * 25.400000 x20 1.0021072170678698 / ^ x14 3.1091329981202156 * x1 -1405.000000 - 15666.000000 x24 + + + + x0 x25 + ~ 16.000000 x23 + 9736 - 16.000000 x23 + 9736 - * / ~ * ~ x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * / - * -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 0 ~ * ~ -100.000000 x8 + ~ -100.000000 x8 + ~ * / * ~ x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + * / ~ -0.997895 x6 8.800000 - -508 + -0.9081765689798138 38.000000 x16 sin / * + + x1 x0 - -7.154491498985021 + x0 x15 ^ -100 + -279.199242 + + -11.682563967704278 + + -88.959518 1.000000 88.856491 x1 / / * -2117 x8 2118.000000 - x22 ~ + + + 0.00077 x5 + * 25.400000 x20 1.0021072170678698 / ^ x14 3.1091329981202156 * x1 -1405.000000 - 15666.000000 x24 + + + + x0 x25 + 16.000000 x23 + 9736 - / ~ / - + * + * u_y -2.640000 0.051731 x7 sqrt x15 8.000000 - - / / -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + / - x18 x24 - x2 x13 x0 1684.200012 - - + - + 0 ~ 0.00077 x5 + * x0 x25 + 16.000000 x23 + 9736 - / ~ 25.400000 x20 1.0021072170678698 / ^ x14 3.1091329981202156 * x1 -1405.000000 - 15666.000000 x24 + + + + 0 ~ * - x0 x25 + 16.000000 x23 + 9736 - / ~ x0 x25 + 16.000000 x23 + 9736 - / ~ * / - * -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 0 ~ * ~ -100.000000 x8 + ~ -100.000000 x8 + ~ * / * ~ x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + x6 -100.000000 x8 + ~ / x14 -211800 / x15 + + * / ~ 1 ~ + -0.997895 x6 8.800000 - -508 + -0.9081765689798138 38.000000 x16 sin / * + + x1 x0 - -7.154491498985021 + x0 x15 ^ -100 + -279.199242 + + -11.682563967704278 + + -88.959518 1.000000 88.856491 x1 / / * -2117 x8 2118.000000 - x22 ~ + + + 0.00077 x5 + * 25.400000 x20 1.0021072170678698 / ^ x14 3.1091329981202156 * x1 -1405.000000 - 15666.000000 x24 + + + + x0 x25 + 16.000000 x23 + 9736 - / ~ / - + * + * + + *
        ```
        from sympy import symbols, cos, sin, tanh, sech, acos, log, sympify, latex, multiline_latex, Float
        import re
        replace_vars = lambda x: re.sub(r'\bx(\d+)\b', r'df["x\1"]', x)
        align_rep = lambda x: x.replace('align*','align').replace(r'\\',r'\nonumber \\').replace(r"\end{align}", r"\label{eq:best_sr_eq_1}""\n"r"\end{align}")
        round_floats = lambda expr, ndigits: expr.xreplace({f: Float(round(float(f), ndigits)) for f in expr.atoms(Float)})
        f, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25 = symbols('f x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25')
        func = '((58 + ((((2.520000 * x2) + 5470.0596345781605) + (3303.5714033364225 ^ (1.320645006282e+07 - (ln(x17) * (x21 - -843.000000))))) + (x8 / (x17 * 0.0007869063691834137)))) * (((((x18 ^ (20200128.000000 - (x7 ^ x7))) + ((x24 + (x15 ^ 16.000000)) + (x16 + (15893.000000 ^ x20)))) + ((0.002105 ^ (-459.704071 - (359.704071 - x2))) + (58.000000 * (x7 + (0.006210 + x12))))) * ((((~(x22) + -88.959518) / (88.856491 * (x8 - 8.000000))) + ((x3 / -2.0200127e+07) + -1)) * (((x22 / (3316.000000 ^ x19)) + (sech(x19) + x11)) * ((0.12004407415282907 + (x14 / x8)) + ((x22 + 39.3125) ^ (0.000770 + x20)))))) - ((((-2.225653 + (x20 / 0.2068956670668029)) + ((0.051731 / (x20 / x6)) + (x20 + x23))) - (((x5 * (x0 + x22)) + ((0.002105 / x11) + 1.5707963267948966)) + (((256.500000 ^ x15) + x17) + (4.00621 + x24)))) * (((((x20 * 360.000000) + -15858.185171) + (360.000000 + (-2.640000 * x7))) / ((8.051731 + ln(x6)) + ((8.000000 - x18) + -781.0930331488324))) + (((x10 * (x16 / x14)) + ((x20 / x15) + 1.7112288116110355)) + (((58.000000 / x1) + 36.310283) + ((0.000770 + x18) + -73.0701029874863)))))))'
        func = func.replace("^","**").replace("~","-")
        func_sym = sympify(func)
        func_sym_r = round_floats(func_sym, 3)
        print(f"func_sym = {align_rep(multiline_latex(f, func_sym_r, 1))}")
        f_res = replace_vars(func)
        print(f"f = {f_res}")
        ```
     */
    
    p_expr.clear();
    p_expr.reserve(100);
    dfdx23.clear();
    dfdx23.reserve(100);
    dfdx24.clear();
    dfdx24.reserve(100);
    grasp.clear();
    grasp.reserve(100);
    for (decltype(results.size()) i = 0; i < results.size(); i++)
    {
        results[i].reserve(100);
    }

    if (x.expression_type == "prefix")
    {
        //Prefix: + term1 term2
        if (!prefactors_computed)
        {
            //prefac_term_1 = w1 * x26
            //prefac_term2 = w0 * (1.0 - x26)
            x.subs_dict["prefac_term_1"] = x.expression_evaluator(x.params, std::vector<std::string>{"*", w1, "x26"}); //* w1 x26
            x.subs_dict["prefac_term_2"] = x.expression_evaluator(x.params, std::vector<std::string>{"*", w0, "-", "1", "x26"}); //* w0 - 1 x26
            x.subs_dict["theta"] = deg2rad(Board::data["x7"]);
            x.subs_dict["u_x"] = x.expression_evaluator(x.params, std::vector<std::string>{"*", "x6", "cos", "theta"});
            x.subs_dict["u_y"] = x.expression_evaluator(x.params, std::vector<std::string>{"*", "x6", "sin", "theta"});
        }
        if (fit)
        {
            //+ * prefac_term_1 log + / 1 - 1 exp ~ f eps * prefac_term_2 log + - 1 / 1 - 1 exp ~ f eps
            results[0] = {"+", "*", "prefac_term_1", "log", "+", "/", "1", "-", "1", "exp", "~"};
            for (const std::string& i: x.pieces[0]) // f
            {
                results[0].push_back(i);
            }
            results[0].push_back(eps); // eps
            results[0].push_back("*"); // *
            results[0].push_back("prefac_term_2"); // prefac_term_2
            results[0].push_back("log"); // log
            results[0].push_back("+"); // +
            results[0].push_back("-"); // -
            results[0].push_back("1"); // 1
            results[0].push_back("/"); // /
            results[0].push_back("1"); // 1
            results[0].push_back("-"); // -
            results[0].push_back("1"); // 1
            results[0].push_back("exp"); // exp
            results[0].push_back("~"); // ~
            for (const std::string& i: x.pieces[0]) // f
            {
                results[0].push_back(i);
            }
            results[0].push_back(eps); // eps
        }
        else
        {
            p_expr = {"/", "1", "-", "1", "exp", "~"}; // / 1 - 1 exp ~
            for (const std::string& i: x.pieces[0]) // f
            {
                p_expr.push_back(i);
            }
            x.subs_dict["p"] = x.expression_evaluator(x.params, p_expr);

            //term1 = * prefac_term_1 log + p eps
            x.subs_dict["term1"] = x.expression_evaluator(x.params, {"*", "prefac_term_1", "log", "+", "p", eps});
            //term2 = * prefac_term_2 log + - 1 p eps
            x.subs_dict["term2"] = x.expression_evaluator(x.params, {"*", "prefac_term_2", "log", "+", "-", "1", "p", eps});
            results[0] = {"+", "term1", "term2"};
        }
        //+ ∂^2f/∂(x23)^2 ∂^2f/∂(x24)^2
        results[1] = {"+"};
        x.derivePrefix(0, x.pieces[0].size()-1, "x23", x.pieces[0], grasp);
        dfdx23 = x.derivat;
        x.derivePrefix(0, dfdx23.size()-1, "x23", dfdx23, grasp);
        for (const std::string& i: x.derivat) //∂^2f/∂(x23)^2
        {
            results[1].push_back(i);
        }
        x.derivePrefix(0, x.pieces[0].size()-1, "x24", x.pieces[0], grasp);
        dfdx24 = x.derivat;
        x.derivePrefix(0, dfdx24.size()-1, "x24", dfdx24, grasp);
        for (const std::string& i: x.derivat) //∂^2f/∂(x24)^2
        {
            results[1].push_back(i);
        }
        //* * p - 1 p + ∂f/∂(x25) + * u_x ∂f/∂(x23) * u_y ∂f/∂(x24)
        results[2].push_back("*"); // *
        if (fit)
        {
            //* / 1 - 1 exp ~ f - 1 / 1 - 1 exp ~ f
            results[2].push_back("*"); // *
            results[2].push_back("/"); // /
            results[2].push_back("1"); // 1
            results[2].push_back("-"); // -
            results[2].push_back("1"); // 1
            results[2].push_back("exp"); // exp
            results[2].push_back("~"); // ~
            for (const std::string& i: x.pieces[0]) // f
            {
                results[2].push_back(i);
            }
            results[2].push_back("-"); // -
            results[2].push_back("1"); // 1
            results[2].push_back("/"); // /
            results[2].push_back("1"); // 1
            results[2].push_back("-"); // -
            results[2].push_back("1"); // 1
            results[2].push_back("exp"); // exp
            results[2].push_back("~"); // ~
            for (const std::string& i: x.pieces[0]) // f
            {
                results[2].push_back(i);
            }
        }
        else
        {
            //* p - 1 p
            results[2].push_back("*"); // *
            results[2].push_back("p"); // p
            results[2].push_back("-"); // -
            results[2].push_back("1"); // 1
            results[2].push_back("p"); // p
        }
        //+ ∂f/∂(x25) + * u_x ∂f/∂(x23) * u_y ∂f/∂(x24)
        results[2].push_back("+"); // +
        x.derivePrefix(0, x.pieces[0].size()-1, "x25", x.pieces[0], grasp);
        for (const std::string& i: x.derivat) //∂^2f/∂(x25)
        {
            results[2].push_back(i);
        }
        results[2].push_back("+"); // +
        results[2].push_back("*"); // *
        results[2].push_back("u_x"); // u_x
        for (const std::string& i: dfdx23) //∂^2f/∂(x23)
        {
            results[2].push_back(i);
        }
        results[2].push_back("*"); // *
        results[2].push_back("u_y"); // u_y
        for (const std::string& i: dfdx24) //∂^2f/∂(x24)
        {
            results[2].push_back(i);
        }
    }
    else if (x.expression_type == "postfix")
    {
        //Postfix: term1 term2 +
        if (!prefactors_computed)
        {
            //prefac_term_1 = w1 * x26
            //prefac_term2 = w0 * (1.0 - x26)
            x.subs_dict["prefac_term_1"] = x.expression_evaluator(x.params, std::vector<std::string>{w1, "x26", "*"}); //w1 x26 *
            x.subs_dict["prefac_term_2"] = x.expression_evaluator(x.params, std::vector<std::string>{w0, "1", "x26", "-", "*"}); //w0 1 x26 - *
            x.subs_dict["theta"] = deg2rad(Board::data["x7"]);
            x.subs_dict["u_x"] = x.expression_evaluator(x.params, std::vector<std::string>{"x6", "theta", "cos", "*"});
            x.subs_dict["u_y"] = x.expression_evaluator(x.params, std::vector<std::string>{"x6", "theta", "sin", "*"});
        }
        if (fit)
        {
            //prefac_term_1 1 1 f ~ exp - / eps + log * prefac_term_2 1 1 1 f ~ exp - / - eps + log * +
            results[0] = {"prefac_term_1", "1", "1"};
            for (const std::string& i: x.pieces[0]) // f
            {
                results[0].push_back(i);
            }
            results[0].push_back("~"); // ~
            results[0].push_back("exp"); // exp
            results[0].push_back("-"); // -
            results[0].push_back("/"); // /
            results[0].push_back(eps); // eps
            results[0].push_back("+"); // +
            results[0].push_back("log"); // log
            results[0].push_back("*"); // *
            results[0].push_back("prefac_term_2"); // prefac_term_2
            results[0].push_back("1"); // 1
            results[0].push_back("1"); // 1
            results[0].push_back("1"); // 1
            for (const std::string& i: x.pieces[0]) // f
            {
                results[0].push_back(i);
            }
            results[0].push_back("~"); // ~
            results[0].push_back("exp"); // exp
            results[0].push_back("-"); // -
            results[0].push_back("/"); // /
            results[0].push_back("-"); // -
            results[0].push_back(eps); // eps
            results[0].push_back("+"); // +
            results[0].push_back("log"); // log
            results[0].push_back("*"); // *
            results[0].push_back("+"); // +
        }
        else
        {
            //p = 1 1 f ~ exp - /
            p_expr = {"1", "1"}; // 1 1
            for (const std::string& i: x.pieces[0]) // f
            {
                p_expr.push_back(i);
            }
            p_expr.push_back("~"); // ~
            p_expr.push_back("exp"); // exp
            p_expr.push_back("-"); // -
            p_expr.push_back("/"); // /
            x.subs_dict["p"] = x.expression_evaluator(x.params, p_expr);
            //term1 = prefac_term_1 p eps + log *
            x.subs_dict["term1"] = x.expression_evaluator(x.params, {"prefac_term_1", "p", eps, "+", "log", "*"});
            //term2 = prefac_term_2 1 p - eps + log *
            x.subs_dict["term2"] = x.expression_evaluator(x.params, {"prefac_term_2", "1", "p", "-", eps, "+", "log", "*"});
            results[0] = {"term1", "term2", "+"};
        }
        //∂^2f/∂(x23)^2 ∂^2f/∂(x24)^2 +
        x.derivePostfix(0, x.pieces[0].size()-1, "x23", x.pieces[0], grasp);
        dfdx23 = x.derivat;
        x.derivePostfix(0, dfdx23.size()-1, "x23", dfdx23, grasp);
        for (const std::string& i: x.derivat) // ∂^2f/∂(x23)^2
        {
            results[1].push_back(i);
        }
        x.derivePostfix(0, x.pieces[0].size()-1, "x24", x.pieces[0], grasp);
        dfdx24 = x.derivat;
        x.derivePostfix(0, dfdx24.size()-1, "x24", dfdx24, grasp);
        for (const std::string& i: x.derivat) // ∂^2f/∂(x24)^2
        {
            results[1].push_back(i);
        }
        results[1].push_back("+"); // +
        //p 1 p - * ∂f/∂(x25) u_x ∂f/∂(x23) * u_y ∂f/∂(x24) * + + *
        if (fit)
        {
            //1 1 f ~ exp - / 1 1 1 f ~ exp - / - *
            results[2].push_back("1"); // 1
            results[2].push_back("1"); // 1
            for (const std::string& i: x.pieces[0]) // f
            {
                results[2].push_back(i);
            }
            results[2].push_back("~"); // ~
            results[2].push_back("exp"); // exp
            results[2].push_back("-"); // -
            results[2].push_back("/"); // /
            results[2].push_back("1"); // 1
            results[2].push_back("1"); // 1
            results[2].push_back("1"); // 1
            for (const std::string& i: x.pieces[0]) // f
            {
                results[2].push_back(i);
            }
            results[2].push_back("~"); // ~
            results[2].push_back("exp"); // exp
            results[2].push_back("-"); // -
            results[2].push_back("/"); // /
            results[2].push_back("-"); // -
            results[2].push_back("*"); // *
        }
        else
        {
            //p 1 p - *
            results[2].push_back("p");
            results[2].push_back("1");
            results[2].push_back("p");
            results[2].push_back("-");
            results[2].push_back("*");
        }
        //∂f/∂(x25) u_x ∂f/∂(x23) * u_y ∂f/∂(x24) * + + *
        x.derivePostfix(0, x.pieces[0].size()-1, "x25", x.pieces[0], grasp);
        for (const std::string& i: x.derivat) // ∂f/∂(x25)
        {
            results[2].push_back(i);
        }
        results[2].push_back("u_x");
        for (const std::string& i: dfdx23) // ∂f/∂(x23)
        {
            results[2].push_back(i);
        }
        results[2].push_back("*"); // *
        results[2].push_back("u_y"); // u_y
        for (const std::string& i: dfdx24) // ∂f/∂(x24)
        {
            results[2].push_back(i);
        }
        results[2].push_back("*"); // *
        results[2].push_back("+"); // +
        results[2].push_back("+"); // +
        results[2].push_back("*"); // *
    }
    prefactors_computed = true;
    return results;
}

/*
||===================================================================================================================================||
|| Equations from here (14-15): https://pubs.aip.org/aip/pop/article/23/3/032102/1015921/Laser-propagation-and-soliton-generation-in ||
||===================================================================================================================================||

 Infix: ∂^2(tanh(u)*((1/sech(u)) - α))/∂ξ^2 + ω_squared_factor*n*(tanh(u)*((1/sech(u)) - α)) - (n/(1+ρ_i*α))*(tanh(u)*(1+(ρ_i/sech(u))))
 Postfix: u tanh 1 u sech / α - * ∂^2/∂ξ^2 ω_squared_factor n * u tanh 1 u sech / α - * * + n 1 ρ_i α * + / u tanh 1 ρ_i u sech / + * * -

 Infix: c_s_squared*ln(n) - ρ_i*((1-(1/sech(u))) + (α/2)*tanh^2(u) - (ρ_i*((tanh(u)*((1/sech(u)) - α))^2))/(2*(1+ρ_i*α)))
 Postfix: c_s_squared n ln * ρ_i 1 1 u sech / - α 2 / u tanh 2 ^ * + ρ_i u tanh 1 u sech / α - * 2 ^ * 2 1 ρ_i α * + * / - * -

 Infix: tanh(u(ξ_min))*((1/sech(u(ξ_min))) - α*const0)
 Postfix: u(ξ_min) tanh 1 u(ξ_min) sech / α const0 * - *

 Infix: tanh(u(ξ_max))*((1/sech(u(ξ_max))) - α*const0)
 Postfix: u(ξ_max) tanh 1 u(ξ_max) sech / α const0 * - *

 Infix: ((1/sech(u(ξ_min))) - α*const0*sech^2(u(ξ_min)))*∂u(x_min)/∂ξ
 Postfix: 1 u(ξ_min) sech / α const0 * u(ξ_min) sech 2 ^ * - ∂u(x_min)/∂ξ *

 Infix: ((1/sech(u(ξ_max))) - α*const0*sech^2(u(ξ_max)))*∂u(x_max)/∂ξ
 Postfix: 1 u(ξ_max) sech / α const0 * u(ξ_max) sech 2 ^ * - ∂u(x_max)/∂ξ *

// Infix: (tanh(u)*((1/sech(u)) - α))^2
// Postfix: u tanh 1 u sech / α - * 2 ^

 Infix: abs(n(ξ) - n(-ξ))
// Postfix: n(ξ) n(-ξ) - 2 ^
 Postfix: n(ξ) n(-ξ) - abs

 Infix: abs(((n(ξ)/1) - 1) - x1)
// Postfix: n(ξ) 1 / 1 - x1 - 2 ^
 Postfix: n(ξ) 1 / 1 - x1 - abs

 Infix: 10*abs((tanh(u)*((1/sech(u)) - α*const0)) - x2)
// Postfix: u tanh 1 u sech / α const0 * - * x2 - 2 ^
 Postfix: u tanh 1 u sech / α const0 * - * x2 - abs 10 *

 {x0: ξ}
 {x.pieces[0]: u, x.pieces[1]: n}

*/
std::vector<std::vector<std::string>> SolitonWaveFengEq14and15Laser(Board& x, bool fit)
{
    assert(x.feature_mins_maxes.size() && "const_tokens must be true for this equation to use feature_mins_maxes!");
    //std::vector<std::vector<std::string>> results(10); //2 equations for the ODE, 4 equations for the boundary conditions, 2 equations for symmetry of a(u) and n respectively, 2 equations for data
    std::vector<std::vector<std::string>> results(9); //2 equations for the ODE, 4 equations for the boundary conditions, 1 equation for symmetry of n respectively, 2 equations for data
    for (decltype(results.size()) i = 0; i < results.size(); i++){results[i].reserve(100);}
    std::vector<std::string> temp, temp_prime;
    temp.reserve(100);
    temp_prime.reserve(100);
    std::vector<int> grasp;
    /*
      For parameters commented-out below (first 2 equations only):
       - Best score = 0.997519, SNE = 2.17869e+27
       - Best expression = ((-0.416147 * x0) / -31.415920), tanh(sech(x0))

        //constexpr const char* rho = "0.000544662309"; // 1/1836, Figs 10-11 caption, https://www.bing.com/search?q=9.1e-31%2F%201.67e-27%20&qs=n&form=QBRE&sp=-1&ghc=1&lq=0&pq=9.1e-31%2F%201.67e-27%20&sc=0-18&sk=&cvid=09FAD78B6CC6414E98D1BED802D49D1C
        //constexpr const char* omega_squared_factor_for_omega_0_point_8_omega_pe = "2.26016865e-7"; //0.8^2 * 4*pi*(q_e^2 / m_e), Figs 10-11 caption "ω = 0.8*ω_{pe}", ω_{pe} = (4*pi*(n_e=n)*(q_e^2))/(m_e), see "III. PROPAGATION MODES", https://www.bing.com/search?q=(((-1.6e-19)%5E2)%2F(9.109382902843941771e-31))*(.8%5E2)*(4*pi)&qs=n&form=QBRE&sp=-1&lq=0&pq=(((-1.6e-19)%5E2)%2F(9.109382902843941771e-31))*(.8%5E2)*(4*pi)&sc=0-57&sk=&cvid=E84D46D8A3F04F38ABE1297CC3515859
        //constexpr const char* omega_squared_factor = omega_squared_factor_for_omega_0_point_8_omega_pe;
        //constexpr const char* cs_squared_factor_for_rho_i_1_over_1836_v_te_0_point_05c_v_ti_0_point_001c = "2.12255036e11"; //(2.99792458e8*2.99792458e8)*(((.05*.05)/1836) + (.001*.001)), https://www.bing.com/search?q=(2.99792458e8*2.99792458e8)*(((.05*.05)%2F1836)%20%2B%20(.001*.001))&qs=n&form=QBRE&sp=-1&lq=0&pq=(2.99792458e8*2.99792458e8)*(((.05*.05)%2F1836)%20%2B%20(.001*.001))&sc=0-60&sk=&cvid=A05685C4D5D440BC9810452F04051A5E
        //constexpr const char* cs_squared = cs_squared_factor_for_rho_i_1_over_1836_v_te_0_point_05c_v_ti_0_point_001c;
        //constexpr const char* alpha_0 = "0";
        //constexpr const char* alpha = alpha_0;
        //constexpr const char* one_plus_rho_i_times_alpha_for_rho_i_1_over_1836_alpha_0 = "1"; //1 + ρ_i*0 = 1
        //constexpr const char* one_plus_rho_i_times_alpha = one_plus_rho_i_times_alpha_for_rho_i_1_over_1836_alpha_0; //1 + ρ_i*α
    */

    /*
      For parameters below (first 6 equations only)
        - Best score = 5.1745, SNE = 3.46556
        - Best expression = arccos((0.000004 ^ sech(x0))), exp(~(sin(cos(x0))))
        - Best expression (original format) = 0.000004 x0 sech ^ arccos, x0 cos sin ~ exp

      For parameters below (first 8 equations)
        - Best score = 7.99935, SNE = 0.000654101
        - Best expression = (1 / (-10.594090 / arcsin(sech(x0)))), cos((sech(x0) - (cos(1) - sech(x0))))
        - Best expression (original format) = 1 -10.594090 x0 sech arcsin / /, x0 sech 1 cos x0 sech - - cos

      For parameters below (all 10 equations)
        - Best score = 9.99802, SNE = 0.00198078
        - Best expression = (cos(1.559132) * sech((x0 * 0.774245))), sech((sech(x0) * (0.157922 ^ cos(4))))
        - Best expression (original format) = 1.559132 cos x0 0.774245 * sech *, x0 sech 0.157922 4 cos ^ * sech

      For parameters below (first 6 equations and last 3 equations WITH `const0`)
        - Best score = 9.99802, SNE = 0.00198078
        - Best expression = (cos(1.559132) * sech((x0 * 0.774245))), sech((sech(x0) * (0.157922 ^ cos(4))))
        - Best expression (original format) = 1.559132 cos x0 0.774245 * sech *, x0 sech 0.157922 4 cos ^ * sech

     For parameters below (same configuration as the one right above but adding `10 *` at the end of last equation and changing threshold `this->isConstTol` to 0.001 instead of 0 and changing `^ 2` to `abs` in last 3 equations
        - Best score = 8.94096, SNE = 0.0602338
        - Best expression = (tanh(tanh(sech(x0))) / (-6.466281 - (tanh(x0) / 2.616570))), sech((3.235163 * (sech(x0) ^ 0.964028)))
        - Best expression (original format) = x0 sech tanh tanh -6.466281 x0 tanh 2.616570 / - /, 3.235163 x0 sech 0.964028 ^ * sech
        - Best differential equation parameters = {(const0, 5.23438)}
        - Best expression parameters = {}
            - Equation 1 SNE: 0.0206838
            - Equation 2 SNE: 2.59327e-10
            - Equation 3 SNE: 1.5193e-08
            - Equation 4 SNE: 9.7235e-09
            - Equation 5 SNE: 1.5193e-08
            - Equation 6 SNE: 9.7235e-09
            - Equation 7 SNE: 0
            - Equation 8 SNE: 0.021413
            - Equation 9 SNE: 0.0178037
     
     Change of convention, current best right now
         Best score = 0.943537, SNE = 0.0598418
         Squared-norm error for each equation: 0.0210858 2.52051e-10 1.42123e-08 1.05092e-08 1.42123e-08 1.05092e-08 0 0.0209531 0.0178028
         Best expression = (tanh(tanh(sech(x0))) / (-6.4342880000000005 - (tanh(x0) / 2.61657))), sech((3.218281828459045 * (sech(x0) ^ 0.9640275800758169)))
         Best expression (original format) = x0 sech tanh tanh -6.4342880000000005 x0 tanh 2.61657 / - /, 3.218281828459045 x0 sech 0.9640275800758169 ^ * sech
    */
    constexpr const char* rho = "0.000544662309"; // 1/1836, Figs 10-11 caption, https://www.bing.com/search?q=9.1e-31%2F%201.67e-27%20&qs=n&form=QBRE&sp=-1&ghc=1&lq=0&pq=9.1e-31%2F%201.67e-27%20&sc=0-18&sk=&cvid=09FAD78B6CC6414E98D1BED802D49D1C
    constexpr const char* omega_squared_factor_for_omega_0_point_8_omega_pe = "0.64"; //0.8^2 ω_{pe} = 0.64 ω_{pe}, Figs 10-11 caption "ω = 0.8*ω_{pe}", ω_{pe} = (4*pi*(n_e=n)*(q_e^2))/(m_e), see "III. PROPAGATION MODES"
    constexpr const char* omega_squared_factor = omega_squared_factor_for_omega_0_point_8_omega_pe;
    constexpr const char* cs_squared_factor_for_rho_i_1_over_1836_v_te_0_point_05c_v_ti_0_point_001c = "2.36165577e-6"; //Figure 10: (((.05*.05)/1836) + (.001*.001)), https://www.bing.com/search?q=(((.05*.05)%2F1836)%20%2B%20(.001*.001))&qs=n&form=QBRE&sp=-1&lq=0&pq=(((.05*.05)%2F1836)%20%2B%20(.001*.001))&sc=1-32&sk=&cvid=C563F2D3B3AB42BAA438BCD2FAF6F307&ajf=10
    constexpr const char* cs_squared = cs_squared_factor_for_rho_i_1_over_1836_v_te_0_point_05c_v_ti_0_point_001c;
    constexpr const char* alpha_0 = "0.4";
    constexpr const char* alpha = alpha_0;
//    constexpr const char* one_plus_rho_i_times_alpha_for_rho_i_1_over_1836_alpha_0 = "1.0002178649237472767"; //1 + ρ_i*0.4 = 1.0002178649237472767
////    constexpr const char* one_plus_rho_i_times_alpha = one_plus_rho_i_times_alpha_for_rho_i_1_over_1836_alpha_0; //1 + ρ_i*α
    constexpr const char* const0 = "5.22145";
//    constexpr const char* const0 = "const0";
//    std::string infty = std::numeric_limits<float>::infinity();

    if (x.expression_type == "prefix")
    {
        throw std::invalid_argument("Prefix not implemented yet for this SolitonWaveFengEq14and15Laser function!");
    }
    else if (x.expression_type == "postfix")
    {
        //u tanh 1 u sech / α - * ∂^2/∂ξ^2 ω_squared_factor n * u tanh 1 u sech / α - * * + n 1 ρ_i α * + / u tanh 1 ρ_i u sech / + * * -
        for (const std::string& i: x.pieces[0]) // u
        {
            temp.push_back(i);
        }
        temp.push_back("tanh"); // tanh
        temp.push_back("1"); // 1
        for (const std::string& i: x.pieces[0]) // u
        {
            temp.push_back(i);
        }
        temp.push_back("sech"); // sech
        temp.push_back("/"); // /
        temp.push_back(alpha); // α
        temp.push_back("-"); // -
        temp.push_back("*"); // *
        //temp now contains: u tanh 1 u sech / α - *
        x.derivePostfix(0, temp.size()-1, "x0", temp, grasp);
        temp_prime = x.derivat;
        x.derivePostfix(0, temp_prime.size()-1, "x0", temp_prime, grasp);
        for (const std::string& i: x.derivat) // u tanh 1 u sech / α - * ∂^2/∂ξ^2
        {
            results[0].push_back(i);
        }
        results[0].push_back(omega_squared_factor); // ω_squared_factor
        for (const std::string& i: x.pieces[1]) // n
        {
            results[0].push_back(i);
        }
        results[0].push_back("*"); // *
        for (const std::string& i: temp) // u tanh 1 u sech / α - *
        {
            results[0].push_back(i);
        }
        results[0].push_back("*"); // *
        results[0].push_back("+"); // +
        for (const std::string& i: x.pieces[1]) // n
        {
            results[0].push_back(i);
        }
        results[0].push_back("1"); // 1
        results[0].push_back(rho); // ρ_i
        results[0].push_back(alpha); // α
        results[0].push_back("*"); // *
        results[0].push_back("+"); // +
        results[0].push_back("/"); // /
        for (const std::string& i: x.pieces[0]) // u
        {
            results[0].push_back(i);
        }
        results[0].push_back("tanh"); // tanh
        results[0].push_back("1"); // 1
        results[0].push_back(rho); // ρ_i
        for (const std::string& i: x.pieces[0]) // u
        {
            results[0].push_back(i);
        }
        results[0].push_back("sech"); // sech
        results[0].push_back("/"); // /
        results[0].push_back("+"); // +
        results[0].push_back("*"); // *
        results[0].push_back("*"); // *
        results[0].push_back("-"); // -
        //c_s_squared n ln * ρ_i 1 1 u sech / - α 2 / u tanh 2 ^ * + ρ_i u tanh 1 u sech / α - * 2 ^ * 2 1 ρ_i α * + * / - * -
        results[1].push_back(cs_squared); // c_s_squared
        for (const std::string& i: x.pieces[1]) // n
        {
            results[1].push_back(i);
        }
        results[1].push_back("ln"); // ln
        results[1].push_back("*"); // *
        results[1].push_back(rho); // ρ_i
        results[1].push_back("1"); // 1
        results[1].push_back("1"); // 1
        for (const std::string& i: x.pieces[0]) // u
        {
            results[1].push_back(i);
        }
        results[1].push_back("sech"); // sech
        results[1].push_back("/"); // /
        results[1].push_back("-"); // -
        results[1].push_back(alpha); // α
        results[1].push_back("2"); // 2
        results[1].push_back("/"); // /
        for (const std::string& i: x.pieces[0]) // u
        {
            results[1].push_back(i);
        }
        results[1].push_back("tanh"); // tanh
        results[1].push_back("2"); // 2
        results[1].push_back("^"); // ^
        results[1].push_back("*"); // *
        results[1].push_back("+"); // +
        results[1].push_back(rho); // ρ_i
        for (const std::string& i: temp) // u tanh 1 u sech / α - *
        {
            results[1].push_back(i);
        }
        results[1].push_back("2"); // 2
        results[1].push_back("^"); // ^
        results[1].push_back("*"); // *
        results[1].push_back("2"); // 2
        results[1].push_back("1"); // 1
        results[1].push_back(rho); // ρ_i
        results[1].push_back(alpha); // α
        results[1].push_back("*"); // *
        results[1].push_back("+"); // +
        results[1].push_back("*"); // *
        results[1].push_back("/"); // /
        results[1].push_back("-"); // -
        results[1].push_back("*"); // *
        results[1].push_back("-"); // -
        //u(ξ_min) tanh 1 u(ξ_min) sech / α const0 * - *
        assert(results[2].size() == 0);
        size_t temp_idx = 0;
        for (const std::string& i: temp) //u(ξ_min) tanh 1 u(ξ_min) sech / α const0 * - *
        {
            if (i == "x0")
            {
                results[2].push_back(x.feature_mins_maxes[i].first);
            }
            else if ((i == alpha) && (temp_idx==temp.size() - 3))
            {
                results[2].push_back(alpha);
                results[2].push_back(const0);
                results[2].push_back("*");
            }
            else
            {
                results[2].push_back(i);
            }
            temp_idx++;
        } //u tanh 1 u sech / α - *
//        if ((2+temp.size()) != results[2].size())
//        {
//            std::scoped_lock error_lock(Board::thread_locker);
//            std::cout << "\ntemp = " << temp << "\nresults[2] = " << results[2] << '\n';
//            throw std::runtime_error(std::string("temp.size() = ")+std::to_string(temp.size())+", results[2].size() = "
//                                     + std::to_string(results[2].size())+"\n");
//        }
        assert((2+temp.size()) == results[2].size()); //sanity check
        //u(ξ_max) tanh 1 u(ξ_max) sech / α const0 * - *
        temp_idx = 0;
        for (const std::string& i: temp) // u(ξ_max) tanh 1 u(ξ_max) sech / α const0 * - *
        {
            if (i == "x0")
            {
                results[3].push_back(x.feature_mins_maxes[i].second);
            }
            else if ((i == alpha) && (temp_idx==temp.size() - 3))
            {
                results[3].push_back(alpha);
                results[3].push_back(const0);
                results[3].push_back("*");
            }
            else
            {
                results[3].push_back(i);
            }
            temp_idx++;
        }
        assert((2+temp.size()) == results[3].size()); //sanity check
        //1 u(ξ_min) sech / α const0 * u(ξ_min) sech 2 ^ * - ∂u(x_min)/∂ξ *
        results[4].push_back("1"); // 1
        for (const std::string& i: x.pieces[0]) // u
        {
            if (i == "x0")
            {
                results[4].push_back(x.feature_mins_maxes[i].first);
            }
            else
            {
                results[4].push_back(i);
            }
        }
        results[4].push_back("sech"); // sech
        results[4].push_back("/"); // /
        results[4].push_back(alpha); // α
        results[4].push_back(const0); // const0
        results[4].push_back("*"); // *
        for (const std::string& i: x.pieces[0]) // u
        {
            if (i == "x0")
            {
                results[4].push_back(x.feature_mins_maxes[i].first);
            }
            else
            {
                results[4].push_back(i);
            }
        }
        results[4].push_back("sech"); // sech
        results[4].push_back("2"); // 2
        results[4].push_back("^"); // ^
        results[4].push_back("*"); // *
        results[4].push_back("-"); // -
        x.derivePostfix(0, x.pieces[0].size()-1, "x0", x.pieces[0], grasp);
        //now `x.derivat` stores ∂u/∂ξ
        for (const std::string& i: x.derivat) // ∂u(x_min)/∂ξ
        {
            if (i == "x0")
            {
                results[4].push_back(x.feature_mins_maxes[i].first);
            }
            else
            {
                results[4].push_back(i);
            }
        }
        results[4].push_back("*"); // *
        //1 u(ξ_max) sech / α const0 * u(ξ_max) sech 2 ^ * - ∂u(x_max)/∂ξ *
        results[5].push_back("1"); // 1
        for (const std::string& i: x.pieces[0]) // u(ξ_max)
        {
            if (i == "x0")
            {
                results[5].push_back(x.feature_mins_maxes[i].second);
            }
            else
            {
                results[5].push_back(i);
            }
        }
        results[5].push_back("sech"); // sech
        results[5].push_back("/"); // /
        results[5].push_back(alpha); // α
        results[5].push_back(const0); // const0
        results[5].push_back("*"); // *
        for (const std::string& i: x.pieces[0]) // u(ξ_max)
        {
            if (i == "x0")
            {
                results[5].push_back(x.feature_mins_maxes[i].second);
            }
            else
            {
                results[5].push_back(i);
            }
        }
        results[5].push_back("sech"); // sech
        results[5].push_back("2"); // 2
        results[5].push_back("^"); // ^
        results[5].push_back("*"); // *
        results[5].push_back("-"); // -
        for (const std::string& i: x.derivat) // ∂u(x_max)/∂ξ
        {
            if (i == "x0")
            {
                results[5].push_back(x.feature_mins_maxes[i].second);
            }
            else
            {
                results[5].push_back(i);
            }
        }
        results[5].push_back("*"); // *
        //u tanh 1 u sech / α - * 2 ^
//        for (const std::string& i: temp) // u tanh 1 u sech / α - *
//        {
//            results[6].push_back(i);
//        }
//        results[6].push_back("2");
//        results[6].push_back("^");
        //n(ξ) n(-ξ) - 2 ^
        //n(ξ) n(-ξ) - abs
        for (const std::string& i: x.pieces[1]) //n(ξ)
        {
            results[6].push_back(i);
        }
        for (const std::string& i: x.pieces[1]) //n(-ξ)
        {
            results[6].push_back(i);
            if (i == "x0")
            {
                results[6].push_back("~");
            }
        }
        results[6].push_back("-");
        //results[6].push_back("2");
        //results[6].push_back("^");
        results[6].push_back("abs");
        //n(ξ) 1 / 1 - x1 - 2 ^
        //n(ξ) 1 / 1 - x1 - abs
        for (const std::string& i: x.pieces[1]) //n(ξ)
        {
            results[7].push_back(i);
        }
        results[7].push_back("1");
        results[7].push_back("/");
        results[7].push_back("1");
        results[7].push_back("-");
        results[7].push_back("x1");
        results[7].push_back("-");
        //results[7].push_back("2");
        //results[7].push_back("^");
        results[7].push_back("abs");
        //u tanh 1 u sech / α const0 * - * x2 - abs 10 *
        temp_idx = 0;
        for (const std::string& i: temp) // u tanh 1 u sech / α const0 * - *
        {
            results[8].push_back(i);
            if ((i == alpha) && (temp_idx==temp.size() - 3))
            {
                results[8].push_back(const0);
                results[8].push_back("*");
            }
            temp_idx++;
        }
        results[8].push_back("x2");
        results[8].push_back("-");
        //results[8].push_back("2");
        //results[8].push_back("^");
        results[8].push_back("abs");
        results[8].push_back("10");
        results[8].push_back("*");
    }
//    results[0] = results[8];
//    results.resize(1);
    return results;
}

std::vector<std::vector<std::string>> VortexRadialProfile(Board& x, bool fit)
{
    std::vector<std::vector<std::string>> results;
    std::vector<std::string> result;
    result.reserve(100);
    std::vector<int> grasp;
    std::vector<std::string> R_prime;
    std::string mu = "1";
    std::string S = "1";
    std::string infty = std::to_string(DBL_MAX);

    /*
     Best score = 0.972546, SNE = 0.0282288
     Squared-norm error for each equation: 0.0282288 0 0
     Best expression = sin(acos((sech(x0) ^ 0.625850140725043)))
     Best expression (original format) = sin acos ^ sech x0 0.625850140725043
     */
    
    if (x.expression_type == "prefix")
    {
        //- + + * / 1 2 R'' * / 1 * 2 r R' * - mu / * S S * * 2 r r R * * R R R
        result.push_back("-");
        result.push_back("+");
        result.push_back("+");
        result.push_back("*");
        result.push_back("/");
        result.push_back("1");
        result.push_back("2");
        x.derivePrefix(0, x.pieces[0].size()-1, "x0", x.pieces[0], grasp);
        R_prime = x.derivat;
        x.derivePrefix(0, R_prime.size()-1, "x0", R_prime, grasp); //derivat will store second derivative of R_prime
        for (const std::string& i: x.derivat) //R''
        {
            result.push_back(i);
        }
        result.push_back("*");
        result.push_back("/");
        result.push_back("1");
        result.push_back("*");
        result.push_back("2");
        result.push_back("x0"); //r
        for (const std::string& i: R_prime) //R'
        {
            result.push_back(i);
        }
        result.push_back("*");
        result.push_back("-");
        result.push_back(mu);
        result.push_back("/");
        result.push_back("*");
        result.push_back(S);
        result.push_back(S);
        result.push_back("*");
        result.push_back("*");
        result.push_back("2");
        result.push_back("x0"); //r
        result.push_back("x0"); //r
        for (const std::string& i: x.pieces[0]) //R
        {
            result.push_back(i);
        }
        result.push_back("*");
        result.push_back("*");
        for (const std::string& i: x.pieces[0]) //R
        {
            result.push_back(i);
        }
        for (const std::string& i: x.pieces[0]) //R
        {
            result.push_back(i);
        }
        for (const std::string& i: x.pieces[0]) //R
        {
            result.push_back(i);
        }
        results.push_back(result);

        //R(0)
        result.clear();
        for (size_t i = 0; i < x.pieces[0].size(); i++)
        {
            if (x.pieces[0][i] == "x0")
            {
                result.push_back("0");
            }
            else
            {
                result.push_back(x.pieces[0][i]);
            }
        }
        results.push_back(result);

        //- R(∞) sqrt mu
        result.clear();
        result.push_back("-");
        for (size_t i = 0; i < x.pieces[0].size(); i++)
        {
            if (x.pieces[0][i] == "x0")
            {
                result.push_back(infty);
            }
            else
            {
                result.push_back(x.pieces[0][i]);
            }
        }
        result.push_back("sqrt");
        result.push_back(mu);
        results.push_back(result);
    }
    else if (x.expression_type == "postfix")
    {
        //1 2 / R'' * 1 2 r * / R' * + mu S S * 2 r r * * / - R * + R R * R * -
        result.push_back("1");
        result.push_back("2");
        result.push_back("/");
        x.derivePostfix(0, x.pieces[0].size()-1, "x0", x.pieces[0], grasp);
        R_prime = x.derivat;
        x.derivePostfix(0, R_prime.size()-1, "x0", R_prime, grasp); //derivat will store second derivative of R_prime
        for (const std::string& i: x.derivat) //R''
        {
            result.push_back(i);
        }
        result.push_back("*");
        result.push_back("1");
        result.push_back("2");
        result.push_back("x0"); //r
        result.push_back("*");
        result.push_back("/");
        for (const std::string& i: R_prime) //R'
        {
            result.push_back(i);
        }
        result.push_back("*");
        result.push_back("+");
        result.push_back(mu);
        result.push_back(S);
        result.push_back(S);
        result.push_back("*");
        result.push_back("2");
        result.push_back("x0"); //r
        result.push_back("x0"); //r
        result.push_back("*");
        result.push_back("*");
        result.push_back("/");
        result.push_back("-");
        for (const std::string& i: x.pieces[0]) //R
        {
            result.push_back(i);
        }
        result.push_back("*");
        result.push_back("+");
        for (const std::string& i: x.pieces[0]) //R
        {
            result.push_back(i);
        }
        for (const std::string& i: x.pieces[0]) //R
        {
            result.push_back(i);
        }
        result.push_back("*");
        for (const std::string& i: x.pieces[0]) //R
        {
            result.push_back(i);
        }
        result.push_back("*");
        result.push_back("-");
        results.push_back(result);

        //R(0)
        result.clear();
        for (size_t i = 0; i < x.pieces[0].size(); i++)
        {
            if (x.pieces[0][i] == "x0")
            {
                result.push_back("0");
            }
            else
            {
                result.push_back(x.pieces[0][i]);
            }
        }
        results.push_back(result);

        //R(∞) mu sqrt -
        result.clear();

        for (size_t i = 0; i < x.pieces[0].size(); i++)
        {
            if (x.pieces[0][i] == "x0")
            {
                result.push_back(infty);
            }
            else
            {
                result.push_back(x.pieces[0][i]);
            }
        }

        result.push_back(mu);
        result.push_back("sqrt");
        result.push_back("-");
        results.push_back(result);
    }
    return results;
}

/*
 Infix: μ*f + ν*f*f - f*f*f - f - 2*∂^2f/∂r^2 - ∂^4f/∂r^4 - ((1/r) * ((2*(∂^3f/∂r^3)) + ((1/r)*(∂^2f/∂r^2)) - ((1/(r*r))*(∂f/∂r)) + ((1/(r*r))*(∂^3f/∂θ^2∂r)) - ((2/(r*r*r))*(∂^2f/∂θ^2)) + (2*(∂f/∂r)))) - ((1/(r*r)) * ((2*(∂^4f/∂θ^2∂r^2)) + ((1/r)*(∂^3f/∂θ^2∂r)) + ((1/(r*r))*(∂^4f/∂θ^4)) - (2*(∂^2f/∂r^2)) + (2*(∂^2f/∂θ^2)))) - ((2/(r*r*r)) * ((∂f/∂r) - (2*(∂^3f/∂θ^2∂r)) + ((3/r)*(∂^2f/∂θ^2))))
Postfix: μ f * ν f * f * f f f * * - + f - 2 ∂^2f/∂r^2 * - ∂^4f/∂r^4 - 2 ∂^3f/∂r^3 * ∂^2f/∂r^2 r / + (∂f/∂r) r r * / - (∂^3f/∂θ^2∂r) r r * / 2 ∂^2f/∂r^2 * r r * r * / - 2 ∂f/∂r * + + r / - 2 ∂^4f/∂θ^2∂r^2 * ∂^3f/∂θ^2∂r r / + (∂^4f/∂θ^4) r r * / + 2 ∂^2f/∂r^2 * - 2 ∂^2f/∂θ^2 * + r r * / - 2 r r * r * / ∂f/∂r 2 ∂^3f/∂θ^2∂r * - 3 r / ∂^2f/∂θ^2 * + * -

 Infix: f(r, θ=2*π) - f(r, θ = 0)
 Postfix: f(r, θ=2*π) f(r, θ = 0) -
 
 Infix: ∂f/∂θ(r, θ=2*π) - ∂f/∂θ(r, θ = 0)
 Postfix: ∂f/∂θ(r, θ=2*π) ∂f/∂θ(r, θ = 0) -

{x0: r, x1: θ, x2: μ, x3: ν}
{x.pieces[0]: f}
*/
std::vector<std::vector<std::string>> SwiftHohenberg(Board& x, bool fit)
{
//    puts("called SwiftHohenberg");
    /*
    Depth = 7:
        Best score = 0.000155487, SNE = 6430.41
        Squared-norm error for each equation: 6402.17 28.2345 0.00012597
        Best expression = (((((((0.010000 + x0) + sech(x1)) ^ (sech(x0) + 11.13)) * 2.714063472005533e-13) + ((((x0 ^ 6.283190) * 1e-08) + 0.01) + 0.7419039201568504)) - (((0.9998848754538172 ^ (x0 ^ 4.029999999999999)) * (sin(~(x1)) * 0.9972802451715356)) * (0.8912763105205549 * cos(asin(cos(x0)))))) - (((0.00010591201460945816 * ((0.28580222883407974 ^ (x0 + 10.000000)) * (1 - sin(x1)))) + ((4.692820413780688e-06 * (12.56638 * (x1 * 2))) + -0.07517032657318981)) + ((((x0 ^ 0.9999500004166653) / ((0.010000 + x0) + 1)) ^ ((sin(x0) + (0.010000 + x0)) * (sin(x1) + (10.000000 / x0)))) + 0.06683273758330441)))
        Best expression (original format) = 0.010000 x0 + x1 sech + x0 sech 11.13 + ^ 2.714063472005533e-13 * x0 6.283190 ^ 1e-08 * 0.01 + 0.7419039201568504 + + 0.9998848754538172 x0 4.029999999999999 ^ ^ x1 ~ sin 0.9972802451715356 * * 0.8912763105205549 x0 cos asin cos * * - 0.00010591201460945816 0.28580222883407974 x0 10.000000 + ^ 1 x1 sin - * * 4.692820413780688e-06 12.56638 x1 2 * * * -0.07517032657318981 + + x0 0.9999500004166653 ^ 0.010000 x0 + 1 + / x0 sin 0.010000 x0 + + x1 sin 10.000000 x0 / + * ^ 0.06683273758330441 + + -
    Depth = 8:
        Best score = 0.000324862, SNE = 3077.23
        Squared-norm error for each equation: 3047.7 29.5272 0.000136635
        Best expression = (((((((x0 + -0.01) + sech(x1)) ^ 11.156528193614346) * 2.714063572022206e-13) + (((((0.010000 + x0) ^ 6.29319) * 1e-08) + 0.0100003333566687) + 0.7493736126143709)) - (((0.9998848754538172 ^ (arcsin(tanh(x0)) / (0.7615941559557649 / (x0 ^ 4)))) * (sin(~((6.283190 + x1))) * 0.9171523356672744)) * (0.7827863849639187 * cos(asin(cos(x0)))))) - (((((x0 + (0.010000166674167114 + x0)) ^ ((0.003734854911714874 ^ (6.283190 / x0)) + 7.570169558264211)) * ((0.28580222883407974 ^ ((0.010000 + x0) + 10.01)) * (((0.010000 ^ x0) + 1.03) - sin(x1)))) + (((-6.1759665127829875 + (-10 + (x1 + x1))) / ((x1 / 0.005) - 1.9195169107150692e+06)) + -0.06767485271943648)) + (((((0.2658022288340797 + x0) ^ 0.9801980198019802) / (x0 + 1.517923178056138)) ^ ((sin((0.010000 + x0)) + (0.03661899347368653 + x0)) * (sin(x1) + (10.01 / (0.010000 + x0))))) + -0.01842414214696351)))
        Best expression (original format) = x0 -0.01 + x1 sech + 11.156528193614346 ^ 2.714063572022206e-13 * 0.010000 x0 + 6.29319 ^ 1e-08 * 0.0100003333566687 + 0.7493736126143709 + + 0.9998848754538172 x0 tanh arcsin 0.7615941559557649 x0 4 ^ / / ^ 6.283190 x1 + ~ sin 0.9171523356672744 * * 0.7827863849639187 x0 cos asin cos * * - x0 0.010000166674167114 x0 + + 0.003734854911714874 6.283190 x0 / ^ 7.570169558264211 + ^ 0.28580222883407974 0.010000 x0 + 10.01 + ^ 0.010000 x0 ^ 1.03 + x1 sin - * * -6.1759665127829875 -10 x1 x1 + + + x1 0.005 / 1.9195169107150692e+06 - / -0.06767485271943648 + + 0.2658022288340797 x0 + 0.9801980198019802 ^ x0 1.517923178056138 + / 0.010000 x0 + sin 0.03661899347368653 x0 + + x1 sin 10.01 0.010000 x0 + / + * ^ -0.01842414214696351 + + -
     Depth = 9:
        Best score = 0.000330276, SNE = 3026.77
        Squared-norm error for each equation: 3009.42 8.13686 9.21243
        Best expression = (((((((x0 + -0.01) + sech((x1 - 1))) ^ 11.156528193614346) * 2.714063572022206e-13) + (((((0.01 + x0) ^ 6.29319) * 1e-08) + 0.010000333356672894) + 0.7493736126143709)) - (((0.9998848754538172 ^ (arcsin(tanh(x0)) / (0.7615941559557649 / (x0 ^ 4)))) * (sin(~((6.28319 + x1))) * 0.9171523356672744)) * (0.7827863849639187 * cos(asin(cos(x0)))))) - (((((x0 + (0.010000166674167114 + (4.692820413780688e-06 + x0))) ^ ((0.003734854911714874 ^ (6.28319 / (0.010000 + x0))) + 7.570169558264211)) * ((0.28580222883407974 ^ ((0.01 + x0) + 10.01)) * (((0.01 ^ x0) + 1) - sin(x1)))) + (((200 + (x1 + x1)) / (((-0.01 + x1) / 0.005) - 1.9195319107150692e+06)) + -0.06767485271943648)) + (((((0.2658022288340797 + (0.00999966667999946 + x0)) ^ 0.9801980198019802) / ((0.01 + x0) + 1.527923178056138)) ^ ((sin((0.01 + x0)) + (0.03661899347368653 + x0)) * (sin(x1) + (10.01 / (0.01 + x0))))) + -0.014598487375910817)))
        Best expression (original format) = x0 -0.01 + x1 1 - sech + 11.156528193614346 ^ 2.714063572022206e-13 * 0.01 x0 + 6.29319 ^ 1e-08 * 0.010000333356672894 + 0.7493736126143709 + + 0.9998848754538172 x0 tanh arcsin 0.7615941559557649 x0 4 ^ / / ^ 6.28319 x1 + ~ sin 0.9171523356672744 * * 0.7827863849639187 x0 cos asin cos * * - x0 0.010000166674167114 4.692820413780688e-06 x0 + + + 0.003734854911714874 6.28319 0.010000 x0 + / ^ 7.570169558264211 + ^ 0.28580222883407974 0.01 x0 + 10.01 + ^ 0.01 x0 ^ 1 + x1 sin - * * 200 x1 x1 + + -0.01 x1 + 0.005 / 1.9195319107150692e+06 - / -0.06767485271943648 + + 0.2658022288340797 0.00999966667999946 x0 + + 0.9801980198019802 ^ 0.01 x0 + 1.527923178056138 + / 0.01 x0 + sin 0.03661899347368653 x0 + + x1 sin 10.01 0.01 x0 + / + * ^ -0.014598487375910817 + + -
     ```
x = "(((((((-0.02 + (x0 + 0.010000)) + sech(x1)) ^ 11.119909200140661) * 2.714063572022206e-13) + (((((0.010000 + x0) ^ 6.28319) * 1e-08) + 0.0025) + 0.7456387750685652)) - (((0.9998848754538172 ^ (arcsin(tanh(x0)) / (1 / (x0 ^ 4)))) * (sin(~((6.283190 + x1))) * 0.940049833749168)) * (0.7927863849639187 * cos(asin(cos(x0)))))) - ((((((0.01 + x0) + (0.009999833334166664 + x0)) ^ 7.50905609893065) * ((0.28580222883407974 ^ (x0 + 10)) * (1.02 - sin(x1)))) + (0.003734854911714874 + (((2.302585092994046 + (0.010000 + x1)) / ((x1 / 10.000000) - 1.9195083249399762e+06)) + -0.0576701598990227))) + (((((0.02 + x0) ^ 0.9801980198019802) / ((-0.01 + x0) + 1.5079230113819708)) ^ ((sin((6.283190 + x0)) + x0) * (sin((6.283190 + x1)) + (10.01 / x0)))) + -0.0132334444522229)))"
print(x.replace("x0","r").replace("x1","theta").replace("^","**").replace("~","-"))
     */
    
    static std::atomic<bool> added_additive{false};
    static std::once_flag init_flag;
    static std::vector<std::vector<std::string>> additive_results(3);
    static std::vector<Eigen::VectorXd> f_res(3);
    std::vector<std::vector<std::string>> results;
    results.reserve(3);
    assert((x.num_diff_eqns == 3) && "SwiftHohenberg is a system of 3 equations!");
    thread_local std::vector<std::string> result, dfdr1, d2fdr2, d3fdr3, d4fdr4,
    dfdtheta1, d2fdtheta2, d3fdtheta3, d4fdtheta4, df3dtheta2dr1,
    prefac_temp;
    result.clear(); dfdr1.clear(); d2fdr2.clear(); d3fdr3.clear(); d4fdr4.clear();
    dfdtheta1.clear(); d2fdtheta2.clear(); d3fdtheta3.clear(); d4fdtheta4.clear(); df3dtheta2dr1.clear();
    prefac_temp.clear();
    
    result.reserve(500);
    thread_local std::vector<int> grasp;
    thread_local bool prefactors_computed = false;
    if (!prefactors_computed)
    {
        prefac_temp.reserve(20);
        assert(((Board::__num_features == 2) || (Board::__num_features == 4)) && "SwiftHohenberg must have either 2 features (r, theta) or 4 features (r, theta, mu, nu)");
    }
    grasp.clear();
    grasp.reserve(500);

    std::string mu = ((Board::__num_features == 2) ? "1" : "x2");
    std::string nu = ((Board::__num_features == 2) ? "1" : "x3");

    std::string infty = std::to_string(DBL_MAX);
       
    // First, compute the Swift-Hohenberg residual of "f0" if provided (namely, as additiveCorrections)
    if (x.add_additive)
    {
        // open -a Google\ Chrome SwiftHohenbergBoosting.png
        std::call_once(init_flag, [&]()
        {
            std::scoped_lock str_lock(Board::thread_locker);
            if (!added_additive)
            {
                if (x.expression_type == "prefix")
                {
                    throw std::invalid_argument("Prefix Additive Corrections not implemented yet for this SwiftHohenberg function!");
                }
                else if (x.expression_type == "postfix")
                {
                    //μ f * ν f * f * f f f * * - + f - 2 ∂^2f/∂r^2 * - ∂^4f/∂r^4 - 2 ∂^3f/∂r^3 * ∂^2f/∂r^2 r / + (∂f/∂r) r r * / - (∂^3f/∂θ^2∂r) r r * / 2 ∂^2f/∂r^2 * r r * r * / - 2 ∂f/∂r * + + r / - 2 ∂^4f/∂θ^2∂r^2 * ∂^3f/∂θ^2∂r r / + (∂^4f/∂θ^4) r r * / + 2 ∂^2f/∂r^2 * - 2 ∂^2f/∂θ^2 * + r r * / - 2 r r * r * / ∂f/∂r 2 ∂^3f/∂θ^2∂r * - 3 r / ∂^2f/∂θ^2 * + * -
                    additive_results[0].push_back(mu); // μ
                    for (const std::string& i: x.additiveCorrections[0]) // f
                    {
                        additive_results[0].push_back(i);
                    }
                    additive_results[0].push_back("*"); // *
                    additive_results[0].push_back(nu); // ν
                    for (const std::string& i: x.additiveCorrections[0]) // f
                    {
                        additive_results[0].push_back(i);
                    }
                    additive_results[0].push_back("*"); // *
                    for (const std::string& i: x.additiveCorrections[0]) // f
                    {
                        additive_results[0].push_back(i);
                    }
                    additive_results[0].push_back("*"); // *
                    for (const std::string& i: x.additiveCorrections[0]) // f
                    {
                        additive_results[0].push_back(i);
                    }
                    for (const std::string& i: x.additiveCorrections[0]) // f
                    {
                        additive_results[0].push_back(i);
                    }
                    for (const std::string& i: x.additiveCorrections[0]) // f
                    {
                        additive_results[0].push_back(i);
                    }
                    additive_results[0].push_back("*"); // *
                    additive_results[0].push_back("*"); // *
                    additive_results[0].push_back("-"); // -
                    additive_results[0].push_back("+"); // +
                    for (const std::string& i: x.additiveCorrections[0]) // f
                    {
                        additive_results[0].push_back(i);
                    }
                    additive_results[0].push_back("-"); // -
                    additive_results[0].push_back("2"); // 2
                    x.derivePostfix(0, x.additiveCorrections[0].size()-1, "x0", x.additiveCorrections[0], grasp);
                    dfdr1 = x.derivat;
                    x.derivePostfix(0, dfdr1.size()-1, "x0", dfdr1, grasp);
                    d2fdr2 = x.derivat;
                    for (const std::string& i: d2fdr2) // ∂^2f/∂r^2
                    {
                        additive_results[0].push_back(i);
                    }
                    additive_results[0].push_back("*"); // *
                    additive_results[0].push_back("-"); // -
                    x.derivePostfix(0, d2fdr2.size()-1, "x0", d2fdr2, grasp);
                    d3fdr3 = x.derivat;
                    x.derivePostfix(0, d3fdr3.size()-1, "x0", d3fdr3, grasp);
                    d4fdr4 = x.derivat;
                    for (const std::string& i: d4fdr4) // ∂^4f/∂r^4
                    {
                        additive_results[0].push_back(i);
                    }
                    additive_results[0].push_back("-"); // -
                    additive_results[0].push_back("2"); // 2
                    for (const std::string& i: d3fdr3) // ∂^3f/∂r^3
                    {
                        additive_results[0].push_back(i);
                    }
                    additive_results[0].push_back("*"); // *
                    for (const std::string& i: d2fdr2) // ∂^2f/∂r^2
                    {
                        additive_results[0].push_back(i);
                    }
                    additive_results[0].push_back("x0"); // r
                    additive_results[0].push_back("/"); // /
                    additive_results[0].push_back("+"); // +
                    for (const std::string& i: dfdr1) // ∂f/∂r
                    {
                        additive_results[0].push_back(i);
                    }
                    if (!prefactors_computed)
                    {
                        prefac_temp = {"x0", "x0", "*"};
                        x.subs_dict["r_squared"] = x.expression_evaluator(x.params, prefac_temp);
                    }
                    additive_results[0].push_back("r_squared"); // r r *
                    additive_results[0].push_back("/"); // /
                    additive_results[0].push_back("-"); // -
                    x.derivePostfix(0, x.additiveCorrections[0].size()-1, "x1", x.additiveCorrections[0], grasp);
                    dfdtheta1 = x.derivat;
                    x.derivePostfix(0, dfdtheta1.size()-1, "x1", dfdtheta1, grasp);
                    d2fdtheta2 = x.derivat;
                    x.derivePostfix(0, d2fdtheta2.size()-1, "x0", d2fdtheta2, grasp);
                    df3dtheta2dr1 = x.derivat;
                    for (const std::string& i: df3dtheta2dr1) // ∂^3f/∂θ^2∂r
                    {
                        additive_results[0].push_back(i);
                    }
                    additive_results[0].push_back("r_squared"); // r r *
                    additive_results[0].push_back("/"); // /
                    additive_results[0].push_back("2"); // 2
                    for (const std::string& i: d2fdr2) // ∂^2f/∂r^2
                    {
                        additive_results[0].push_back(i);
                    }
                    additive_results[0].push_back("*"); // *
                    if (!prefactors_computed)
                    {
                        prefac_temp = {"x0", "x0", "*", "x0", "*"};
                        x.subs_dict["r_cubed"] = x.expression_evaluator(x.params, prefac_temp);
                    }
                    additive_results[0].push_back("r_cubed"); // r r * r *
                    additive_results[0].push_back("/"); // /
                    additive_results[0].push_back("-"); // -
                    additive_results[0].push_back("2"); // 2
                    for (const std::string& i: dfdr1) // ∂f/∂r
                    {
                        additive_results[0].push_back(i);
                    }
                    additive_results[0].push_back("*"); // *
                    additive_results[0].push_back("+"); // +
                    additive_results[0].push_back("+"); // +
                    additive_results[0].push_back("x0"); // r
                    additive_results[0].push_back("/"); // /
                    additive_results[0].push_back("-"); // -
                    additive_results[0].push_back("2"); // 2
                    x.derivePostfix(0, df3dtheta2dr1.size()-1, "x0", df3dtheta2dr1, grasp);
                    for (const std::string& i: x.derivat) //∂^4f/∂θ^2∂r^2
                    {
                        additive_results[0].push_back(i);
                    }
                    additive_results[0].push_back("*"); // *
                    for (const std::string& i: df3dtheta2dr1) // ∂^3f/∂θ^2∂r
                    {
                        additive_results[0].push_back(i);
                    }
                    additive_results[0].push_back("x0"); // r
                    additive_results[0].push_back("/"); // /
                    additive_results[0].push_back("+"); // +
                    x.derivePostfix(0, d2fdtheta2.size()-1, "x1", d2fdtheta2, grasp);
                    d3fdtheta3 = x.derivat;
                    x.derivePostfix(0, d3fdtheta3.size()-1, "x1", d3fdtheta3, grasp);
                    d4fdtheta4 = x.derivat;
                    for (const std::string& i: d4fdtheta4) // ∂^4f/∂θ^4
                    {
                        additive_results[0].push_back(i);
                    }
                    additive_results[0].push_back("r_squared"); // r r *
                    additive_results[0].push_back("/"); // /
                    additive_results[0].push_back("+"); // +
                    additive_results[0].push_back("2"); // 2
                    for (const std::string& i: d2fdr2) // ∂^2f/∂r^2
                    {
                        additive_results[0].push_back(i);
                    }
                    additive_results[0].push_back("*"); // *
                    additive_results[0].push_back("-"); // -
                    additive_results[0].push_back("2"); // 2
                    for (const std::string& i: d2fdtheta2) // ∂^2f/∂θ^2
                    {
                        additive_results[0].push_back(i);
                    }
                    additive_results[0].push_back("*"); // *
                    additive_results[0].push_back("+"); // +
                    additive_results[0].push_back("r_squared"); // r r *
                    additive_results[0].push_back("/"); // /
                    additive_results[0].push_back("-"); // -
                    additive_results[0].push_back("2"); // 2
                    additive_results[0].push_back("r_cubed"); // r r * r *
                    additive_results[0].push_back("/"); // /
                    for (const std::string& i: dfdr1) // ∂f/∂r
                    {
                        additive_results[0].push_back(i);
                    }
                    additive_results[0].push_back("2"); // 2
                    for (const std::string& i: df3dtheta2dr1) // ∂^3f/∂θ^2∂r
                    {
                        additive_results[0].push_back(i);
                    }
                    additive_results[0].push_back("*"); // *
                    additive_results[0].push_back("-"); // -
                    if (!prefactors_computed)
                    {
                        prefac_temp = {"3", "x0", "/"};
                        x.subs_dict["3_over_r"] = x.expression_evaluator(x.params, prefac_temp);
                    }
                    additive_results[0].push_back("3_over_r"); // 3 r /
                    for (const std::string& i: d2fdtheta2) // ∂^2f/∂θ^2
                    {
                        additive_results[0].push_back(i);
                    }
                    additive_results[0].push_back("*"); // *
                    additive_results[0].push_back("+"); // +
                    additive_results[0].push_back("*"); // *
                    additive_results[0].push_back("-"); // -
                    //f(r, θ=2*π) f(r, θ = 0) -
                    for (const std::string& i: x.additiveCorrections[0]) //f(r, θ=2*π)
                    {
                        if (i == "x1")
                        {
                            additive_results[1].push_back("6.283185307179586");
                        }
                        else
                        {
                            additive_results[1].push_back(i);
                        }
                    }
                    for (const std::string& i: x.additiveCorrections[0]) //f(r, θ = 0)
                    {
                        if (i == "x1")
                        {
                            additive_results[1].push_back("0");
                        }
                        else
                        {
                            additive_results[1].push_back(i);
                        }
                    }
                    additive_results[1].push_back("-");
                    //∂f/∂θ(r, θ=2*π) ∂f/∂θ(r, θ = 0) -
                    for (const std::string& i: dfdtheta1) //∂f/∂θ(r, θ=2*π)
                    {
                        if (i == "x1")
                        {
                            additive_results[2].push_back("6.283185307179586");
                        }
                        else
                        {
                            additive_results[2].push_back(i);
                        }
                    }
                    for (const std::string& i: dfdtheta1) //∂f/∂θ(r, θ = 0)
                    {
                        if (i == "x1")
                        {
                            additive_results[2].push_back("0");
                        }
                        else
                        {
                            additive_results[2].push_back(i);
                        }
                    }
                    additive_results[2].push_back("-");
                }
                puts("Evaluating additive_results");
                f_res[0] = x.expression_evaluator(x.params, additive_results[0]);
                puts("Done evaluating additive_results[0]");
                f_res[1] = x.expression_evaluator(x.params, additive_results[1]);
                puts("Done evaluating additive_results[1]");
                f_res[2] = x.expression_evaluator(x.params, additive_results[2]);
                added_additive = true;
                puts("Done evaluating additive_results");
            }
        });
        if (!prefactors_computed)
        {
            assert(added_additive);
            x.subs_dict["fres0"] = f_res[0];
            x.subs_dict["fres1"] = f_res[1];
            x.subs_dict["fres2"] = f_res[2];
            x.subs_dict["f0"] = x.expression_evaluator(x.params, x.additiveCorrections[0]);
        }
    }
    
    //Now, compute the main SH equations for the function "f" being sought
    if (x.expression_type == "prefix")
    {
        //- - - - - - + * μ f - * * ν f f * f * f f f * 2 ∂^2f/∂r^2 ∂^4f/∂r^4 / + - + * 2 ∂^3f/∂r^3 / ∂^2f/∂r^2 r / (∂f/∂r) * r r + - / (∂^3f/∂θ^2∂r) * r r / * 2 ∂^2f/∂r^2 * * r r r * 2 ∂f/∂r r / + - + + * 2 ∂^4f/∂θ^2∂r^2 / ∂^3f/∂θ^2∂r r / (∂^4f/∂θ^4) * r r * 2 ∂^2f/∂r^2 * 2 ∂^2f/∂θ^2 * r r * / 2 * * r r r + - ∂f/∂r * 2 ∂^3f/∂θ^2∂r * / 3 r ∂^2f/∂θ^2
        throw std::invalid_argument("Prefix not implemented yet for this SwiftHohenberg function!");
    }
    else if (x.expression_type == "postfix")
    {
        //μ f * ν f * f * f f f * * - + f - 2 ∂^2f/∂r^2 * - ∂^4f/∂r^4 - 2 ∂^3f/∂r^3 * ∂^2f/∂r^2 r / + (∂f/∂r) r r * / - (∂^3f/∂θ^2∂r) r r * / 2 ∂^2f/∂r^2 * r r * r * / - 2 ∂f/∂r * + + r / - 2 ∂^4f/∂θ^2∂r^2 * ∂^3f/∂θ^2∂r r / + (∂^4f/∂θ^4) r r * / + 2 ∂^2f/∂r^2 * - 2 ∂^2f/∂θ^2 * + r r * / - 2 r r * r * / ∂f/∂r 2 ∂^3f/∂θ^2∂r * - 3 r / ∂^2f/∂θ^2 * + * -
        result.push_back(mu); // μ
        if (fit) //need to push_back each token since eval(f) will change as the consts in f are optimized
        {
            for (const std::string& i: x.pieces[0]) // f
            {
                result.push_back(i);
            }
        }
        else
        {
            x.subs_dict["f"] = x.expression_evaluator(x.params, x.pieces[0]);
            result.push_back("f"); // f
        }
        result.push_back("*"); // *
        result.push_back(nu); // ν
        if (fit)
        {
            for (const std::string& i: x.pieces[0]) // f
            {
                result.push_back(i);
            }
        }
        else
        {
            result.push_back("f"); // f
        }
        result.push_back("*"); // *
        if (fit)
        {
            for (const std::string& i: x.pieces[0]) // f
            {
                result.push_back(i);
            }
        }
        else
        {
            result.push_back("f"); // f
        }
        result.push_back("*"); // *
        if (fit) //need to push_back each token since eval(f) will change as the consts in f are optimized
        {
            for (const std::string& i: x.pieces[0]) // f
            {
                result.push_back(i);
            }
            for (const std::string& i: x.pieces[0]) // f
            {
                result.push_back(i);
            }
            for (const std::string& i: x.pieces[0]) // f
            {
                result.push_back(i);
            }
        }
        else
        {
            result.push_back("f"); // f
            result.push_back("f"); // f
            result.push_back("f"); // f
        }
        result.push_back("*"); // *
        result.push_back("*"); // *
        result.push_back("-"); // -
        result.push_back("+"); // +
        if (fit) //need to push_back each token since eval(f) will change as the consts in f are optimized
        {
            for (const std::string& i: x.pieces[0]) // f
            {
                result.push_back(i);
            }
        }
        else
        {
            result.push_back("f"); // f
        }
        result.push_back("-"); // -
        result.push_back("2"); // 2
        x.derivePostfix(0, x.pieces[0].size()-1, "x0", x.pieces[0], grasp);
        dfdr1 = x.derivat;
        if (!fit)
        {
            x.subs_dict["dfdr1"] = x.expression_evaluator(x.params, dfdr1);
        }
        x.derivePostfix(0, dfdr1.size()-1, "x0", dfdr1, grasp);
        d2fdr2 = x.derivat;
        if (fit)
        {
            for (const std::string& i: d2fdr2) // ∂^2f/∂r^2
            {
                result.push_back(i);
            }
        }
        else
        {
            x.subs_dict["d2fdr2"] = x.expression_evaluator(x.params, d2fdr2);
            result.push_back("d2fdr2"); // ∂^2f/∂r^2
        }
        result.push_back("*"); // *
        result.push_back("-"); // -
        x.derivePostfix(0, d2fdr2.size()-1, "x0", d2fdr2, grasp);
        d3fdr3 = x.derivat;
        x.derivePostfix(0, d3fdr3.size()-1, "x0", d3fdr3, grasp);
        d4fdr4 = x.derivat;
        for (const std::string& i: d4fdr4) // ∂^4f/∂r^4
        {
            result.push_back(i);
        }
        result.push_back("-"); // -
        result.push_back("2"); // 2
        for (const std::string& i: d3fdr3) // ∂^3f/∂r^3
        {
            result.push_back(i);
        }
        result.push_back("*"); // *
        if (fit)
        {
            for (const std::string& i: d2fdr2) // ∂^2f/∂r^2
            {
                result.push_back(i);
            }
        }
        else
        {
            result.push_back("d2fdr2"); // ∂^2f/∂r^2
        }
        result.push_back("x0"); // r
        result.push_back("/"); // /
        result.push_back("+"); // +
        if (fit)
        {
            for (const std::string& i: dfdr1) // ∂f/∂r
            {
                result.push_back(i);
            }
        }
        else
        {
            result.push_back("dfdr1"); // ∂f/∂r
        }
//        result.push_back("x0"); // r
//        result.push_back("x0"); // r
//        result.push_back("*"); // *
        if (!x.subs_dict.count("r_squared"))
        {
            prefac_temp = {"x0", "x0", "*"};
            x.subs_dict["r_squared"] = x.expression_evaluator(x.params, prefac_temp);
        }
        result.push_back("r_squared"); // r r *
        result.push_back("/"); // /
        result.push_back("-"); // -
        x.derivePostfix(0, x.pieces[0].size()-1, "x1", x.pieces[0], grasp);
        dfdtheta1 = x.derivat;
        x.derivePostfix(0, dfdtheta1.size()-1, "x1", dfdtheta1, grasp);
        d2fdtheta2 = x.derivat;
        x.derivePostfix(0, d2fdtheta2.size()-1, "x0", d2fdtheta2, grasp);
        df3dtheta2dr1 = x.derivat;
        if (fit)
        {
            for (const std::string& i: df3dtheta2dr1) // ∂^3f/∂θ^2∂r
            {
                result.push_back(i);
            }
        }
        else
        {
            x.subs_dict["df3dtheta2dr1"] = x.expression_evaluator(x.params, df3dtheta2dr1);
            result.push_back("df3dtheta2dr1"); // ∂^3f/∂θ^2∂r
        }
//        result.push_back("x0"); // r
//        result.push_back("x0"); // r
//        result.push_back("*"); // *
        result.push_back("r_squared"); // r r *
        result.push_back("/"); // /
        result.push_back("2"); // 2
        if (fit)
        {
            for (const std::string& i: d2fdr2) // ∂^2f/∂r^2
            {
                result.push_back(i);
            }
        }
        else
        {
            result.push_back("d2fdr2"); // ∂^2f/∂r^2
        }
        result.push_back("*"); // *
//        result.push_back("x0"); // r
//        result.push_back("x0"); // r
//        result.push_back("*"); // *
//        result.push_back("x0"); // r
//        result.push_back("*"); // *
        if (!x.subs_dict.count("r_cubed"))
        {
            prefac_temp = {"x0", "x0", "*", "x0", "*"};
            x.subs_dict["r_cubed"] = x.expression_evaluator(x.params, prefac_temp);
        }
        result.push_back("r_cubed"); // r r * r *
        result.push_back("/"); // /
        result.push_back("-"); // -
        result.push_back("2"); // 2
        if (fit)
        {
            for (const std::string& i: dfdr1) // ∂f/∂r
            {
                result.push_back(i);
            }
        }
        else
        {
            result.push_back("dfdr1"); // ∂f/∂r
        }
        result.push_back("*"); // *
        result.push_back("+"); // +
        result.push_back("+"); // +
        result.push_back("x0"); // r
        result.push_back("/"); // /
        result.push_back("-"); // -
        result.push_back("2"); // 2
        x.derivePostfix(0, df3dtheta2dr1.size()-1, "x0", df3dtheta2dr1, grasp);
        for (const std::string& i: x.derivat) //∂^4f/∂θ^2∂r^2
        {
            result.push_back(i);
        }
        result.push_back("*"); // *
        if (fit)
        {
            for (const std::string& i: df3dtheta2dr1) // ∂^3f/∂θ^2∂r
            {
                result.push_back(i);
            }
        }
        else
        {
            result.push_back("df3dtheta2dr1"); // ∂^3f/∂θ^2∂r
        }
        result.push_back("x0"); // r
        result.push_back("/"); // /
        result.push_back("+"); // +
        x.derivePostfix(0, d2fdtheta2.size()-1, "x1", d2fdtheta2, grasp);
        d3fdtheta3 = x.derivat;
        x.derivePostfix(0, d3fdtheta3.size()-1, "x1", d3fdtheta3, grasp);
        d4fdtheta4 = x.derivat;
        for (const std::string& i: d4fdtheta4) // ∂^4f/∂θ^4
        {
            result.push_back(i);
        }
//        result.push_back("x0"); // r
//        result.push_back("x0"); // r
//        result.push_back("*"); // *
        result.push_back("r_squared"); // r r *
        result.push_back("/"); // /
        result.push_back("+"); // +
        result.push_back("2"); // 2
        if (fit)
        {
            for (const std::string& i: d2fdr2) // ∂^2f/∂r^2
            {
                result.push_back(i);
            }
        }
        else
        {
            result.push_back("d2fdr2"); // ∂^2f/∂r^2
        }
        result.push_back("*"); // *
        result.push_back("-"); // -
        result.push_back("2"); // 2
        if (fit)
        {
            for (const std::string& i: d2fdtheta2) // ∂^2f/∂θ^2
            {
                result.push_back(i);
            }
        }
        else
        {
            x.subs_dict["d2fdtheta2"] = x.expression_evaluator(x.params, d2fdtheta2);
            result.push_back("d2fdtheta2"); // ∂^2f/∂θ^2
        }
        result.push_back("*"); // *
        result.push_back("+"); // +
//        result.push_back("x0"); // r
//        result.push_back("x0"); // r
//        result.push_back("*"); // *
        result.push_back("r_squared"); // r r *
        result.push_back("/"); // /
        result.push_back("-"); // -
        result.push_back("2"); // 2
//        result.push_back("x0"); // r
//        result.push_back("x0"); // r
//        result.push_back("*"); // *
//        result.push_back("x0"); // r
//        result.push_back("*"); // *
        result.push_back("r_cubed"); // r r * r *
        result.push_back("/"); // /
        if (fit)
        {
            for (const std::string& i: dfdr1) // ∂f/∂r
            {
                result.push_back(i);
            }
        }
        else
        {
            result.push_back("dfdr1"); // ∂f/∂r
        }
        result.push_back("2"); // 2
        if (fit)
        {
            for (const std::string& i: df3dtheta2dr1) // ∂^3f/∂θ^2∂r
            {
                result.push_back(i);
            }
        }
        else
        {
            result.push_back("df3dtheta2dr1"); // ∂^3f/∂θ^2∂r
        }
        result.push_back("*"); // *
        result.push_back("-"); // -
//        result.push_back("3"); // 3
//        result.push_back("x0"); // r
//        result.push_back("/"); // /
        if (!x.subs_dict.count("3_over_r"))
        {
            prefac_temp = {"3", "x0", "/"};
            x.subs_dict["3_over_r"] = x.expression_evaluator(x.params, prefac_temp);
        }
        result.push_back("3_over_r"); // 3 r /
        if (fit)
        {
            for (const std::string& i: d2fdtheta2) // ∂^2f/∂θ^2
            {
                result.push_back(i);
            }
        }
        else
        {
            result.push_back("d2fdtheta2"); // ∂^2f/∂θ^2
        }
        result.push_back("*"); // *
        result.push_back("+"); // +
        result.push_back("*"); // *
        result.push_back("-"); // -
        results.push_back(result);
        result.clear();
        //f(r, θ=2*π) f(r, θ = 0) -
        for (const std::string& i: x.pieces[0]) //f(r, θ=2*π)
        {
            if (i == "x1")
            {
                result.push_back("6.283185307179586");
            }
            else
            {
                result.push_back(i);
            }
        }
        for (const std::string& i: x.pieces[0]) //f(r, θ = 0)
        {
            if (i == "x1")
            {
                result.push_back("0");
            }
            else
            {
                result.push_back(i);
            }
        }
        result.push_back("-");
        results.push_back(result);
        result.clear();
        //∂f/∂θ(r, θ=2*π) ∂f/∂θ(r, θ = 0) -
        for (const std::string& i: dfdtheta1) //∂f/∂θ(r, θ=2*π)
        {
            if (i == "x1")
            {
                result.push_back("6.283185307179586");
            }
            else
            {
                result.push_back(i);
            }
        }
        for (const std::string& i: dfdtheta1) //∂f/∂θ(r, θ = 0)
        {
            if (i == "x1")
            {
                result.push_back("0");
            }
            else
            {
                result.push_back(i);
            }
        }
        result.push_back("-");
        results.push_back(result);
        result.clear();
    }
    
    // Finally, add everything together for the "f0 + f" candidate solution!
    if (x.add_additive)
    {
        assert(results.size() == 3);
        assert(x.subs_dict.count("f0"));
        assert(x.subs_dict.count("fres0"));
        assert(x.subs_dict.count("fres1"));
        assert(x.subs_dict.count("fres2"));
        assert(x.subs_dict.count("f"));
        assert(x.subs_dict.count("r_squared"));
        //f0 * (2*nu*f - 3*f0*f - 3*f*f)
        if (x.expression_type == "prefix")
        {
            throw std::invalid_argument("Prefix not implemented yet for this SwiftHohenberg function!");
        }
        else if (x.expression_type == "postfix")
        {
            
            //f0 2 ν * f * 3 f0 * f * - 3 f * f * - *
            results[0].push_back("f0"); //f0
            results[0].push_back("2"); //2
            results[0].push_back(nu); //ν
            results[0].push_back("*"); //*
            if (fit)
            {
                for (const std::string& i: x.pieces[0]) // f
                {
                    results[0].push_back(i);
                }
            }
            else
            {
                results[0].push_back("f"); // f
            }
            results[0].push_back("*"); //*
            results[0].push_back("3"); //3
            results[0].push_back("f0"); //f0
            results[0].push_back("*"); //*
            if (fit)
            {
                for (const std::string& i: x.pieces[0]) // f
                {
                    results[0].push_back(i);
                }
            }
            else
            {
                results[0].push_back("f"); // f
            }
            results[0].push_back("*"); //*
            results[0].push_back("-"); //-
            results[0].push_back("3"); //3
            if (fit)
            {
                for (const std::string& i: x.pieces[0]) // f
                {
                    results[0].push_back(i);
                }
            }
            else
            {
                results[0].push_back("f"); // f
            }
            results[0].push_back("*"); //*
            if (fit)
            {
                for (const std::string& i: x.pieces[0]) // f
                {
                    results[0].push_back(i);
                }
            }
            else
            {
                results[0].push_back("f"); // f
            }
            results[0].push_back("*"); //*
            results[0].push_back("-"); //-
            results[0].push_back("*"); //*
            
            //Add "f0 2 ν * f * 3 f0 * f * - 3 f * f * - *" to SH(f)[0]
            results[0].push_back("+"); //+
            
            //Add SH(f0)[0] to results[0]
            results[0].push_back("fres0");
            results[0].push_back("+");
            
            //Add SH(f0)[1] to results[1]
            results[1].push_back("fres1");
            results[1].push_back("+");

            //Add SH(f0)[2] to results[2]
            results[2].push_back("fres2");
            results[2].push_back("+");
        }
    }
    prefactors_computed = true;
//    std::cout << "results = " << results << '\n';
    return results;
}

//x0 -> x, x1 -> y, x2 -> t
std::vector<std::vector<std::string>> TwoDAdvectionDiffusion_1(Board& x, bool fit)
{
    std::vector<std::vector<std::string>> results;
    std::vector<std::string> result;
    result.reserve(100);
    std::vector<int> grasp;
    grasp.reserve(100);
    std::vector<std::string> temp;
    temp.reserve(50);
    std::string kappa = "const0";
    if (x.expression_type == "prefix")
    {
        //- + T_t * - 1 * y y T_x * kappa + T_{xx} T_{yy}
        result.push_back("-"); //-
        result.push_back("+"); //+
        x.derivePrefix(0, x.pieces[0].size()-1, "x2", x.pieces[0], grasp);
        for (const std::string& i: x.derivat) //T_t
        {
            result.push_back(i);
        }
        result.push_back("*");
        result.push_back("-");
        result.push_back("1");
        result.push_back("*");
        result.push_back("x1");
        result.push_back("x1");
        x.derivePrefix(0, x.pieces[0].size()-1, "x0", x.pieces[0], grasp);
        for (const std::string& i: x.derivat) //T_x
        {
            result.push_back(i);
        }
        result.push_back("*"); //*
        result.push_back(kappa); //kappa
        result.push_back("+"); //+
        x.derivePrefix(0, x.pieces[0].size()-1, "x0", x.pieces[0], grasp);
        temp = x.derivat;
        x.derivePrefix(0, temp.size()-1, "x0", temp, grasp);
        for (const std::string& i: x.derivat) //T_xx
        {
            result.push_back(i);
        }
        x.derivePrefix(0, x.pieces[0].size()-1, "x1", x.pieces[0], grasp);
        temp = x.derivat;
        x.derivePrefix(0, temp.size()-1, "x1", temp, grasp);
        for (const std::string& i: x.derivat) //T_yy
        {
            result.push_back(i);
        }
    }
    else if (x.expression_type == "postfix")
    {
        //T_t 1 y y * - T_x * + kappa T_{xx} T_{yy} + * -
        x.derivePostfix(0, x.pieces[0].size()-1, "x2", x.pieces[0], grasp);
        for (const std::string& i: x.derivat) //T_t
        {
            result.push_back(i);
        }
        result.push_back("1");
        result.push_back("x1");
        result.push_back("x1");
        result.push_back("*");
        result.push_back("-");
        x.derivePostfix(0, x.pieces[0].size()-1, "x0", x.pieces[0], grasp);
        for (const std::string& i: x.derivat) //T_x
        {
            result.push_back(i);
        }
        result.push_back("*");
        result.push_back("+"); //+
        result.push_back(kappa); //kappa
        x.derivePostfix(0, x.pieces[0].size()-1, "x0", x.pieces[0], grasp);
        temp = x.derivat;
        x.derivePostfix(0, temp.size()-1, "x0", temp, grasp);
        for (const std::string& i: x.derivat) //T_xx
        {
            result.push_back(i);
        }

        x.derivePostfix(0, x.pieces[0].size()-1, "x1", x.pieces[0], grasp);
        temp = x.derivat;
        x.derivePostfix(0, temp.size()-1, "x1", temp, grasp);
        for (const std::string& i: x.derivat) //T_yy
        {
            result.push_back(i);
        }
        result.push_back("+"); //+
        result.push_back("*"); //*
        result.push_back("-"); //-
    }
    results.push_back(result);
    return results;
}

//x0 -> x, x1 -> y, x2 -> t
std::vector<std::vector<std::string>> TwoDAdvectionDiffusion_2(Board& x, bool fit)
{
    std::vector<std::vector<std::string>> results;
    std::vector<std::string> result;
    result.reserve(100);
    std::vector<int> grasp;
    grasp.reserve(100);
    std::vector<std::string> temp;
    temp.reserve(50);
    std::string kappa = "const0";
    if (x.expression_type == "prefix")
    {
        //- + + T_t * sin * 4 y T_x * cos * 4 x T_y * kappa + T_{xx} T_{yy}
        result.push_back("-"); //-
        result.push_back("+"); //+
        result.push_back("+"); //+

        x.derivePrefix(0, x.pieces[0].size()-1, "x2", x.pieces[0], grasp);
        for (const std::string& i: x.derivat) //T_t
        {
            result.push_back(i);
        }
        result.push_back("*");
        result.push_back("sin");
        result.push_back("*");
        result.push_back("4");
        result.push_back("x1");

        x.derivePrefix(0, x.pieces[0].size()-1, "x0", x.pieces[0], grasp);
        for (const std::string& i: x.derivat) //T_x
        {
            result.push_back(i);
        }

        result.push_back("*");
        result.push_back("cos");
        result.push_back("*");
        result.push_back("4");
        result.push_back("x0");

        x.derivePrefix(0, x.pieces[0].size()-1, "x1", x.pieces[0], grasp);
        for (const std::string& i: x.derivat) //T_y
        {
            result.push_back(i);
        }

        result.push_back("*"); //*
        result.push_back(kappa); //kappa
        result.push_back("+"); //+
        x.derivePrefix(0, x.pieces[0].size()-1, "x0", x.pieces[0], grasp);
        temp = x.derivat;
        x.derivePrefix(0, temp.size()-1, "x0", temp, grasp);
        for (const std::string& i: x.derivat) //T_xx
        {
            result.push_back(i);
        }

        x.derivePrefix(0, x.pieces[0].size()-1, "x1", x.pieces[0], grasp);
        temp = x.derivat;
        x.derivePrefix(0, temp.size()-1, "x1", temp, grasp);
        for (const std::string& i: x.derivat) //T_yy
        {
            result.push_back(i);
        }

    }
    else if (x.expression_type == "postfix")
    {
        //T_t 4 y * sin T_x * + 4 x * cos T_y * + kappa T_{xx} T_{yy} + * -
        x.derivePostfix(0, x.pieces[0].size()-1, "x2", x.pieces[0], grasp);
        for (const std::string& i: x.derivat) //T_t
        {
            result.push_back(i);
        }
        result.push_back("4");
        result.push_back("x1");
        result.push_back("*");
        result.push_back("sin");
        x.derivePostfix(0, x.pieces[0].size()-1, "x0", x.pieces[0], grasp); //derivat will store first derivative of temp wrt x
        for (const std::string& i: x.derivat) //T_x
        {
            result.push_back(i);
        }
        result.push_back("*"); //*
        result.push_back("+"); //+
        result.push_back("4");
        result.push_back("x0");
        result.push_back("*");
        result.push_back("cos");
        x.derivePostfix(0, x.pieces[0].size()-1, "x1", x.pieces[0], grasp);
        for (const std::string& i: x.derivat) //T_y
        {
            result.push_back(i);
        }
        result.push_back("*"); //*
        result.push_back("+"); //+
        result.push_back(kappa); //kappa
        x.derivePostfix(0, x.pieces[0].size()-1, "x0", x.pieces[0], grasp);
        temp = x.derivat;
        x.derivePostfix(0, temp.size()-1, "x0", temp, grasp);
        for (const std::string& i: x.derivat) //T_xx
        {
            result.push_back(i);
        }
        x.derivePostfix(0, x.pieces[0].size()-1, "x1", x.pieces[0], grasp);
        temp = x.derivat;
        x.derivePostfix(0, temp.size()-1, "x1", temp, grasp);
        for (const std::string& i: x.derivat) //T_yy
        {
            result.push_back(i);
        }
        result.push_back("+"); //+
        result.push_back("*"); //*
        result.push_back("-"); //-
    }
    results.push_back(result);
    return results;
}

//x0 -> x, x1 -> y, x2 -> t
std::vector<std::vector<std::string>> sech_squared_trial(Board& x, bool fit)
{
    std::vector<std::vector<std::string>> results;
    std::vector<std::string> result;
    result.reserve(100);
    std::vector<int> grasp;
    grasp.reserve(100);
    std::vector<std::string> temp;
    temp.reserve(50);
    if (x.expression_type == "prefix")
    {
        //- f_hat' * * sech - A * ϵ x sech - A * ϵ x * sech - B * D x sech - B * D x
        result.push_back("-"); // -
        x.derivePrefix(0, x.pieces[0].size()-1, "x0", x.pieces[0], grasp);
        for (const std::string& i: x.derivat) // f_hat'
        {
            result.push_back(i);
        }

        //* * sech - A * ϵ x sech - A * ϵ x * sech - B * D x sech - B * D x
        result.push_back("*"); //*
        result.push_back("*"); //*
        result.push_back("sech"); //sech
        result.push_back("-"); //-
        for (const std::string& i: x.pieces[1]) //A
        {
            result.push_back(i);
        }
        result.push_back("*"); //*
        for (const std::string& i: x.pieces[4]) //ϵ
        {
            result.push_back(i);
        }
        result.push_back("x0"); //x
        result.push_back("sech"); //sech
        result.push_back("-"); //-
        for (const std::string& i: x.pieces[1]) //A
        {
            result.push_back(i);
        }
        result.push_back("*"); //*
        for (const std::string& i: x.pieces[4]) //ϵ
        {
            result.push_back(i);
        }
        result.push_back("x0"); //x
        result.push_back("*"); //*
        result.push_back("sech"); //sech
        result.push_back("-"); //-
        for (const std::string& i: x.pieces[2]) //B
        {
            result.push_back(i);
        }
        result.push_back("*"); //*
        for (const std::string& i: x.pieces[3]) //D
        {
            result.push_back(i);
        }
        result.push_back("x0"); //x
        result.push_back("sech"); //sech
        result.push_back("-"); //-
        for (const std::string& i: x.pieces[2]) //B
        {
            result.push_back(i);
        }
        result.push_back("*"); //*
        for (const std::string& i: x.pieces[3]) //D
        {
            result.push_back(i);
        }
        result.push_back("x0"); //x
    }
    else if (x.expression_type == "postfix")
    {
        //f_hat' A ϵ x * - sech A ϵ x * - sech * B D x * - sech B D x * - sech * * -
        x.derivePostfix(0, x.pieces[0].size()-1, "x0", x.pieces[0], grasp);
        for (const std::string& i: x.derivat) // f_hat'
        {
            result.push_back(i);
        }
        for (const std::string& i: x.pieces[1]) //A
        {
            result.push_back(i);
        }
        for (const std::string& i: x.pieces[4]) //ϵ
        {
            result.push_back(i);
        }
        result.push_back("x0"); //x
        result.push_back("*"); //*
        result.push_back("-"); //-
        result.push_back("sech"); //sech
        for (const std::string& i: x.pieces[1]) //A
        {
            result.push_back(i);
        }
        for (const std::string& i: x.pieces[4]) //ϵ
        {
            result.push_back(i);
        }
        result.push_back("x0"); //x
        result.push_back("*"); //*
        result.push_back("-"); //-
        result.push_back("sech"); //sech
        result.push_back("*"); //*
        for (const std::string& i: x.pieces[2]) //B
        {
            result.push_back(i);
        }
        for (const std::string& i: x.pieces[3]) //D
        {
            result.push_back(i);
        }
        result.push_back("x0"); //x
        result.push_back("*"); //*
        result.push_back("-"); //-
        result.push_back("sech"); //sech
        for (const std::string& i: x.pieces[2]) //B
        {
            result.push_back(i);
        }
        for (const std::string& i: x.pieces[3]) //D
        {
            result.push_back(i);
        }
        result.push_back("x0"); //x
        result.push_back("*"); //*
        result.push_back("-"); //-
        result.push_back("sech"); //sech
        result.push_back("*"); //*
        result.push_back("*"); //*
        result.push_back("-"); //-
    }
    results.push_back(result);
    return results;
}

//https://dl.acm.org/doi/pdf/10.1145/3449639.3459345?casa_token=Np-_TMqxeJEAAAAA:8u-d6UyINV6Ex02kG9LthsQHAXMh2oxx3M4FG8ioP0hGgstIW45X8b709XOuaif5D_DVOm_FwFo
//https://core.ac.uk/download/pdf/6651886.pdf
void SimulatedAnnealing(std::vector<std::vector<std::string>> (*diffeq)(Board&, bool),
                        size_t num_diff_eqns,
                        const Eigen::MatrixXd& data,
                        const std::vector<int>& depth,
                        const std::string& expression_type = "prefix",
                        size_t num_consts_diff = 0,
                        const std::string& method = "LevenbergMarquardt",
                        const int num_fit_iter = 1,
                        const std::string& fit_grad_method = "naive_numerical",
                        const bool cache = true,
                        const double time = 120.0 /*time to run the algorithm in seconds*/,
                        unsigned int num_threads = 0,
                        bool const_tokens = false,
                        double isConstTol = 1e-1,
                        bool use_const_pieces = false,
                        bool simplifyOriginal = false,
                        int numDataCols = 0,
                        bool mustHaveAllFeatures = true,
                        const std::vector<std::vector<std::string>>& custom_features = {},
                        const std::string& bestExpressionFileName = "",
                        const std::vector<int>& maxSize = {},
                        const std::vector<std::vector<std::string>>& additive_corrections = {},
                        bool graphEval = false,
                        int print_every = 1000000,
                        bool printDiffEq = false,
                        const std::vector<std::string>& bad_ops = {},
                        const std::vector<std::vector<std::string>>& seed_expressions = {},
                        bool exit_early = false,
                        int custom_rand_seed = -1,
                        const double T_min = 0.0,
                        const double T_max = 0.0,
                        double (*temp_func)(const double, const double) = [](double ratio, double t) -> double {return pow(ratio, t/(t+1.0));},
                        const char* SNE_file_name = "",
                        bool completeTree = false,
                        const std::string& pert_option = "sub_tree")
{
    assert(simplifyOriginal == false);
    if (num_threads == 0)
    {
        unsigned int temp = std::thread::hardware_concurrency();
        num_threads = ((temp <= 1) ? 1 : temp);
        printf("num_threads = %u\n", num_threads);
    }
    if ( ((RANDOM_SEED >= 0) || (custom_rand_seed >= 0)) && (num_threads > 1))
    {
        std::cout << "Warning, fixing the random seed will make every thread do the same annealing.\nDo you want to continue anyway (y/n)? ";
        char ans;
        std::cin >> ans;
        if ((ans != 'y') && (ans != 'Y'))
        {
            exit(1);
        }
    }
    if (exit_early)
    {
        if (num_threads != 1)
        {
            throw std::runtime_error("Error, make sure `num_threads = 1` when setting `exit_early=true`!");
        }
        else if (num_consts_diff > 0)
        {
            throw std::runtime_error("Error, make sure `num_consts_diff = 0` when setting `exit_early=true`!");
        }
    }


    std::vector<std::thread> threads(num_threads);
    std::latch sync_point(num_threads);

    /*
     Outside of thread:
    */
    std::atomic<double> max_score{0.0};
    std::atomic<double> best_SNE{DBL_MAX};
    std::string best_expression, orig_expression, best_expr_result, orig_expr_result;
    std::vector<double> best_sne_vec;
    std::ostream* out = &std::cout;
    std::ofstream outFile;

    auto start_time = Clock::now();

    /*
     Inside of thread:
     */
    auto func = [&diffeq, &num_diff_eqns, &depth, &expression_type, &num_consts_diff, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &best_expr_result, &orig_expr_result, &const_tokens, &isConstTol, &use_const_pieces, &simplifyOriginal, &numDataCols, &mustHaveAllFeatures, &custom_features, &seed_expressions, &exit_early, &custom_rand_seed, &T_min, &T_max, &temp_func, &completeTree, &pert_option, &best_SNE, &best_sne_vec, &bestExpressionFileName, &maxSize, &additive_corrections, &graphEval, &print_every, &printDiffEq, &bad_ops, &outFile, &out](int thread_idx)
    {
        std::random_device rand_dev;
        #if RANDOM_SEED < 0
            std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
        #else
            std::mt19937 generator;
            generator.seed(RANDOM_SEED);
        #endif
        if (custom_rand_seed >= 0)
        {
            generator.seed(custom_rand_seed);
        }
        Board x(diffeq, num_diff_eqns, true, depth, expression_type, num_consts_diff, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol, use_const_pieces, simplifyOriginal, numDataCols, mustHaveAllFeatures, custom_features, maxSize, additive_corrections, graphEval, completeTree, bad_ops);
        sync_point.arrive_and_wait();
        Board secondary(diffeq, num_diff_eqns, false, std::vector<int>(depth.size(), 0), expression_type, num_consts_diff, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol, use_const_pieces, simplifyOriginal, numDataCols, mustHaveAllFeatures, custom_features, maxSize, additive_corrections, graphEval, completeTree, bad_ops); //For perturbations
        assert(secondary.pieces.size() == secondary.n.size());
        assert(secondary.pieces.size() == x.pieces.size());
        assert(secondary.pieces.size() == x.n.size());
        assert(secondary.num_objectives == x.num_objectives);
        double score = 0.0;

        std::vector<std::vector<std::string>> current(depth.size());
        std::vector<std::pair<int, int>> sub_exprs;
        std::vector<std::string> temp_legal_moves;
        size_t piece_to_perturb_idx; //Used in the case of a depth-0 perturbation
        size_t starting_sub_array_idx; //Used in case pert_sub_array is true
        std::string piece_to_perturb; piece_to_perturb.reserve(10); //Used in the case of a depth-0 perturbation
        std::string piece_to_replace_with; piece_to_replace_with.reserve(10); //Used in the case of a depth-0 perturbation or when pert_sub_array is true
        std::vector<std::uniform_int_distribution<int>> rand_depth_dists(depth.size());
        std::vector<int> rand_depths(depth.size());

        size_t temp_sz;
//        std::string expression, orig_expression, best_expression;
        assert((T_max >= T_min) && (T_max >= 0.0) && (T_min >= 0.0));
        const double ratio = (T_max > 0.0) ? (T_min/T_max) : std::numeric_limits<double>::infinity();
        double T = T_max;
        bool update_current = false;

        auto P = [&](double delta)
        {
            return exp(delta/T);
        };

        auto updateScore = [&](double r = 1.0)
        {
//            assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).first == x.n);
            //update the current expression if (1.) the new one is better, (2.) the new one's worse (with Probability P(Δscore), (3.) if `exit_early` is true
            update_current = ((score > max_score)
                              || ((T > 0.0) ? (x.pos_dist(generator) < P(score-max_score)) : false)
                              || (exit_early));

            if (update_current)
            {
                if (completeTree) //update current expression
                {
//                    std::cout << "x.pieces before complete_tree = " << x.pieces << '\n';
                    x.pieces = x.complete_tree(x.pieces);
//                    std::cout << "x.pieces after complete_tree = " << x.pieces << '\n';
                }
                current = x.pieces; //update current expression
                if ((score > max_score) || exit_early)
                {
                    max_score = score;
                    std::scoped_lock str_lock(Board::thread_locker);
                    best_SNE = x.SNE_curr;
                    best_sne_vec = x.SNE_curr_vec;
                    best_expression = x._to_infix();
                    orig_expression = x.expression();
                    best_expr_result = x._to_infix(x.diffeq_result);
                    orig_expr_result = x.expression(x.diffeq_result);
                    if (bestExpressionFileName.size())
                    {
                        outFile.open(bestExpressionFileName, std::ios::app);
                        if (outFile.is_open())
                        {
                            out = &outFile;
                        }
                        else
                        {
                            out = &std::cout;
                        }
                    }
                    (*out) << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
                    (*out) << "Time spent fitting = " << Board::fit_time << " seconds\n";
                    (*out) << "Best score = " << score << ", SNE = " << best_SNE << '\n';
                    (*out) << "Squared-norm error for each equation: " << best_sne_vec << '\n';
                    (*out) << "Best expression = " << best_expression << '\n';
                    (*out) << "Best expression (original format) = " << orig_expression << '\n';
                    if (printDiffEq)
                    {
                        (*out) << "Best diff result = " << best_expr_result << '\n';
                        (*out) << "Best expression (original format) = " << orig_expr_result << '\n';
                        (*out) << "Best differential equation parameters = " << x.print_diff_params() << '\n';
                        (*out) << "Best expression parameters = " << x.print_expression_params() << '\n';
                        (*out) << "Total system result = " << best_expr_result << '\n';
                        (*out) << "Total system result (original format) = " << orig_expr_result << '\n';
                    }
                    if (outFile.is_open())
                    {
                        outFile.close();
                    }
                }
            }
            else
            {
                x.pieces = current; //reset perturbed state to current state
            }
            T = std::max(T_min, r*T);
//            printf("T = %e\n", T);
        };

        //performs the transformation "const{>=x.num_consts_diff}" -> "const"
        //on each such token in x.pieces
        auto reset_const_token_labels = [&]()
        {
            for (std::vector<std::string>& x_expr: x.pieces)
            {
                for (std::string& token: x_expr)
                {
                    if (token.compare(0, 5, "const") == 0)
                    {
                        std::string int_suffix = token.substr(5);
                        if (int_suffix.size())
                        {
                            int int_suffix_num = std::stoi(int_suffix);
                            if (int_suffix_num >= static_cast<int>(x.num_consts_diff)) //then it's a const that belongs EXCLUSIVELY to pieces -> reset it
                            {
                                token = "const";
                            }
                            else
                            {
                                //Below we test that token is of the form `const{0 <= num < x.num_consts_diff}`
                                assert(((0 <= int_suffix_num) && (int_suffix_num < static_cast<int>(x.num_consts_diff))));
                            }
                        }
                    }
                }
            }
        };

        //`n` is the vector of mutation-tree depths that dictate how each expression in `x.pieces` will be perturbed, and `i` is the time index
        auto Perturbation = [&](const std::vector<int>& n, double i)
        {
            //Step 1: Generate a random depth-n sub-expression `secondary_one.pieces`
            sub_exprs.clear();
            secondary.n = n;

            for (int jdx = 0; jdx < secondary.num_objectives; jdx++)
            {
                //Pre-validation: clears and sanity check
                secondary.pieces[jdx].clear();
                sub_exprs.clear();
                assert((secondary.n[jdx] <= x.n[jdx]) || (pert_option == "sub_array") || (pert_option == "n_random"));
                if (pert_option == "sub_array")
                {
                    /*
                     random starting index of the `n[jdx]` sized sub-array,
                     can be from 0 to pieces[jdx].size() - n[jdx].
                     Example: if n[jdx] = 1, then can start anywhere from 0 to pieces[jdx].size() - 1.
                     Example: if n[jdx] = pieces[jdx].size(), then can only start at 0 (to (pieces[jdx].size() - pieces[jdx].size()) = 0)
                    */
//                    n[jdx] = 3; //must also comment out `const` in `const std::vector<int>& n` above for this hack..
                    std::uniform_int_distribution<int> distribution(0, x.pieces[jdx].size() - n[jdx]);
                    starting_sub_array_idx = distribution(generator);
                    for (size_t ps_idx = starting_sub_array_idx; ps_idx < (starting_sub_array_idx + n[jdx]); ps_idx++)
                    {
                        piece_to_perturb = x.pieces[jdx][ps_idx];
                        if (x.is_unary(piece_to_perturb))
                        {
                            piece_to_replace_with = Board::__unary_operators[Board::unary_dist(generator)];
                        }
                        else if (x.is_binary(piece_to_perturb))
                        {
                            piece_to_replace_with = Board::__binary_operators[Board::binary_dist(generator)];
                        }
                        else
                        {
                            piece_to_replace_with = Board::una_bin_leaf_legal_moves_dict[false][false][true][Board::leaf_dist(generator)];
                        }
                        assert(piece_to_replace_with.size());
                        std::swap(x.pieces[jdx][ps_idx], piece_to_replace_with);
                    }
                }
                else if (pert_option == "n_random")
                {
                    /*
                     choose n[jdx] random indices from 0 to pieces[jdx].size() to swap out
                     */
                    std::uniform_int_distribution<int> distribution(0, x.pieces[jdx].size() - 1);
                    for (int random_idx = 0; random_idx < n[jdx]; random_idx++)
                    {
                        starting_sub_array_idx = distribution(generator);
                        piece_to_perturb = x.pieces[jdx][starting_sub_array_idx];
                        if (x.is_unary(piece_to_perturb))
                        {
                            piece_to_replace_with = Board::__unary_operators[Board::unary_dist(generator)];
                        }
                        else if (x.is_binary(piece_to_perturb))
                        {
                            piece_to_replace_with = Board::__binary_operators[Board::binary_dist(generator)];
                        }
                        else
                        {
                            piece_to_replace_with = Board::una_bin_leaf_legal_moves_dict[false][false][true][Board::leaf_dist(generator)];
                        }
                        assert(piece_to_replace_with.size());
                        std::swap(x.pieces[jdx][starting_sub_array_idx], piece_to_replace_with);
                    }
                }
                //else: `pert_option == "sub_tree"`
                //Step 1a: check for the special case of a depth-0 (i.e. 1 operand) perturbation
                else if (n[jdx] == 0)
                {
                    temp_sz = x.pieces[jdx].size();
                    std::uniform_int_distribution<int> distribution(0, temp_sz - 1); // A random integer generator which generates an index corresponding to an allowed move
                    piece_to_perturb_idx = distribution(generator);
                    piece_to_perturb = x.pieces[jdx][piece_to_perturb_idx];
                    if (x.is_unary(piece_to_perturb))
                    {
                        piece_to_replace_with = Board::__unary_operators[Board::unary_dist(generator)];
                    }
                    else if (x.is_binary(piece_to_perturb))
                    {
                        piece_to_replace_with = Board::__binary_operators[Board::binary_dist(generator)];
                    }
                    else
                    {
                        piece_to_replace_with = Board::una_bin_leaf_legal_moves_dict[false][false][true][Board::leaf_dist(generator)];
                    }
                    assert(piece_to_replace_with.size());
                    std::swap(x.pieces[jdx][piece_to_perturb_idx], piece_to_replace_with);
                }
                else
                {
                    //Step 1b: generate a random expression
                    while (secondary.complete_status(jdx) == -1)
                    {
                        temp_legal_moves = secondary.get_legal_moves(jdx); //the legal moves
                        temp_sz = temp_legal_moves.size(); //the number of legal moves

                        assert(temp_sz);
                        std::uniform_int_distribution<int> distribution(0, temp_sz - 1); // A random integer generator which generates an index corresponding to an allowed move
                        secondary.pieces[jdx].emplace_back(temp_legal_moves[distribution(generator)]); //make the randomly chosen valid move
                        assert(secondary.pieces[jdx].back().size());
                    }
                    assert(secondary.pieces[jdx].size());
                    if (jdx < secondary.num_objectives - 1)
                    {
                        assert(((secondary.expression_type == "prefix") ? secondary.getPNdepth(secondary.pieces[jdx], jdx) : secondary.getRPNdepth(secondary.pieces[jdx], jdx)).first == secondary.n[jdx]);
                        assert(((secondary.expression_type == "prefix") ? secondary.getPNdepth(secondary.pieces[jdx], jdx) : secondary.getRPNdepth(secondary.pieces[jdx], jdx)).second);
                    }
                    //Step 2a: swap in case when perturbation depth == expression depth
                    if (n[jdx] == x.n[jdx])
                    {
                        std::swap(secondary.pieces[jdx], x.pieces[jdx]);
                    }
                    else
                    {
                        //Step 2b: Else, start by identifying the starting and stopping index pairs of all depth `secondary.n[jdx] sub-expressions
                        //in `x.pieces[jdx]` and store them in an std::vector<std::pair<int, int>> called `sub_exprs`.
                        secondary.get_indices(sub_exprs, x.pieces[jdx], jdx);
                        assert ((sub_exprs.size()));
                        assert((secondary.n[jdx] <= x.n[jdx]));
                        //Step 3: Generate a uniform int from 0 to sub_exprs.size() - 1 called `pert_ind`
                        std::uniform_int_distribution<int> distribution(0, sub_exprs.size() - 1);
                        int pert_ind = distribution(generator);

                        //Step 4: Substitute sub_exprs_1[pert_ind] in x.pieces[jdx] with secondary_one.pieces[jdx]
                        auto start = x.pieces[jdx].begin() + sub_exprs[pert_ind].first;
                        auto end = x.pieces[jdx].begin() + std::min(sub_exprs[pert_ind].second, static_cast<int>(x.pieces[jdx].size()));
                        x.pieces[jdx].erase(start, end+1);
                        x.pieces[jdx].insert(start, secondary.pieces[jdx].begin(), secondary.pieces[jdx].end()); //could be a move operation: secondary.pieces doesn't need to be in a defined state after this->params, or erase+insert -> replace?
                        assert(x.pieces[jdx].size() && x.pieces.size());
                        auto depth_and_completion = ((x.expression_type == "prefix") ? x.getPNdepth(x.pieces[jdx], jdx) : x.getRPNdepth(x.pieces[jdx], jdx));
                        assert(depth_and_completion.first == x.n[jdx]);
                    }
                }
                //Step 5: Reset const token labels in pieces
                if (x.use_const_pieces)
                {
                    reset_const_token_labels();
                }
                //Step 6: Evaluate the new mutated `x.pieces` and update score if needed
                score = x.complete_status(x.pieces.size() - 1, false);
//                if (score < 0.0)
//                {
//                    throw(std::runtime_error("score = "+std::to_string(score)));
//                }
                assert(score >= 0.0);
                updateScore(temp_func(ratio, i));
            }
        };

        //Step 1: generate a random expression
        if (seed_expressions.empty())
        {
            for (int jdx = 0; jdx < x.num_objectives; jdx++)
            {
                while ((score = x.complete_status(jdx)) == -1)
                {
                    temp_legal_moves = x.get_legal_moves(jdx); //the legal moves
                    temp_sz = temp_legal_moves.size(); //the number of legal moves
                    assert(temp_sz);
                    std::uniform_int_distribution<int> distribution(0, temp_sz - 1); // A random integer generator which generates an index corresponding to an allowed move
                    x.pieces[jdx].emplace_back(temp_legal_moves[distribution(generator)]); //make the randomly chosen valid move
                    current[jdx].push_back(x.pieces[jdx].back());
                }
                auto depth_and_completion = ((x.expression_type == "prefix") ? x.getPNdepth(x.pieces[jdx], jdx) : x.getRPNdepth(x.pieces[jdx], jdx));
                assert(depth_and_completion.first == x.n[jdx]);
                assert(depth_and_completion.second);
                //x.n[jdx] = depth_and_completion.first;
                bool pert_elem = ((pert_option == "sub_array") || (pert_option == "n_random"));
                rand_depth_dists[jdx] = std::uniform_int_distribution<int>((pert_elem ? 1 : 0), ((pert_elem) ? x.pieces[jdx].size() : x.n[jdx]));
                rand_depths[jdx] = rand_depth_dists[jdx](generator);
            }
        }
        else
        {
            assert(x.pieces.size() == decltype(x.pieces.size())(x.num_objectives));
            for (int jdx = 0; jdx < x.num_objectives; jdx++)
            {
                x.pieces[jdx] = seed_expressions[jdx];
                current[jdx] = seed_expressions[jdx];
                assert(all_check(x.pieces[jdx]));
                auto depth_and_completion = ((x.expression_type == "prefix") ? x.getPNdepth(x.pieces[jdx], jdx) : x.getRPNdepth(x.pieces[jdx], jdx));
                assert((depth_and_completion.first <= x.n[jdx]) && ("Seed expression depth of x.pieces[" + std::to_string(jdx) + "] = " + std::to_string(depth_and_completion.first)).c_str());
                assert(depth_and_completion.second);
                //x.n[jdx] = depth_and_completion.first;
                bool pert_elem = ((pert_option == "sub_array") || (pert_option == "n_random"));
                rand_depth_dists[jdx] = std::uniform_int_distribution<int>((pert_elem ? 1 : 0), ((pert_elem) ? x.pieces[jdx].size() : x.n[jdx]));
                assert(x.pieces[jdx].size());
            }
            assert(all_checks(x.pieces));
            assert((x.pieces.size() == static_cast<decltype(x.pieces.size())>(x.num_objectives)) && (x.pieces.size()));
            score = x.complete_status(x.pieces.size() - 1, false);
            //std::cout << "score = " << score << '\n';
        }
        reset_const_token_labels();
        updateScore(1.0);
        assert(all_checks(x.pieces));
        if (exit_early)
        {
            exit(1);
        }
        for (double i = 0; (timeElapsedSince(start_time) < time); i++)
        {
            if (i && (static_cast<int>(i)%print_every == 0))
            {
                std::scoped_lock progress_lock(Board::thread_locker);
                
                if (use_const_pieces)
                {
                    std::cout << "Thread " << thread_idx << " Iteration " << i << '\n';
                    std::cout << "Thread " << thread_idx << " Unique expressions = " << Board::expression_dict.size() << '\n';
                }
                else
                {
                    std::cout << "Thread " << thread_idx << " Iteration " << i << '\n';
                }
            }
            for (int jdx = 0; jdx < x.num_objectives; jdx++)
            {
                /*
                 If pert_sub_array is true, then `rand_depth_dists[jdx](generator)` returns
                 a random-integer from [1, x.pieces[jdx].size()] (the size of the sub-array to mutate).
                 Else, `rand_depth_dists[jdx](generator)` returns
                 a random-integer from [0, x.n[jdx]] (the depth of sub-tree to swap).
                 */
                rand_depths[jdx] = rand_depth_dists[jdx](generator);
            }
            assert(all_checks(x.pieces));
            Perturbation(rand_depths, i);
        }
        puts("Done with symbolic regression");
    };
    //Starting the threads each with a separate version of `func`
    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads[i] = std::thread(func, i);
    }

    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads[i].join();
    }

    std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
    std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
    std::cout << "Best score = " << max_score << ", SNE = " << best_SNE << '\n';
    if (strlen(SNE_file_name))
    {
        // Open file in append mode
        std::ofstream out(SNE_file_name, std::ios::app);
        out << best_sne_vec << '\n';
        out.close();
    }
    std::cout << "Squared-norm error for each equation: " << best_sne_vec << '\n';
    std::cout << "Best expression = " << best_expression << '\n';
    std::cout << "Best expression (original format) = " << orig_expression << '\n';
    if (printDiffEq)
    {
        std::cout << "Best diff result = " << best_expr_result << '\n';
        std::cout << "Best expression (original format) = " << orig_expr_result << '\n';
    }
}
////
//////https://arxiv.org/abs/2310.06609
//void GP(std::vector<std::string> (*diffeq)(Board&), size_t num_diff_eqns, const Eigen::MatrixXd& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true, double time = 120, unsigned int num_threads = 0, bool const_tokens = false, double isConstTol = 1e-1)
//{
//    if (num_threads == 0)
//    {
//        unsigned int temp = std::thread::hardware_concurrency();
//        num_threads = ((temp <= 1) ? 1 : temp);
//    }
//
//    std::vector<std::thread> threads(num_threads);
//    std::latch sync_point(num_threads);
//
//    /*
//     Outside of thread:
//     */
//    std::atomic<double> max_score{0.0};
//    std::atomic<double> best_SNE{DBL_MAX};
//    std::string best_expression, orig_expression, best_expr_result, orig_expr_result;
//
//    auto start_time = Clock::now();
//
//    /*
//     Inside of thread:
//     */
//
//    auto func = [&diffeq, &depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &best_expr_result, &orig_expr_result, &const_tokens, &isConstTol, &best_SNE]()
//    {
//        std::random_device rand_dev;
//        std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
//        Board x(diffeq, true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol);
//        sync_point.arrive_and_wait();
//        Board secondary_one(diffeq, false, (depth > 0) ? depth-1 : 0, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol), secondary_two(diffeq, false, (depth > 0) ? depth-1 : 0, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol); //For crossover and mutations
//        double score = 0.0, mut_prob = 0.8, rand_mut_cross;
//        constexpr int init_population = 2000;
//        std::vector<std::pair<std::vector<std::string>, double>> individuals;
//        std::pair<std::vector<std::string>, double> individual_1, individual_2;
//        std::vector<std::pair<int, int>> sub_exprs_1, sub_exprs_2;
//        individuals.reserve(2*init_population);
//        std::vector<std::string> temp_legal_moves;
//        std::uniform_int_distribution<int> rand_depth_dist(0, x.n - 1), selector_dist(0, init_population - 1);
//        int rand_depth, rand_individual_idx_1, rand_individual_idx_2;
//        std::uniform_real_distribution<double> rand_mut_cross_dist(0.0, 1.0);
//        size_t temp_sz;
//    //    std::string expression, orig_expression, best_expression;
//
//        auto updateScore = [&]()
//        {
//            assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).first == x.n);
//            assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).second);
//            if (score > max_score)
//            {
//                max_score = score;
//                std::scoped_lock str_lock(Board::thread_locker);
//                best_SNE = x.SNE_curr;
//                best_expression = x._to_infix();
//                orig_expression = x.expression();
//                best_expr_result = x._to_infix(x.diffeq_result);
//                orig_expr_result = x.expression(x.diffeq_result);
//                std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
//                std::cout << "Best score = " << score << ", SNE = " << best_SNE << '\n';
//                std::cout << "Best expression = " << best_expression << '\n';
//                std::cout << "Best expression (original format) = " << orig_expression << '\n';
//                std::cout << "Best diff result = " << best_expr_result << '\n';
//                std::cout << "Best expression (original format) = " << orig_expr_result << '\n';
//            }
//        };
//
//        //Step 1, generate init_population expressions
//        for (int i = 0; i < init_population; i++)
//        {
//            while ((score = x.complete_status()) == -1)
//            {
//                temp_legal_moves = x.get_legal_moves(); //the legal moves
//                temp_sz = temp_legal_moves.size(); //the number of legal moves
//                assert(temp_sz);
//                std::uniform_int_distribution<int> distribution(0, temp_sz - 1); // A random integer generator which generates an index corresponding to an allowed move
//                x.pieces.push_back(temp_legal_moves[distribution(generator)]); //make the randomly chosen valid move
//            }
//            updateScore();
//            individuals.push_back(std::make_pair(x.pieces, score));
//            x.pieces.clear();
//        }
//
//        auto Mutation = [&](int n)
//        {
//            //Step 1: Generate a random depth-n sub-expression `secondary_one.pieces`
//            secondary_one.pieces.clear();
//            sub_exprs_1.clear();
//            secondary_one.n = n;
//            while (secondary_one.complete_status() == -1)
//            {
//                temp_legal_moves = secondary_one.get_legal_moves();
//                std::uniform_int_distribution<int> distribution(0, temp_legal_moves.size() - 1);
//                secondary_one.pieces.push_back(temp_legal_moves[distribution(generator)]);
//            }
//
//            assert(((secondary_one.expression_type == "prefix") ? secondary_one.getPNdepth(secondary_one.pieces) : secondary_one.getRPNdepth(secondary_one.pieces)).first == secondary_one.n);
//            assert(((secondary_one.expression_type == "prefix") ? secondary_one.getPNdepth(secondary_one.pieces) : secondary_one.getRPNdepth(secondary_one.pieces)).second);
//
//            //Step 2: Identify the starting and stopping index pairs of all depth-n sub-expressions
//            //in `x.pieces` and store them in an std::vector<std::pair<int, int>>
//            //called `sub_exprs_1`.
//            x.pieces = individuals[selector_dist(generator)].first; //A randomly selected individual to be mutated
//            secondary_one.get_indices(sub_exprs_1, x.pieces);
//
//            //Step 3: Generate a uniform int from 0 to sub_exprs.size() - 1 called `mut_ind`
//            std::uniform_int_distribution<int> distribution(0, sub_exprs_1.size() - 1);
//            int mut_ind = distribution(generator);
//
//            //Step 4: Substitute sub_exprs_1[mut_ind] in x.pieces with secondary_one.pieces
//
//            auto start = x.pieces.begin() + sub_exprs_1[mut_ind].first;
//            auto end = std::min(x.pieces.begin() + sub_exprs_1[mut_ind].second, x.pieces.end()-1);
//            x.pieces.erase(start, end+1);
//            x.pieces.insert(start, secondary_one.pieces.begin(), secondary_one.pieces.end());
//
//            //Step 5: Reset const token labels in pieces
//            size_t const_counter = 0;
//            for (std::string& token: x.pieces)
//            {
//                if (token.compare(0, 5, "const") == 0)
//                {
//                    token = "const" + std::to_string(const_counter++);
//                }
//            }
//
//            //Step 6: Evaluate the new mutated `x.pieces` and update score if needed
//            score = x.complete_status(false);
//            updateScore();
//            individuals.push_back(std::make_pair(x.pieces, score));
//        };
//
//        auto Crossover = [&](int n)
//        {
//            sub_exprs_1.clear();
//            sub_exprs_2.clear();
//            secondary_one.n = n;
//            secondary_two.n = n;
//
//            rand_individual_idx_1 = selector_dist(generator);
//            assert(individuals.size() && rand_individual_idx_1 < individuals.size());
//            individual_1 = individuals[rand_individual_idx_1];
//
//            do {
//                rand_individual_idx_2 = selector_dist(generator);
//            } while (rand_individual_idx_2 == rand_individual_idx_1);
//            assert(individuals.size() && rand_individual_idx_1 < individuals.size());
//            individual_2 = individuals[rand_individual_idx_2];
//
//            //Step 1: Identify the starting and stopping index pairs of all depth-n sub-expressions
//            //in `individual_1.first` and store them in an std::vector<std::pair<int, int>> called `sub_exprs_1`.
//            secondary_one.get_indices(sub_exprs_1, individual_1.first);
//
//            //Step 2: Identify the starting and stopping index pairs of all depth-n sub-expressions
//            //in `individual_2.first` and store them in an std::vector<std::pair<int, int>> called `sub_exprs_2`.
//            secondary_two.get_indices(sub_exprs_2, individual_2.first);
//
//            //Step 3: Generate a random uniform int from 0 to sub_exprs_1.size() - 1 called `mut_ind_1`
//            std::uniform_int_distribution<int> distribution_1(0, sub_exprs_1.size() - 1);
//            int mut_ind_1 = distribution_1(generator);
//
//            //Step 4: Generate a random uniform int from 0 to sub_exprs_2.size() - 1 called `mut_ind_2`
//            std::uniform_int_distribution<int> distribution_2(0, sub_exprs_2.size() - 1);
//            int mut_ind_2 = distribution_2(generator);
//
//            //Step 5: Swap sub_exprs_1[mut_ind_1] in individual_1.first with sub_exprs_2[mut_ind_2] in individual_2.first
//            auto start_1 = individual_1.first.begin() + sub_exprs_1[mut_ind_1].first;
//            auto end_1 = std::min(individual_1.first.begin() + sub_exprs_1[mut_ind_1].second, individual_1.first.end());
//
//            auto start_2 = individual_2.first.begin() + sub_exprs_2[mut_ind_2].first;
//            auto end_2 = std::min(individual_2.first.begin() + sub_exprs_2[mut_ind_2].second, individual_2.first.end());
//
//    //        insert the range start_2, end_2+1 into individual_1 and the range start_1, end_1+1 into individual_2.
//
//            if ((end_1 - start_1) < (end_2 - start_2))
//            {
//                std::swap_ranges(start_1, end_1+1, start_2);
//                //Insert remaining part of sub_individual_2.first into individual_1.first
//                individual_1.first.insert(end_1+1, start_2 + (end_1+1-start_1), end_2+1);
//                //Remove the remaining part of sub_individual_2.first from individual_2.first
//                individual_2.first.erase(start_2 + (end_1+1-start_1), end_2+1);
//            }
//            else if ((end_2 - start_2) < (end_1 - start_1))
//            {
//                std::swap_ranges(start_2, end_2+1, start_1);
//                //Insert remaining part of sub_individual_1.first into individual_2.first
//                individual_2.first.insert(end_2+1, start_1 + (end_2+1-start_2), end_1+1);
//                //Remove the remaining part of sub_individual_1.first from individual_1.first
//                individual_1.first.erase(start_1 + (end_2+1-start_2), end_1+1);
//            }
//            else
//            {
//                std::swap_ranges(start_1, end_1+1, start_2);
//            }
//
//            //Step 6: Reset const token labels in individual_1.first
//            size_t const_counter = 0;
//            for (std::string& token: individual_1.first)
//            {
//                if (token.compare(0, 5, "const") == 0)
//                {
//                    token = "const" + std::to_string(const_counter++);
//                }
//            }
//
//            //Step 7: Evaluate the new `x.pieces` and update score if needed
//            x.pieces = individual_1.first;
//            score = x.complete_status(false);
//            updateScore();
//
//            individuals.push_back(std::make_pair(x.pieces, score));
//
//            //Step 8: Reset const token labels in individual_2.first
//            const_counter = 0;
//            for (std::string& token: individual_2.first)
//            {
//                if (token.compare(0, 5, "const") == 0)
//                {
//                    token = "const" + std::to_string(const_counter++);
//                }
//            }
//
//            //Step 9: Evaluate the new `x.pieces` and update score if needed
//            x.pieces = individual_2.first;
//            score = x.complete_status(false);
//            updateScore();
//
//            individuals.push_back(std::make_pair(x.pieces, score));
//        };
//
//
//        for (/*int ngen = 0*/; (timeElapsedSince(start_time) < time); /*ngen++*/)
//        {
////            if (ngen && (ngen%5 == 0))
////            {
////                std::cout << "Unique expressions = " << Board::expression_dict.size() << '\n';
////            }
//            //Produce N additional individuals through crossover and mutation
//            for (int n = 0; n < init_population; n++)
//            {
//                //Step 1: Generate a random number between 0 and 1 called `rand_mut_cross`
//                rand_mut_cross = rand_mut_cross_dist(generator);
//
//                //Step 2: Generate a random uniform int from 0 to x.n - 1 called `rand_depth`
//                rand_depth = rand_depth_dist(generator);
//
//                //Step 4: Call Mutation function if 0 <= rand_mut_cross <= mut_prob, else select Crossover
//                if (rand_mut_cross <= mut_prob)
//                {
//                    Mutation(rand_depth);
//                }
//                else
//                {
//                    Crossover(rand_depth);
//                }
//            }
//            std::sort(individuals.begin(), individuals.end(),
//            [](std::pair<std::vector<std::string>, double>& individual_1, std::pair<std::vector<std::string>, double>& individual_2)
//            {
//                return individual_1.second > individual_2.second;
//            });
//            individuals.resize(init_population);
//        }
//    };
//
//    for (unsigned int i = 0; i < num_threads; i++)
//    {
//        threads[i] = std::thread(func); //TODO: (maybe) provide a depth argument to func to specify if different threads should focus on different depth expressions (and modify the search functions accordingly)?
//    }
//
//    for (unsigned int i = 0; i < num_threads; i++)
//    {
//        threads[i].join();
//    }
//
//    std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
//    std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
//    std::cout << "Best score = " << max_score << ", SNE = " << best_SNE << '\n';
//    std::cout << "Best expression = " << best_expression << '\n';
//    std::cout << "Best expression (original format) = " << orig_expression << '\n';
//    std::cout << "Best diff result = " << best_expr_result << '\n';
//    std::cout << "Best expression (original format) = " << orig_expr_result << '\n';
//}
//
//void PSO(std::vector<std::string> (*diffeq)(Board&), size_t num_diff_eqns, const Eigen::MatrixXd& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true, double time = 120, unsigned int num_threads = 0, bool const_tokens = false, double isConstTol = 1e-1)
//{
//    if (num_threads == 0)
//    {
//        unsigned int temp = std::thread::hardware_concurrency();
//        num_threads = ((temp <= 1) ? 1 : temp);
//    }
//
//    std::vector<std::thread> threads(num_threads);
//    std::latch sync_point(num_threads);
//
//    /*
//     Outside of thread:
//     */
//
//    std::atomic<double> max_score{0.0};
//    std::atomic<double> best_SNE{DBL_MAX};
//    std::string best_expression, orig_expression, best_expr_result, orig_expr_result;
//
//    auto start_time = Clock::now();
//
//    /*
//     Inside of thread:
//     */
//
//    auto func = [&diffeq, &depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &best_expr_result, &orig_expr_result, &const_tokens, &isConstTol, &best_SNE]()
//    {
//        std::random_device rand_dev;
//        std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
//        Board x(diffeq, true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol);
//
//        sync_point.arrive_and_wait();
//        double score = 0, check_point_score = 0;
//        std::vector<std::string> temp_legal_moves;
//
//        size_t temp_sz;
//    //    std::string expression, orig_expression, best_expression;
//
//        /*
//         For this setup, we don't know a-priori the number of particles, so we generate them and their corresponding velocities as needed
//         */
//        std::vector<double> particle_positions, best_positions, v, curr_positions;
//        particle_positions.reserve(x.reserve_amount); //stores record of all current particle position indices
//        best_positions.reserve(x.reserve_amount); //indices corresponding to best pieces
//        curr_positions.reserve(x.reserve_amount); //indices corresponding to x.pieces
//        v.reserve(x.reserve_amount); //stores record of all current particle velocities
//        double rp, rg, new_v, c = 0.0;
//        int c_count = 0;
//        std::unordered_map<double, std::unordered_map<int, int>> Nsa;
//        std::unordered_map<double, std::unordered_map<int, double>> Psa;
//        std::unordered_map<int, double> p_i_vals, p_i;
//
//        /*
//         In this implementation of PSO:
//
//             The traditional PSO initializes the particle positions to be between 0 and 1. However, in this application,
//             the particle positions are discrete values and any of the legal integer tokens (moves). The
//             velocities are continuous-valued and perturb the postions, which are subsequently constrained by rounding to
//             the nearest whole number then taking the modulo w.r.t. the # of allowed legal moves.
//
//         */
//
//        for (int iter = 0; (timeElapsedSince(start_time) < time); iter++)
//        {
//            if (iter && (iter%50000 == 0))
//            {
//    //            std::cout << "Unique expressions = " << Board::expression_dict.size() << '\n';
//    //            std::cout << "check_point_score = " << check_point_score
//    //            << ", max_score = " << max_score << ", c = " << c << '\n';
//                if (check_point_score == max_score)
//                {
//                    c_count++;
//                    std::uniform_real_distribution<double> temp(-c_count, c_count);
//    //                std::cout << "c: " << c << " -> ";
//                    c = temp(generator);
//    //                std::cout << c << '\n';
//                }
//                else
//                {
//    //                std::cout << "c: " << c << " -> ";
//                    c = 0.0; //if new best found, reset c and try to exploit the new best
//                    c_count = 0;
//    //                std::cout << c << '\n';
//                }
//                check_point_score = max_score;
//            }
//
//            for (int i = 0; (score = x.complete_status()) == -1; i++) //i is the index of the token
//            {
//                rp = x.pos_dist(generator), rg = x.pos_dist(generator);
//                temp_legal_moves = x.get_legal_moves(); //the legal moves
//                temp_sz = temp_legal_moves.size(); //the number of legal moves
//
//                if (i == static_cast<int>(particle_positions.size())) //Then we need to create a new particle with some initial position and velocity
//                {
//                    particle_positions.push_back(x.pos_dist(generator));
//                    v.push_back(x.vel_dist(generator));
//                }
//
//                particle_positions[i] = trueMod(std::round(particle_positions[i]), temp_sz);
//                x.pieces.push_back(temp_legal_moves[particle_positions[i]]); //x.pieces holds the pieces corresponding to the indices
//                curr_positions.push_back(particle_positions[i]);
//                if (i == static_cast<int>(best_positions.size()))
//                {
//                    best_positions.push_back(x.pos_dist(generator));
//                    best_positions[i] = trueMod(std::round(best_positions[i]), temp_sz);
//                }
//                //https://hal.science/hal-00764996
//                //https://www.researchgate.net/publication/216300408_An_off-the-shelf_PSO
//                new_v = (0.721*v[i] + x.phi_1*rg*(best_positions[i] - particle_positions[i]) + x.phi_2*rp*(p_i[i] - particle_positions[i]) + c);
//                v[i] = copysign(std::min(new_v, DBL_MAX), new_v);
//                particle_positions[i] += v[i];
//                Nsa[curr_positions[i]][i]++;
//            }
//
//            for (int i = 0; i < static_cast<int>(curr_positions.size()); i++)
//            {
//                Psa[curr_positions[i]][i] = (Psa[curr_positions[i]][i]+score)/Nsa[curr_positions[i]][i];
//                if (Psa[curr_positions[i]][i] > p_i_vals[i])
//                {
//                    p_i[i] = curr_positions[i];
//                }
//                p_i_vals[i] = std::max(p_i_vals[i], Psa[curr_positions[i]][i]);
//
//            }
//
//            if (score > max_score)
//            {
//                for (int idx = 0; idx < static_cast<int>(curr_positions.size()); idx++)
//                {
//                    best_positions[idx] = curr_positions[idx];
//                }
//                max_score = score;
//                std::scoped_lock str_lock(Board::thread_locker);
//                best_SNE = x.SNE_curr;
//                best_expression = x._to_infix();
//                orig_expression = x.expression();
//                best_expr_result = x._to_infix(x.diffeq_result);
//                orig_expr_result = x.expression(x.diffeq_result);
//                std::cout << "Best score = " << score << ", SNE = " << best_SNE << '\n';
//                std::cout << "Best expression = " << best_expression << '\n';
//                std::cout << "Best expression (original format) = " << orig_expression << '\n';
//                std::cout << "Best diff result = " << best_expr_result << '\n';
//                std::cout << "Best expression (original format) = " << orig_expr_result << '\n';
//            }
//            x.pieces.clear();
//            curr_positions.clear();
//        }
//    };
//
//    for (unsigned int i = 0; i < num_threads; i++)
//    {
//        threads[i] = std::thread(func); //TODO: (maybe) provide a depth argument to func to specify if different threads should focus on different depth expressions (and modify the search functions accordingly)?
//    }
//
//    for (unsigned int i = 0; i < num_threads; i++)
//    {
//        threads[i].join();
//    }
//
//    std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
//    std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
//    std::cout << "Best score = " << max_score << ", SNE = " << best_SNE << '\n';
//    std::cout << "Best expression = " << best_expression << '\n';
//    std::cout << "Best expression (original format) = " << orig_expression << '\n';
//    std::cout << "Best diff result = " << best_expr_result << '\n';
//    std::cout << "Best expression (original format) = " << orig_expr_result << '\n';
//}
//
////https://arxiv.org/abs/2205.13134
//void ConcurrentMCTS(std::vector<std::string> (*diffeq)(Board&), size_t num_diff_eqns, const Eigen::MatrixXd& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true, double time = 120, unsigned int num_threads = 0, bool const_tokens = false, double isConstTol = 1e-1)
//{
//    if (num_threads == 0)
//    {
//        unsigned int temp = std::thread::hardware_concurrency();
//        num_threads = ((temp <= 1) ? 1 : temp);
//    }
//
//    std::vector<std::thread> threads(num_threads);
//    std::latch sync_point(num_threads);
//
//    /*
//     Outside of thread:
//     */
//    std::atomic<double> max_score{0.0};
//    std::atomic<double> best_SNE{DBL_MAX};
//
//    std::string best_expression, orig_expression, best_expr_result, orig_expr_result;
//
//    auto start_time = Clock::now();
//
//    /*
//     Inside of thread:
//     */
//
//    boost::concurrent_flat_map<std::string, boost::concurrent_flat_map<std::string, double>> Qsa;
//    boost::concurrent_flat_map<std::string, boost::concurrent_flat_map<std::string, int>> Nsa;
//    boost::concurrent_flat_map<std::string, int> Ns;
//
//    auto func = [&diffeq, &depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &best_expr_result, &orig_expr_result, &const_tokens, &isConstTol, &best_SNE, &Qsa, &Nsa, &Ns]()
//    {
//        std::random_device rand_dev;
//        std::mt19937 thread_local generator(rand_dev());
//        Board x(diffeq, true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol);
//
//        sync_point.arrive_and_wait();
//        double score = 0.0, check_point_score = 0.0, UCT, UCT_best;
//        std::string best_act;
//
//        std::vector<std::string> temp_legal_moves;
//        std::string state;
//
//        double c = 1.4; //"controls the balance between exploration and exploitation", see equation 2 here: https://web.engr.oregonstate.edu/~afern/classes/cs533/notes/uct.pdf, top of page 8 here: https://arxiv.org/pdf/1402.6028.pdf, first formula in section 4. Experiments here: https://cesa-bianchi.di.unimi.it/Pubblicazioni/ml-02.pdf
//        std::vector<std::pair<std::string, std::string>> moveTracker;
//        moveTracker.reserve(x.reserve_amount);
//        temp_legal_moves.reserve(x.reserve_amount);
//        state.reserve(2*x.reserve_amount);
//        //        double str_convert_time = 0.0;
//        auto getString  = [&]()
//        {
//            if (!x.pieces.empty())
//            {
//                state += x.pieces[x.pieces.size()-1] + " ";
//            }
//        };
//
//        for (int i = 0; (timeElapsedSince(start_time) < time); i++)
//        {
//            if (i && (i%1000 == 0))
//            {
//                //                    std::cout << "Unique expressions = " << Board::expression_dict.size() << '\n';
//                //                    std::cout << "check_point_score = " << check_point_score
//                //                    << ", max_score = " << max_score << ", c = " << c << '\n';
//                if (check_point_score == max_score)
//                {
//                    //                        std::cout << "c: " << c << " -> ";
//                    c += 1.4;
//                    //                        std::cout << c << '\n';
//                }
//                else
//                {
//                    //                        std::cout << "c: " << c << " -> ";
//                    c = 1.4; //if new best found, reset c and try to exploit the new best
//                    //                        std::cout << c << '\n';
//                    check_point_score = max_score;
//                }
//            }
//            state.clear();
//            while ((score = x.complete_status()) == -1)
//            {
//                temp_legal_moves = x.get_legal_moves();
//                assert(temp_legal_moves.size());
//
////                for (double i: temp_legal_moves)
////                {
////                    assert(i >= 0.0);
////                }
////                    auto start_time = Clock::now();
//                getString();
////                    str_convert_time += timeElapsedSince(start_time);
//                UCT = 0.0;
//                UCT_best = -DBL_MAX;
//                best_act = temp_legal_moves[0];
//                std::vector<std::string> best_acts;
//                best_acts.reserve(temp_legal_moves.size());
//
//                for (const std::string& a : temp_legal_moves)
//                {
////                    assert(a > -1.0);
////                    boost::concurrent_flat_map<std::string, boost::concurrent_flat_map<double, double>>
//                    if (Nsa.contains(state))
//                    {
//                        int Nsa_contains_a = 0;
//                        Nsa.cvisit(state, [&](const auto& x)
//                        {
//                            if (x.second.contains(a))
//                            {
//                               x.second.cvisit(a, [&](const auto& y)
//                               {
//                                   Nsa_contains_a = y.second;
//                               });
//                            }
//                        });
//                        if (Nsa_contains_a)
//                        {
//                            double Qsa_s_a;
//                            int Ns_s, Nsa_s_a;
//                            Qsa.cvisit(state, [&](const auto& x)
//                            {
//                                x.second.cvisit(a, [&](const auto& y)
//                                {
//                                    Qsa_s_a = y.second;
//                                });
//                            });
//                            Nsa.cvisit(state, [&](const auto& x)
//                            {
//                                x.second.cvisit(a, [&](const auto& y)
//                                {
//                                    Nsa_s_a = y.second;
//                                });
//                            });
//                            Ns.cvisit(state, [&](const auto& x)
//                            {
//                                Ns_s = x.second;
//                            });
//                            UCT = Qsa_s_a + c*sqrt(log(Ns_s)/Nsa_s_a);
//                        }
//                        else
//                        {
//                            Nsa.visit(state, [&](auto& x)
//                            {
//                                x.second.insert_or_assign(a, 0);
//                            });
//                            Qsa.visit(state, [&](auto& x)
//                            {
//                               x.second.insert_or_assign(a, 0.0);
//                            });
//                            Ns.insert_or_assign(state, 0);
//                            best_acts.push_back(a);
//                            UCT = -DBL_MAX;
//                        }
//                    }
//                    else
//                    {
//                        Nsa.insert_or_assign(state, boost::concurrent_flat_map<std::string, int>({{a, 0}}));
//                        Qsa.insert_or_assign(state, boost::concurrent_flat_map<std::string, double>({{a, 0.0}}));
//                        Ns.insert_or_assign(state, 0);
//                        best_acts.push_back(a);
//                        UCT = -DBL_MAX;
//                    }
//
//                    if (UCT > UCT_best)
//                    {
//                        best_act = a;
//                        UCT_best = UCT;
//                    }
//                }
////                assert(best_acts.size() || (best_act > -1.0));
//                if (best_acts.size())
//                {
//                    std::uniform_int_distribution<int> distribution(0, best_acts.size() - 1);
//                    best_act = best_acts[distribution(generator)];
//                }
//
//                x.pieces.push_back(best_act);
//                moveTracker.push_back(make_pair(state, best_act));
////                assert(Ns.contains(state));
//                Ns.visit(state, [&](auto& x)
//                {
//                    x.second++;
//                });
////                assert(Nsa.contains(state));
//                Nsa.visit(state, [&](auto& x)
//                {
//                    if (!x.second.contains(best_act))
//                    {
//                        x.second.insert_or_assign(best_act, 0);
//                    }
////                    assert( x.second.contains(best_act));
//                    x.second.visit(best_act, [&](auto& y)
//                    {
//                       y.second++;
//                    });
//                });
//            }
//            //backprop reward `score`
//            for (auto& state_action: moveTracker)
//            {
////                assert(Qsa.contains(state_action.first));
//                Qsa.visit(state_action.first, [&](auto& x)
//                {
////                    assert(x.second.contains(state_action.second));
//                    if (!x.second.contains(state_action.second))
//                    {
//                        Nsa.visit(state, [&](auto& y)
//                        {
//                            y.second.insert_or_assign(state_action.second, 0);
//                        });
//                        x.second.insert_or_assign(state_action.second, 0.0);
//                    }
//
//                    x.second.visit(state_action.second, [&](auto& y)
//                    {
//                        y.second = std::max(y.second, score);
//                    });
//                });
//            }
//
//            if (score > max_score)
//            {
//                max_score = score;
//                std::scoped_lock str_lock(Board::thread_locker);
//                best_SNE = x.SNE_curr;
//                best_expression = x._to_infix();
//                orig_expression = x.expression();
//                best_expr_result = x._to_infix(x.diffeq_result);
//                orig_expr_result = x.expression(x.diffeq_result);
//            }
//            x.pieces.clear();
//            moveTracker.clear();
//        }
//    };
//
//    for (unsigned int i = 0; i < num_threads; i++)
//    {
//        threads[i] = std::thread(func); //TODO: (maybe) provide a depth argument to func to specify if different threads should focus on different depth expressions (and modify the search functions accordingly)?
//    }
//
//    for (unsigned int i = 0; i < num_threads; i++)
//    {
//        threads[i].join();
//    }
//
//    std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
//    std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
//    std::cout << "Best score = " << max_score << ", SNE = " << best_SNE << '\n';
//    std::cout << "Best expression = " << best_expression << '\n';
//    std::cout << "Best expression (original format) = " << orig_expression << '\n';
//    std::cout << "Best diff result = " << best_expr_result << '\n';
//    std::cout << "Best expression (original format) = " << orig_expr_result << '\n';
//}
//
////https://arxiv.org/abs/2205.13134
//void MCTS(std::vector<std::string> (*diffeq)(Board&), size_t num_diff_eqns, const Eigen::MatrixXd& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true, double time = 120, unsigned int num_threads = 0, bool const_tokens = false, double isConstTol = 1e-1)
//{
//    if (num_threads == 0)
//    {
//        unsigned int temp = std::thread::hardware_concurrency();
//        num_threads = ((temp <= 1) ? 1 : temp);
//    }
//
//    std::vector<std::thread> threads(num_threads);
//    std::latch sync_point(num_threads);
//
//    /*
//     Outside of thread:
//     */
//    std::atomic<double> max_score{0.0};
//    std::atomic<double> best_SNE{DBL_MAX};
//    std::string best_expression, orig_expression, best_expr_result, orig_expr_result;
//
//    auto start_time = Clock::now();
//
//    /*
//     Inside of thread:
//     */
//
//    auto func = [&diffeq, &depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &best_expr_result, &orig_expr_result, &const_tokens, &isConstTol, &best_SNE]()
//    {
//        std::random_device rand_dev;
//        std::mt19937 thread_local generator(rand_dev());
//        Board x(diffeq, true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol);
//
//        sync_point.arrive_and_wait();
//        double score = 0.0, check_point_score = 0.0, UCT, UCT_best;
//        std::string best_act;
//
//        std::vector<std::string> temp_legal_moves;
//        std::unordered_map<std::string, std::unordered_map<std::string, double>> Qsa, Nsa;
//        std::unordered_map<std::string, double> Ns;
//        std::string state;
//
//        double c = 1.4; //"controls the balance between exploration and exploitation", see equation 2 here: https://web.engr.oregonstate.edu/~afern/classes/cs533/notes/uct.pdf, top of page 8 here: https://arxiv.org/pdf/1402.6028.pdf, first formula in section 4. Experiments here: https://cesa-bianchi.di.unimi.it/Pubblicazioni/ml-02.pdf
//        std::vector<std::pair<std::string, std::string>> moveTracker;
//        moveTracker.reserve(x.reserve_amount);
//        temp_legal_moves.reserve(x.reserve_amount);
//        state.reserve(2*x.reserve_amount);
//        //        double str_convert_time = 0.0;
//        auto getString  = [&]()
//        {
//            if (!x.pieces.empty())
//            {
//                state += (x.pieces[x.pieces.size()-1] + " ");
//            }
//        };
//
//        for (int i = 0; (((timeElapsedSince(start_time) < time) || (Board::expression_dict.size() < 105614388))); i++)
//        {
//            if (!(Board::expression_dict.size()%1000000))
//            {
//                std::scoped_lock str_lock(Board::thread_locker);
//                std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
//            }
//            if (i && (i%500 == 0))
//            {
//                //                    std::cout << "Unique expressions = " << Board::expression_dict.size() << '\n';
//                //                    std::cout << "check_point_score = " << check_point_score
//                //                    << ", max_score = " << max_score << ", c = " << c << '\n';
//                if (check_point_score == max_score)
//                {
//                    //                        std::cout << "c: " << c << " -> ";
//                    c += 1.4;
//                    //                        std::cout << c << '\n';
//                }
//                else
//                {
//                    //                        std::cout << "c: " << c << " -> ";
//                    c = 1.4; //if new best found, reset c and try to exploit the new best
//                    //                        std::cout << c << '\n';
//                    check_point_score = max_score;
//                }
//            }
//            state.clear();
//            while ((score = x.complete_status()) == -1)
//            {
//                temp_legal_moves = x.get_legal_moves();
//                assert(temp_legal_moves.size());
////                    auto start_time = Clock::now();
//                getString();
////                    str_convert_time += timeElapsedSince(start_time);
//                UCT = 0.0;
//                UCT_best = -DBL_MAX;
//                best_act = temp_legal_moves[0];
//                std::vector<std::string> best_acts;
//                best_acts.reserve(temp_legal_moves.size());
//
//                for (const std::string& a : temp_legal_moves)
//                {
//                    if (Nsa[state].count(a))
//                    {
//                        UCT = Qsa[state][a] + c*sqrt(log(Ns[state])/Nsa[state][a]);
//                    }
//                    else
//                    {
//                        //not explored -> explore it.
//                        best_acts.push_back(a);
//                        UCT = -DBL_MAX;
//                    }
//
//                    if (UCT > UCT_best)
//                    {
//                        best_act = a;
//                        UCT_best = UCT;
//                    }
//                }
//
//                if (best_acts.size())
//                {
//                    std::uniform_int_distribution<int> distribution(0, best_acts.size() - 1);
//                    best_act = best_acts[distribution(generator)];
//                }
//                x.pieces.push_back(best_act);
//                moveTracker.push_back(make_pair(state, best_act));
//                Ns[state]++;
//                Nsa[state][best_act]++;
//            }
//            //backprop reward `score`
//            for (auto& state_action: moveTracker)
//            {
//                Qsa[state_action.first][state_action.second] = std::max(Qsa[state_action.first][state_action.second], score);
//            }
//
//            if (score > max_score)
//            {
//                max_score = score;
//                std::scoped_lock str_lock(Board::thread_locker);
//                best_SNE = x.SNE_curr;
//                best_expression = x._to_infix();
//                orig_expression = x.expression();
//                best_expr_result = x._to_infix(x.diffeq_result);
//                orig_expr_result = x.expression(x.diffeq_result);
//                std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
//                std::cout << "Best score = " << score << ", SNE = " << best_SNE << '\n';
//                std::cout << "Best expression = " << best_expression << '\n';
//                std::cout << "Best expression (original format) = " << orig_expression << '\n';
//                std::cout << "Best diff result = " << best_expr_result << '\n';
//                std::cout << "Best expression (original format) = " << orig_expr_result << '\n';
//            }
//            x.pieces.clear();
//            moveTracker.clear();
//        }
//    };
//
//    for (unsigned int i = 0; i < num_threads; i++)
//    {
//        threads[i] = std::thread(func); //TODO: (maybe) provide a depth argument to func to specify if different threads should focus on different depth expressions (and modify the search functions accordingly)?
//    }
//
//    for (unsigned int i = 0; i < num_threads; i++)
//    {
//        threads[i].join();
//    }
//
//    std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
//    std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
//    std::cout << "Best score = " << max_score << ", SNE = " << best_SNE << '\n';
//    std::cout << "Best expression = " << best_expression << '\n';
//    std::cout << "Best expression (original format) = " << orig_expression << '\n';
//    std::cout << "Best diff result = " << best_expr_result << '\n';
//    std::cout << "Best expression (original format) = " << orig_expr_result << '\n';
//}

void RandomSearch(std::vector<std::vector<std::string>> (*diffeq)(Board&, bool),
                  size_t num_diff_eqns,
                  const Eigen::MatrixXd& data,
                  const std::vector<int>& depth,
                  const std::string& expression_type = "prefix",
                  size_t num_consts_diff = 0,
                  const std::string& method = "LevenbergMarquardt",
                  const int num_fit_iter = 1,
                  const std::string& fit_grad_method = "naive_numerical",
                  const bool cache = true,
                  const double time = 120.0 /*time to run the algorithm in seconds*/,
                  unsigned int num_threads = 0,
                  bool const_tokens = false,
                  double isConstTol = 1e-1,
                  bool use_const_pieces = false,
                  int numDataCols = 0,
                  bool mustHaveAllFeatures = true,
                  const std::vector<std::vector<std::string>>& custom_features = {},
                  const std::string& bestExpressionFileName = "",
                  const std::vector<int>& maxSize = {},
                  const std::vector<std::vector<std::string>>& additive_corrections = {},
                  bool graphEval = false,
                  int print_every = 1000000,
                  bool printDiffEq = false,
                  const std::vector<std::string>& bad_ops = {})
{
    if (num_threads == 0)
    {
        unsigned int temp = std::thread::hardware_concurrency();
        num_threads = ((temp <= 1) ? 1 : temp);
    }

    std::vector<std::thread> threads(num_threads);
    std::latch sync_point(num_threads);

    /*
     Outside of thread:
     */

    std::atomic<double> max_score{0.0};
    std::atomic<double> best_SNE{DBL_MAX};
    std::string best_expression, orig_expression, best_expr_result, orig_expr_result;
    std::vector<double> best_sne_vec;
    std::ostream* out = &std::cout;
    std::ofstream outFile;

    auto start_time = Clock::now();

    /*
     Inside of thread:
     */

    auto func = [&diffeq, &num_diff_eqns, &depth, &expression_type, &num_consts_diff, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &best_expr_result, &orig_expr_result, &const_tokens, &use_const_pieces, &numDataCols, &mustHaveAllFeatures, &custom_features, &isConstTol, &best_SNE, &best_sne_vec, &bestExpressionFileName, &maxSize, &additive_corrections, &graphEval, &print_every, &printDiffEq, &bad_ops, &outFile, &out](int thread_idx)
    {
        std::random_device rand_dev;
        std::mt19937 thread_local generator(rand_dev()); // Mersenne Twister random number generator

        Board x(diffeq, num_diff_eqns, true, depth, expression_type, num_consts_diff, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol, use_const_pieces, true, numDataCols, mustHaveAllFeatures, custom_features, maxSize, additive_corrections, graphEval, false, bad_ops);

        sync_point.arrive_and_wait();
        double score = 0.0;
        std::vector<std::string> temp_legal_moves;
        size_t temp_sz;

        for (double i = 0; (timeElapsedSince(start_time) < time); i++)
        {
            if (i && (static_cast<int>(i)%print_every == 0))
            {
                std::scoped_lock progress_lock(Board::thread_locker);
                
                if (use_const_pieces)
                {
                    std::cout << "Thread " << thread_idx << " Iteration " << i << '\n';
                    std::cout << "Thread " << thread_idx << " Unique expressions = " << Board::expression_dict.size() << '\n';
                }
                else
                {
                    std::cout << "Thread " << thread_idx << " Iteration " << i << '\n';
                }
            }
            
            for (int jdx = 0; jdx < x.num_objectives; jdx++)
            {
//                x.pieces[jdx] = {"x0", "tanh"};
                while ((score = x.complete_status(jdx)) == -1)
                {
                    temp_legal_moves = x.get_legal_moves(jdx); //the legal moves
                    temp_sz = temp_legal_moves.size(); //the number of legal moves

                    assert(temp_sz);
                    std::uniform_int_distribution<int> distribution(0, temp_sz - 1); // A random integer generator which generates an index corresponding to an allowed move
                    {
                        x.pieces[jdx].emplace_back(temp_legal_moves[distribution(generator)]); //make the randomly chosen valid move
                    }
                }
                if (jdx < x.num_objectives - 1)
                {
                    assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces[jdx], jdx) : x.getRPNdepth(x.pieces[jdx], jdx)).first == x.n[jdx]);
                    assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces[jdx], jdx) : x.getRPNdepth(x.pieces[jdx], jdx)).second);
                }
            }
//            printf("score = %f\n", score);

            if (score > max_score)
            {
                max_score = score;
                std::scoped_lock str_lock(Board::thread_locker);
                best_SNE = x.SNE_curr;
                best_sne_vec = x.SNE_curr_vec;
                best_expression = x._to_infix();
                orig_expression = x.expression();
                best_expr_result = x._to_infix(x.diffeq_result);
                orig_expr_result = x.expression(x.diffeq_result);
                if (bestExpressionFileName.size())
                {
                    outFile.open(bestExpressionFileName, std::ios::app);
                    if (outFile.is_open())
                    {
                        out = &outFile;
                    }
                    else
                    {
                        out = &std::cout;
                    }
                }
                (*out) << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
                (*out) << "Time spent fitting = " << Board::fit_time << " seconds\n";
                (*out) << "Best score = " << score << ", SNE = " << best_SNE << '\n';
                (*out) << "Squared-norm error for each equation: " << best_sne_vec << '\n';
                (*out) << "Best expression = " << best_expression << '\n';
                (*out) << "Best expression (original format) = " << orig_expression << '\n';
                if (printDiffEq)
                {
                    (*out) << "Best diff result = " << best_expr_result << '\n';
                    (*out) << "Best expression (original format) = " << orig_expr_result << '\n';
                    (*out) << "Best differential equation parameters = " << x.print_diff_params() << '\n';
                    (*out) << "Best expression parameters = " << x.print_expression_params() << '\n';
                    (*out) << "Total system result = " << best_expr_result << '\n';
                    (*out) << "Total system result (original format) = " << orig_expr_result << '\n';
                }
                if (outFile.is_open())
                {
                    outFile.close();
                }
            }
//            else
//            {
//                std::cout << "expression = " << x._to_infix() << '\n';
//                std::cout << "score = " << score << '\n';
//            }

            for (decltype(x.pieces.size()) jdx = 0; jdx < x.pieces.size(); jdx++)
            {
                x.pieces[jdx].clear();
            }
        }
        puts("Done with symbolic regression");
    };

    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads[i] = std::thread(func, i);
    }

    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads[i].join();
    }

    std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
    std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
    std::cout << "Best score = " << max_score << ", SNE = " << best_SNE << '\n';
    std::cout << "Squared-norm error for each equation: " << best_sne_vec << '\n';
    std::cout << "Best expression = " << best_expression << '\n';
    std::cout << "Best expression (original format) = " << orig_expression << '\n';
    if (printDiffEq)
    {
        std::cout << "Best diff result = " << best_expr_result << '\n';
        std::cout << "Best expression (original format) = " << orig_expr_result << '\n';
    }
}

namespace ExampleProblems
{
    void SwiftHohenbergTest(int random_seed, const char* algorithm, double time)
    {
        double threshold = 1.0;
        bool mu_equals_nu_1_only = true;
        auto data1 = ((mu_equals_nu_1_only) ?
                     createMeshgridVectors(330, 2, {0.01, 0.0}, {10.0, 6.28319}) :
                     createMeshgridVectors(10, 4, {0.01, 0.0, 0.01, 0.01}, {10.0, 6.28319, 10, 10}));
        
        if (strcmp(algorithm, "RandomSearch") == 0)
        {
            RandomSearch(SwiftHohenberg /*differential equation to solve*/,
                         3 /*number of equations in differential equation system*/,
                         data1 /*data used to solve differential equation*/,
                         std::vector<int>{5} /*fixed depths of generated solution*/,
                         "postfix" /*expression representation*/,
                         0 /*num_consts_diff: number of constants in differential equation*/,
                         "LevenbergMarquardt" /*fit method if expression contains const tokens*/,
                         5 /*number of fit iterations*/,
                         "naive_numerical" /*method for computing the gradient*/,
                         true /*cache*/,
                         time /*time to run the algorithm in seconds*/,
                         0 /*num threads*/,
                         true /*`const_tokens`: whether to include const tokens {0, 1, 2, 4}*/,
                         threshold /*threshold for which solutions cannot be constant*/,
                         false /*whether or not to include constant tokens in the generated expressions, independent of the num_consts_diff tokens in the differential equation you are trying to solve*/,
                         0 /*number of data columns that constitute labels and not independent variables/features*/,
                         true /*whether or not to include ALL of the features in all of the generated expressions*/,
                         {} /*custom features that the SR-found equations are required to contain*/,
                         "" /*filename to save current best expression found (instead of outputting them to standard out*/,
                         {} /*optional max-sizes of each of the expressions in the generated solution*/,
                         {} /*function-vector to be added to each funtion-vector found by symbolic-regressor in each iteration; logic is user-implemented*/,
                         false /*whether or not to evaluate the expression as a directed-acylclic graph (dag); maybe useful if many repeated strucures present in diffeq*/,
                         50 /*`print_every` number of expressions generated before thread prints to standard out*/,
                         false /*whether to explicitly print out the result of plugging in the best found expression into the system being solved*/,
                         std::vector<std::string>{} /*operators to restrict in the search*/);
        }
        else
        {
            SimulatedAnnealing(SwiftHohenberg /*differential equation to solve*/,
                3 /*number of equations in differential equation system*/,
                data1 /*data used to solve differential equation*/,
                std::vector<int>{9} /*fixed depths of generated solution*/,
                "postfix" /*expression representation*/,
                0/*2*/ /*num_consts_diff: number of constants in differential equation*/,
                "LevenbergMarquardt" /*fit method if expression contains const tokens*/,
                5 /*number of fit iterations*/,
                "naive_numerical" /*method for computing the gradient*/,
                true /*cache*/,
                time /*time to run the algorithm in seconds*/,
                0 /*num threads*/,
                true /*`const_tokens`: whether to include const tokens {0, 1, 2, 4}*/,
                threshold /*threshold for which solutions cannot be constant*/,
                false /*whether to include or not to include constant tokens in the generated expressions, independent of the num_consts_diff tokens in the differential equation you are trying to solve*/,
                false, /*Whether to simplify the expression on every iteration (perturbation) of the seed expression vector*/
                0 /*number of data columns that constitute labels and not independent variables/features*/,
                true /*whether or not to include ALL of the features in all of the generated expressions*/,
                {} /*custom features that the SR-found equations are required to contain*/,
                "SwiftHohenbergBest.txt", //"" /*filename to save current best expression found (instead of outputting them to standard out*/
                {} /*optional max-sizes of each of the expressions in the generated solution*/,
                {/*split("x0 -0.01 + x1 sech + 11.156528193614346 ^ 2.714063572022206e-13 * 0.010000 x0 + 6.29319 ^ 1e-08 * 0.0100003333566687 + 0.7493736126143709 + + 0.9998848754538172 x0 tanh arcsin 0.7615941559557649 x0 4 ^ / / ^ 6.283190 x1 + ~ sin 0.9171523356672744 * * 0.7827863849639187 x0 cos asin cos * * - x0 x0 + 0.003734854911714874 6.283190 x0 / ^ 7.570169558264211 + ^ 0.28580222883407974 0.010000 x0 + 10.01 + ^ 0.010000 x0 ^ 1.03 + x1 sin - * * -6.1759665127829875 -10 x1 x1 + + + x1 0.005 / 1.9195169107150692e+06 - / -0.06767485271943648 + + 0.2658022288340797 x0 + 0.9801980198019802 ^ x0 1.517923178056138 + / 0.010000 x0 + sin 0.03661899347368653 x0 + + x1 sin 10.01 0.010000 x0 + / + * ^ -0.01842414214696351 + + -")*/} /*function-vector to be added to each funtion-vector found by symbolic-regressor in each iteration; logic is user-implemented*/,
                true /*whether or not to evaluate the expression as a directed-acylclic graph (dag); maybe useful if many repeated strucures present in diffeq*/,
                50 /*`print_every` number of expressions generated before thread prints to standard out*/,
                false /*whether to explicitly print out the result of plugging in the best found expression into the system being solved*/,
                std::vector<std::string>{} /*operators to restrict in the search*/,
                {split("0 0 + x0 0 + + 0 0 + 0 -0.01 + + + 0 0 + 0 x1 + + sech + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 11.156528193614346 + + + + ^ 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 2.714063572022206e-13 + + + + + * 0 0.010000 + 0 x0 + + 0 0 + 0 6.29319 + + ^ 0 0 + 0 0 + + 0 0 + 0 1e-08 + + + * 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 0.0100003333566687 + + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 0.7493736126143709 + + + + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 0.9998848754538172 + + + + 0 x0 + tanh arcsin 0 0 + 0 0.7615941559557649 + + 0 x0 + 0 4 + ^ / / ^ 0 6.283190 + 0 x1 + + ~ sin 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 0.9171523356672744 + + + + * * 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 0.7827863849639187 + + + + + 0 0 + 0 x0 + + cos asin cos * * - 0 0 + 0 0 + + 0 0 + 0 x0 + + + 0 0 + 0 0.010000166674167114 + + 0 0 + 0 x0 + + + + 0 0 + 0 0.003734854911714874 + + 0 6.283190 + 0 x0 + / ^ 0 0 + 0 0 + + 0 0 + 0 7.570169558264211 + + + + ^ 0 0 + 0 0 + + 0 0 + 0 0.28580222883407974 + + + 0 0.010000 + 0 x0 + + 0 0 + 0 10.01 + + + ^ 0 0.010000 + 0 x0 + ^ 0 0 + 0 1.03 + + + 0 0 + 0 x1 + + sin - * * 0 0 + 0 0 + + 0 0 + 0 -6.1759665127829875 + + + 0 0 + 0 -10 + + 0 x1 + 0 x1 + + + + 0 0 + 0 x1 + + 0 0 + 0 0.005 + + / 0 0 + 0 0 + + 0 0 + 0 1.9195169107150692e+06 + + + - / 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 -0.06767485271943648 + + + + + + + 0 0 + 0 0.2658022288340797 + + 0 0 + 0 x0 + + + 0 0 + 0 0 + + 0 0 + 0 0.9801980198019802 + + + ^ 0 0 + 0 0 + + 0 0 + 0 x0 + + + 0 0 + 0 0 + + 0 0 + 0 1.517923178056138 + + + + / 0 0.010000 + 0 x0 + + sin 0 0 + 0 0.03661899347368653 + + 0 0 + 0 x0 + + + + 0 0 + 0 x1 + + sin 0 0 + 0 10.01 + + 0 0.010000 + 0 x0 + + / + * ^ 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 -0.01842414214696351 + + + + + + + + -")} /*seed expressions*/,
                false /*whether to exit right after computing the score for the seed epxression (default `false`)*/,
                random_seed /*value for random seed, < 0 means it will be set to RANDOM_SEED if RANDOM_SEED > 0 else with std::mt19937*/,
                0.0 /*T_min*/,
                0.0 /*T_max*/,
                [](double ratio, double t_val) -> double {return 0.9;} /*Temperature update `T = std::max(T_min, r*T)`, where `r` is the return-value of this function, `ratio` is defined as `T_min / T_max`, and `t_val` is the current time, where 1 time-step = 1 applied simulated-annealing perturbation */,
                "" /*file to save SNE values in each equation in the differential equation system; if empty, data not saved but outputted to screen*/,
                true /*where or not to complete the trees of each sr-expression after a new best expression-vec is found*/,
                "sub_tree" /*perturbation option: either "sub_array", "n_random", or (default) "sub_tree"*/);
        }
    }
    
    void VortexRadialProfileTest(int random_seed, const char* algorithm, double time)
    {
        double threshold = 0.04;
        auto data = createMeshgridVectors(101, 1, {0.0001}, {10.0});
        if (strcmp(algorithm, "RandomSearch") == 0)
        {
            RandomSearch(VortexRadialProfile /*differential equation to solve*/,
                         3 /*number of equations in differential equation system*/,
                         data /*data used to solve differential equation*/,
                         std::vector<int>{24} /*fixed depths of generated solution*/,
                         "prefix" /*expression representation*/,
                         0 /*num_consts_diff: number of constants in differential equation*/,
                         "LevenbergMarquardt" /*fit method if expression contains const tokens*/,
                         5 /*number of fit iterations*/,
                         "naive_numerical" /*method for computing the gradient*/,
                         true /*cache*/,
                         time /*time to run the algorithm in seconds*/,
                         0 /*num threads*/,
                         true /*`const_tokens`: whether to include const tokens {0, 1, 2, 4}*/,
                         threshold /*threshold for which solutions cannot be constant*/,
                         false /*whether or not to include constant tokens in the generated expressions, independent of the num_consts_diff tokens in the differential equation you are trying to solve*/,
                         0 /*number of data columns that constitute labels and not independent variables/features*/,
                         true /*whether or not to include ALL of the features in all of the generated expressions*/,
                         {} /*custom features that the SR-found equations are required to contain*/,
                         "", // "vortexTest.txt" /*filename to save current best expression found (instead of outputting them to standard out*/
                         {} /*optional max-sizes of each of the expressions in the generated solution*/,
                         {} /*function-vector to be added to each funtion-vector found by symbolic-regressor in each iteration; logic is user-implemented*/,
                         false /*whether or not to evaluate the expression as a directed-acylclic graph (dag); maybe useful if many repeated strucures present in diffeq*/,
                         50 /*`print_every` number of expressions generated before thread prints to standard out*/,
                         false /*whether to explicitly print out the result of plugging in the best found expression into the system being solved*/,
                         std::vector<std::string>{} /*operators to restrict in the search*/);
        }
        else
        {
            SimulatedAnnealing(VortexRadialProfile /*differential equation to solve*/,
                3 /*number of equations in differential equation system*/,
                data /*data used to solve differential equation*/,
                std::vector<int>{20} /*fixed depths of generated solution*/,
                "prefix" /*expression representation*/,
                0/*2*/ /*num_consts_diff: number of constants in differential equation*/,
                "LevenbergMarquardt" /*fit method if expression contains const tokens*/,
                5 /*number of fit iterations*/,
                "naive_numerical" /*method for computing the gradient*/,
                true /*cache*/,
                time /*time to run the algorithm in seconds*/,
                0 /*num threads*/,
                true /*`const_tokens`: whether to include const tokens {0, 1, 2, 4}*/,
                threshold /*threshold for which solutions cannot be constant*/,
                true /*whether to include or not to include constant tokens in the generated expressions, independent of the num_consts_diff tokens in the differential equation you are trying to solve*/,
                false, /*Whether to simplify the expression on every iteration (perturbation) of the seed expression vector*/
                0 /*number of data columns that constitute labels and not independent variables/features*/,
                true /*whether or not to include ALL of the features in all of the generated expressions*/,
                {} /*custom features that the SR-found equations are required to contain*/,
                "" /*filename to save current best expression found (instead of outputting them to standard out)*/,
                {} /*optional max-sizes of each of the expressions in the generated solution*/,
                {} /*function-vector to be added to each funtion-vector found by symbolic-regressor in each iteration; logic is user-implemented*/,
                false /*whether or not to evaluate the expression as a directed-acylclic graph (dag); maybe useful if many repeated strucures present in diffeq*/,
                1000000 /*`print_every` number of expressions generated before thread prints to standard out*/,
                false /*whether to explicitly print out the result of plugging in the best found expression into the system being solved*/,
                std::vector<std::string>{} /*operators to restrict in the search*/,
                {} /*seed expressions*/,
                false /*whether to exit right after computing the score for the seed epxression (default `false`)*/,
                random_seed /*value for random seed, < 0 means it will be set to RANDOM_SEED if RANDOM_SEED > 0 else with std::mt19937*/,
                0.0 /*T_min*/,
                0.0 /*T_max*/,
                [](double ratio, double t_val) -> double {return 0.9;} /*Temperature update `T = std::max(T_min, r*T)`, where `r` is the return-value of this function, `ratio` is defined as `T_min / T_max`, and `t_val` is the current time, where 1 time-step = 1 applied simulated-annealing perturbation */,
                "" /*file to save SNE values in each equation in the differential equation system; if empty, data not saved but outputted to screen*/,
                false /*where or not to complete the trees of each sr-expression after a new best expression-vec is found*/,
                "sub_tree" /*perturbation option: either "sub_array", "n_random", or (default) "sub_tree"*/);
        }
    }

    void SolitonWaveFengEq14and15LaserTest(int random_seed, const char* algorithm, double time)
    {
        double threshold = 0.001;
        Eigen::MatrixXd data(127, 3);
        data << -10.43428828745382, 0.0012964163037524623, -0.0010561600135938624, -10.317688181744055, 0.0012964163037524623, -0.0010590449646629705, -10.201088076034292, 0.0012964163037524623, -0.001061929915732083, -10.084487970324528, 0.0012964163037524623, -0.0010807860775096723, -9.967887864614763, 0.0012964163037524623, -0.0011318227224482134, -8.91848691322689, 0.0012964163037524623, -0.0010936643774922638, -8.801886807517125, 0.0012964163037524623, -0.0010965493285613778, -8.68528670180736, 0.0012964163037524623, -0.0010994342796304824, -8.568686596097598, 0.0012964163037524623, -0.0010915881967886473, -8.481236516815274, 0.0012964163037524623, -0.0010761774391480077, -7.460985591854841, 0.0012964163037524623, -0.0007317424693321756, -7.344385486145076, 0.0012964163037524623, -0.0006616950327002919, -7.2277853804353125, 0.0012964163037524623, -0.0006260437338381488, -7.111185274725549, 0.0012964163037524623, -0.0005903924349760059, -7.023735195443225, 0.0012964163037524623, -0.0005636539608293984, -6.003484270482792, 0.0012964163037524623, -0.0002904584970633939, -5.886884164773028, 0.0012964163037524623, -0.0002933434481325109, -5.770284059063264, 0.0012964163037524623, -0.00029622839920161436, -5.6536839533535, 0.0012964163037524623, -0.0002634502321286223, -5.566233874071178, 0.0012964163037524623, -0.00014361535924022599, -5.04153339837724, -0.0023812961476865346, 0.0005753938780901555, -4.487682896255862, -0.0023812961476865346, 0.0025630321446362203, -4.371082790546097, -0.0023812961476865346, 0.003170470816595319, -4.254482684836333, -0.004220152373406005, 0.0039049987818705924, -4.13788257912657, -0.0023812961476865346, 0.004744215952619693, -3.1370650051177638, -0.0477397497154341, 0.015162386082458477, -3.0301815748838123, -0.04651384556495444, 0.017571736480597892, -3.0301815748838123, -0.059385839144990904, 0.017571736480597892, -2.9135814691740496, -0.0612246953707104, 0.020200118733113566, -2.9135814691740496, -0.0722578327250274, 0.020200118733113566, -2.826131389891726, -0.0722578327250274, 0.022171405422500352, -2.272280887770348, -0.20097776852539212, 0.03380538107006965, -2.204264159439653, -0.2193663307825871, 0.03703478194098867, -2.1556807820605837, -0.23959374926550153, 0.039341496848788016, -2.1265307556331425, -0.22672175568546504, 0.04072552579346761, -1.864180517786174, -0.3664748288401468, 0.05165624361142156, -1.864180517786174, -0.3811856786459028, 0.05165624361142156, -1.835030491358733, -0.40692966580597567, 0.0529119057852572, -1.7961637894554787, -0.3958965284516587, 0.05212629254610853, -1.7767304385038507, -0.3738302537430248, 0.05316534620334737, -1.7767304385038507, -0.42899594051460965, 0.05316534620334737, -1.7378637366005965, -0.41673689900981303, 0.05538744310479808, -1.5143802006568823, -0.5503604514120964, 0.06859392065655061, -1.5143802006568823, -0.5632324449921329, 0.06859392065655061, -1.456080147802, -0.5797821510236083, 0.07148952152552111, -1.4269301213745589, -0.6092038506351203, 0.07068320834571716, -1.4269301213745589, -0.5650713012178524, 0.07068320834571716, -1.3880634194713046, -0.5944930008293643, 0.07399308257283442, -1.2520299628099139, -0.6643695374067052, 0.07771025027470406, -1.096563155196895, -0.7342460739840461, 0.08765946352014711, -1.0382631023420146, -0.75631234869268, 0.09068147600343156, -1.0188297513903866, -0.7342460739840461, 0.09032323761866505, -0.9605296985355043, -0.763667773595558, 0.09139473856823228, -0.9313796721080632, -0.7765397671755945, 0.0949841086018029, -0.34837914355924404, -0.8813545720416058, 0.1078952792642798, -0.2900790907043618, -0.8923877093959227, 0.1081109856961209, -0.2317790378494795, -0.8850322844930447, 0.1083264085954597, -0.11517893213971497, -0.8887099969444838, 0.10825685757412504, -0.11517893213971497, -0.8997431342988007, 0.10825685757412504, 0.0014211735700477846, -0.8960654218473618, 0.10889242358867447, 0.4095215435542219, -0.8776768595901667, 0.10071037481903938, 0.7301718342560726, -0.8261888852700209, 0.09222130882213057, 0.7593218606835137, -0.8133168916899844, 0.0907023965524053, 0.817621913538396, -0.8004448981099479, 0.09121769317212448, 0.8467719399658353, -0.8151557479157039, 0.09057423883080643, 0.8759219663932765, -0.7857340483041919, 0.08956661658439113, 0.9050719928207176, -0.8004448981099479, 0.08778230764296704, 0.9439386947239718, -0.7710231984984359, 0.08540322905440165, 0.9633720456755999, -0.7894117607556309, 0.08421368976011888, 1.2840223363774506, -0.6423032626980713, 0.07096278110376601, 1.2840223363774506, -0.6551752562781077, 0.07096278110376601, 1.2937390118532637, -0.6239147004408763, 0.07102632037452986, 1.352039064708146, -0.6018484257322423, 0.06793381671873283, 1.371472415659774, -0.6239147004408763, 0.06690298216680045, 1.4297724685146562, -0.5944930008293643, 0.06381047851100341, 1.4297724685146562, -0.6055261381836813, 0.06381047851100341, 1.6046726270793013, -0.4657730650289996, 0.0563485619345576, 1.6435393289825555, -0.45473992767468263, 0.054811139292982716, 1.7018393818374378, -0.43267365296604865, 0.049947734849127005, 1.7212727327890658, -0.45473992767468263, 0.05004382699481444, 1.779572785643948, -0.41796280316029266, 0.04866441212888443, 1.779572785643948, -0.42899594051460965, 0.04866441212888443, 1.954472944208593, -0.28924286735992794, 0.04236496314010472, 1.9933396461118473, -0.27820973000561094, 0.040965085587042566, 2.0710730499183576, -0.256143455296977, 0.0381653304809182, 2.0710730499183576, -0.270854305102733, 0.0381653304809182, 2.0710730499183576, -0.28188744245704994, 0.0381653304809182, 2.109939751821612, -0.2402067013407414, 0.03676545292785605, 2.1876731556281204, -0.23039946813690404, 0.03396569782173175, 2.333423287765326, -0.1715560689138802, 0.030854610209602847, 2.6346402275155505, -0.10167953233653931, 0.021605693132999516, 2.712373631322059, -0.10167953233653931, 0.01973791820879841, 2.7415236577495, -0.08696868253078335, 0.019037502612222985, 2.7998237106043806, -0.0722578327250274, 0.017636671419072176, 2.8289737370318218, -0.08329097007934436, 0.01693625582249675, 3.7326245562824916, -0.013414433502003498, 0.007472265835869461, 3.849224661992256, -0.006059008599125504, 0.006376637533005264, 3.849224661992256, -0.017092145953442495, 0.006376637533005264, 3.9658247677020206, -0.006059008599125504, 0.005281009230141067, 4.082424873411785, -0.006059008599125504, 0.004185380927276871, 4.199024979121546, -0.006059008599125504, 0.0038337565701724525, 4.315625084831311, -0.006059008599125504, 0.002806815055583041, 4.432225190541075, -0.0005424399219670362, 0.0018962403086712788, 4.54882529625084, 0.0012964163037524623, 0.0022326809651833437, 4.665425401960604, 0.0012964163037524623, 0.0014163423316503833, 4.752875481242928, 0.0012964163037524623, 0.001872028283842192, 5.77312640620336, -0.0005424399219670362, -0.0005818385550432711, 5.889726511913125, 0.0012964163037524623, -0.000584723506112388, 6.006326617622889, 0.0012964163037524623, -0.0005876084571814913, 6.12292672333265, 0.0012964163037524623, -0.0006079755495703998, 6.239526829042415, 0.0012964163037524623, -0.0006811290130572356, 7.23062772757541, 0.0012964163037524623, -0.00136232556390265, 7.347227833285174, 0.0012964163037524623, -0.0015817304103852947, 7.463827938994935, 0.0012964163037524623, -0.0018011352568679273, 7.5804280447047, 0.0012964163037524623, -0.001939549782349283, 7.667878123987023, 0.0012964163037524623, -0.0019254043838014701, 8.163428573253519, 0.0012964163037524623, -0.0017486845884493267, 8.688129048947456, 0.0012964163037524623, -0.0015615695110176451, 8.80472915465722, 0.0012964163037524623, -0.001650762588452242, 8.921329260366985, 0.0012964163037524623, -0.001896787283102471, 9.03792936607675, 0.0012964163037524623, -0.001610327152429707, 9.154529471786514, 0.0012964163037524623, -0.0015408317932039549, 9.65007992105301, 0.0012964163037524623, -0.0015530928352476675, 10.145630370319505, 0.0012964163037524623, -0.00156535387729138, 10.26223047602927, 0.0012964163037524623, -0.0015682388283604879, 10.378830581739034, 0.0012964163037524623, -0.0015711237794295918, 10.466280661021358, -0.0023812961476865346, -0.0015732874927314232;
        std::cout << "data = " << data << '\n';
        if (strcmp(algorithm, "RandomSearch") == 0)
        {
            RandomSearch(SolitonWaveFengEq14and15Laser /*differential equation to solve*/,
                9 /*number of equations in differential equation system*/,
                data /*data used to solve differential equation*/,
                std::vector<int>{3, 3} /*fixed depths of generated solution*/,
                "postfix" /*expression representation*/,
                0 /*num_consts_diff: number of constants in differential equation*/,
                "LevenbergMarquardt" /*fit method if expression contains const tokens*/,
                5 /*number of fit iterations*/,
                "naive_numerical" /*method for computing the gradient*/,
                true /*cache*/,
                time /*time to run the algorithm in seconds*/,
                0 /*num threads*/,
                true /*`const_tokens`: whether to include const tokens {0, 1, 2, 4}*/,
                threshold /*threshold for which solutions cannot be constant*/,
                true /*whether to include or not to include constant tokens in the generated expressions, independent of the num_consts_diff tokens in the differential equation you are trying to solve*/,
                 2 /*number of data columns that constitute labels and not independent variables/features*/,
                 true /*whether or not to include ALL of the features in all of the generated expressions*/,
                 {} /*custom features that the SR-found equations are required to contain*/,
                 "" /*filename to save current best expression found (instead of outputting them to standard out)*/,
                 {} /*optional max-sizes of each of the expressions in the generated solution*/,
                 {} /*function-vector to be added to each funtion-vector found by symbolic-regressor in each iteration; logic is user-implemented*/,
                 false /*whether or not to evaluate the expression as a directed-acylclic graph (dag); maybe useful if many repeated strucures present in diffeq*/,
                 50 /*`print_every` number of expressions generated before thread prints to standard out*/,
                 false /*whether to explicitly print out the result of plugging in the best found expression into the system being solved*/,
                 std::vector<std::string>{} /*operators to restrict in the search*/);
        }
        else
        {
            SimulatedAnnealing(SolitonWaveFengEq14and15Laser /*differential equation to solve*/,
                9 /*number of equations in differential equation system*/,
                data /*data used to solve differential equation*/,
                std::vector<int>{4, 4} /*fixed depths of generated solution*/,
                "postfix" /*expression representation*/,
                0 /*num_consts_diff: number of constants in differential equation*/,
                "LevenbergMarquardt" /*fit method if expression contains const tokens*/,
                5 /*number of fit iterations*/,
                "naive_numerical" /*method for computing the gradient*/,
                true /*cache*/,
                time /*time to run the algorithm in seconds*/,
                0 /*num threads*/,
                true /*`const_tokens`: whether to include const tokens {0, 1, 2, 4}*/,
                threshold /*threshold for which solutions cannot be constant*/,
                false /*whether to include or not to include constant tokens in the generated expressions, independent of the num_consts_diff tokens in the differential equation you are trying to solve*/,
                false, /*Whether to simplify the ORIGINAL expression on every iteration (perturbation) of the seed expression vector; if false a copy is maintained so that simplification on this->pieces can still happen*/
                2 /*number of data columns that constitute labels and not independent variables/features*/,
                true /*whether or not to include ALL of the features in all of the generated expressions*/,
                {} /*custom features that the SR-found equations are required to contain*/,
                "" /*filename to save current best expression found (instead of outputting them to standard out)*/,
                {} /*optional max-sizes of each of the expressions in the generated solution*/,
                {} /*function-vector to be added to each funtion-vector found by symbolic-regressor in each iteration; logic is user-implemented*/,
                false /*whether or not to evaluate the expression as a directed-acylclic graph (dag); maybe useful if many repeated strucures present in diffeq*/,
                1000000 /*`print_every` number of expressions generated before thread prints to standard out*/,
                false /*whether to explicitly print out the result of plugging in the best found expression into the system being solved*/,
                std::vector<std::string>{} /*operators to restrict in the search*/,
                {split("x0 sech tanh tanh 0 0 + 0 -6.4342880000000005 + + x0 tanh 0 2.61657 + / - /"), split("0 0 + 0 -3.2171440000000002 + + x0 sech 0 0.9640275800758169 + ^ * sech")} /*seed expressions*/,
                false /*whether to exit right after computing the score for the seed expression (default `false`)*/,
                random_seed /*value for random seed, < 0 means it will be set to RANDOM_SEED if RANDOM_SEED > 0 else with std::mt19937*/,
                0.0 /*T_min*/,
                0.0 /*T_max*/,
                [](double ratio, double t_val) -> double {return 0.9;} /*Temperature update `T = std::max(T_min, r*T)`, where `r` is the return-value of this function, `ratio` is defined as `T_min / T_max`, and `t_val` is the current time, where 1 time-step = 1 applied simulated-annealing perturbation */,
                "" /*file to save SNE values in each equation in the differential equation system; if empty, data not saved but outputted to screen*/,
                false /*where or not to complete the trees of each sr-expression after a new best expression-vec is found*/,
                "sub_tree" /*perturbation option: either "sub_array", "n_random", or (default) "sub_tree"*/);
        }
    }
    void WildfireSpreadTSTest(int random_seed, const char* algorithm, double time)
    {
        double threshold = 0.0;
        bool validation = true;
        Eigen::MatrixXd data;
        if (validation)
        {
            data = load_csv("/Users/edwardfinkelstein/SDSU_UCI/UCIFall2025/CS274E/8006177/WildfireSpreadTS/2020/fire_24332933/fire_24332933_with_target_bin.csv", 1632925, 27);
        }
        else
        {
            data = load_csv("/Users/edwardfinkelstein/SDSU_UCI/UCIFall2025/CS274E/8006177/WildfireSpreadTS/2020/fire_23654679/fire_23654679_with_target_bin.csv", 2756544, 27);
        }
            
        std::cout << "Data loaded!\nFirst 10 rows\n=============\n";
        std::cout << data.topRows(10) << '\n';
        if (strcmp(algorithm, "RandomSearch") == 0)
        {
            RandomSearch(WildfireSpreadTS /*differential equation to solve*/,
                3 /*number of equations in differential equation system*/,
                data /*data used to solve differential equation*/,
                std::vector<int>{6} /*fixed depths of generated solution*/,
                "prefix" /*expression representation*/,
                0 /*num_consts_diff: number of constants in differential equation*/,
                "LevenbergMarquardt" /*fit method if expression contains const tokens*/,
                5 /*number of fit iterations*/,
                "naive_numerical" /*method for computing the gradient*/,
                true /*cache*/,
                time /*time to run the algorithm in seconds*/,
                0 /*num threads*/,
                true /*`const_tokens`: whether to include const tokens {0, 1, 2, 4}*/,
                threshold /*threshold for which solutions cannot be constant*/,
                false /*whether to include or not to include constant tokens in the generated expressions, independent of the num_consts_diff tokens in the differential equation you are trying to solve*/,
                 1 /*number of data columns that constitute labels and not independent variables/features*/,
                 false /*whether or not to include ALL of the features in all of the generated expressions*/,
                 {} /*custom features that the SR-found equations are required to contain*/,
                 "" /*filename to save current best expression found (instead of outputting them to standard out)*/,
                 {} /*optional max-sizes of each of the expressions in the generated solution*/,
                 {} /*function-vector to be added to each funtion-vector found by symbolic-regressor in each iteration; logic is user-implemented*/,
                 false /*whether or not to evaluate the expression as a directed-acylclic graph (dag); maybe useful if many repeated strucures present in diffeq*/,
                 50 /*`print_every` number of expressions generated before thread prints to standard out*/,
                 false /*whether to explicitly print out the result of plugging in the best found expression into the system being solved*/,
                 std::vector<std::string>{} /*operators to restrict in the search*/);
        }
        else
        {
            SimulatedAnnealing(WildfireSpreadTS /*differential equation to solve*/,
                3 /*number of equations in differential equation system*/,
                data /*data used to solve differential equation*/,
                std::vector<int>{8} /*fixed depths of generated solution*/,
                "postfix" /*expression representation*/,
                0 /*num_consts_diff: number of constants in differential equation*/,
                "LevenbergMarquardt" /*fit method if expression contains const tokens*/,
                5 /*number of fit iterations*/,
                "naive_numerical" /*method for computing the gradient*/,
                true /*cache*/,
                time /*time to run the algorithm in seconds*/,
                ((validation) ? 1 : 0) /*num threads*/,
                true /*`const_tokens`: whether to include const tokens {0, 1, 2, 4}*/,
                threshold /*threshold for which solutions cannot be constant*/,
                false /*whether to include or not to include constant tokens in the generated expressions, independent of the num_consts_diff tokens in the differential equation you are trying to solve*/,
                false, /*Whether to simplify the ORIGINAL expression on every iteration (perturbation) of the seed expression vector; if false a copy is maintained so that simplification on this->pieces can still happen*/
                1 /*number of data columns that constitute labels and not independent variables/features*/,
                false /*whether or not to include ALL of the features in all of the generated expressions*/,
                {{"x23", "x24", "x25"}} /*custom features that the SR-found equations are required to contain*/,
                "",// "BestNextDayFire.txt" /*filename to save current best expression found (instead of outputting them to standard out)*/,
                {} /*optional max-sizes of each of the expressions in the generated solution*/,
                {} /*function-vector to be added to each funtion-vector found by symbolic-regressor in each iteration; logic is user-implemented*/,
                false /*whether or not to evaluate the expression as a directed-acylclic graph (dag); maybe useful if many repeated strucures present in diffeq*/,
                50 /*`print_every` number of expressions generated before thread prints to standard out*/,
                false /*whether to explicitly print out the result of plugging in the best found expression into the system being solved*/,
                std::vector<std::string>{} /*operators to restrict in the search*/,
                {split("-2.640000 0.051731 x7 sqrt x15 8.000000 - - / / -1803.016571 x21 36.293228 x0 * * x17 -843.000000 + x15 15893.000000 + + + + x6 x19 -100.000000 x8 + - / x14 -211800 / x15 + + / - x18 x24 - x2 x7 x0 1684.200012 - - + - + -7.446376466569234 -3.225653 x6 8.800000 - + -508 + -0.9081765689798138 38.000000 x16 sin / * + + x1 x0 - 0.0007699998478223693 + -16.82119949898502 + x0 x15 ^ -100 + -279.200012 -2.640000 x16 / + + + -28.995355508740936 + + -88.959518 1.000000 88.856491 x1 / / * -2118 x8 2118.000000 - x22 ~ + + + 0.00077 x5 + * 25.400000 x20 1.0021072170678698 / ^ 15893.000000 x22 x8 + + x1 -1405.000000 - 15666.000000 x24 + + + + x0 x25 + 16.000000 x23 + 9736 - / ~ / - + *")} /*seed expressions*/,
                validation /*whether to exit right after computing the score for the seed expression (default `false`)*/,
                random_seed /*value for random seed, < 0 means it will be set to RANDOM_SEED if RANDOM_SEED > 0 else with std::mt19937*/,
                0.0 /*T_min*/,
                0.0 /*T_max*/,
                [](double ratio, double t_val) -> double {return 0.9;} /*Temperature update `T = std::max(T_min, r*T)`, where `r` is the return-value of this function, `ratio` is defined as `T_min / T_max`, and `t_val` is the current time, where 1 time-step = 1 applied simulated-annealing perturbation */,
                "" /*file to save SNE values in each equation in the differential equation system; if empty, data not saved but outputted to screen*/,
                false /*where or not to complete the trees of each sr-expression after a new best expression-vec is found*/,
                "sub_tree" /*perturbation option: either "sub_array", "n_random", or (default) "sub_tree"*/);
        }
    }
    void InPaintWildfireSpreadTSTest(int random_seed, const char* algorithm, double time)
    {
        double threshold = 0.0;
        bool validation = true;
        Eigen::MatrixXd data;
        if (validation)
        {
            data = load_csv("/Users/edwardfinkelstein/SDSU_UCI/UCIFall2025/CS274E/8006177/WildfireSpreadTS/2020/fire_24332933/fire_24332933_inpainting_dataset.csv", 1882110, 103);
        }
        else
        {
            data = load_csv("/Users/edwardfinkelstein/SDSU_UCI/UCIFall2025/CS274E/Deep-Gen-Project/data/fire_23654679/fire_23654679_inpainting_dataset.csv", 2834850, 103);
        }
        std::cout << "Data loaded!\nFirst 10 rows\n=============\n";
        std::cout << data.topRows(10) << '\n';
        if (strcmp(algorithm, "RandomSearch") == 0)
        {
            RandomSearch(InPaintWildfireSpreadTS /*differential equation to solve*/,
                3 /*number of equations in differential equation system*/,
                data /*data used to solve differential equation*/,
                std::vector<int>{3} /*fixed depths of generated solution*/,
                "prefix" /*expression representation*/,
                0 /*num_consts_diff: number of constants in differential equation*/,
                "LevenbergMarquardt" /*fit method if expression contains const tokens*/,
                5 /*number of fit iterations*/,
                "naive_numerical" /*method for computing the gradient*/,
                true /*cache*/,
                time /*time to run the algorithm in seconds*/,
                0 /*num threads*/,
                true /*`const_tokens`: whether to include const tokens {0, 1, 2, 4}*/,
                threshold /*threshold for which solutions cannot be constant*/,
                false /*whether to include or not to include constant tokens in the generated expressions, independent of the num_consts_diff tokens in the differential equation you are trying to solve*/,
                 1 /*number of data columns that constitute labels and not independent variables/features*/,
                 false /*whether or not to include ALL of the features in all of the generated expressions*/,
                 {{"x100", "x101"}} /*custom features that the SR-found equations are required to contain*/,
                 "" /*filename to save current best expression found (instead of outputting them to standard out)*/,
                 {} /*optional max-sizes of each of the expressions in the generated solution*/,
                 {} /*function-vector to be added to each funtion-vector found by symbolic-regressor in each iteration; logic is user-implemented*/,
                 false /*whether or not to evaluate the expression as a directed-acylclic graph (dag); maybe useful if many repeated strucures present in diffeq*/,
                 50 /*`print_every` number of expressions generated before thread prints to standard out*/,
                 false /*whether to explicitly print out the result of plugging in the best found expression into the system being solved*/,
                 std::vector<std::string>{} /*operators to restrict in the search*/);
        }
        else
        {
            SimulatedAnnealing(InPaintWildfireSpreadTS /*differential equation to solve*/,
                3 /*number of equations in differential equation system*/,
                data /*data used to solve differential equation*/,
                std::vector<int>{7} /*fixed depths of generated solution*/,
                "prefix" /*expression representation*/,
                0 /*num_consts_diff: number of constants in differential equation*/,
                "LevenbergMarquardt" /*fit method if expression contains const tokens*/,
                5 /*number of fit iterations*/,
                "naive_numerical" /*method for computing the gradient*/,
                true /*cache*/,
                time /*time to run the algorithm in seconds*/,
                ((validation) ? 1 : 0) /*num threads*/,
                true /*`const_tokens`: whether to include const tokens {0, 1, 2, 4}*/,
                threshold /*threshold for which solutions cannot be constant*/,
                false /*whether to include or not to include constant tokens in the generated expressions, independent of the num_consts_diff tokens in the differential equation you are trying to solve*/,
                false, /*Whether to simplify the ORIGINAL expression on every iteration (perturbation) of the seed expression vector; if false a copy is maintained so that simplification on this->pieces can still happen*/
                1 /*number of data columns that constitute labels and not independent variables/features*/,
                false /*whether or not to include ALL of the features in all of the generated expressions*/,
                {{"x100", "x101"}} /*custom features that the SR-found equations are required to contain*/,
                "", //"BestInpaint.txt" /*filename to save current best expression found (instead of outputting them to standard out)*/
                {} /*optional max-sizes of each of the expressions in the generated solution*/,
                {} /*function-vector to be added to each funtion-vector found by symbolic-regressor in each iteration; logic is user-implemented*/,
                false /*whether or not to evaluate the expression as a directed-acylclic graph (dag); maybe useful if many repeated strucures present in diffeq*/,
                50 /*`print_every` number of expressions generated before thread prints to standard out*/,
                false /*whether to explicitly print out the result of plugging in the best found expression into the system being solved*/,
                std::vector<std::string>{} /*operators to restrict in the search*/,
                {split("/ + * + + ln cos x46 + + 0 0 + 0 -3.4799761065034414 + * + 0 x95 + 0 1.620943 + + 0 x95 ~ x48 ~ ^ + cos x41 + 0 52.64009483497598 - + 0 2.7907071011403315 cos x93 + ^ + * + 0 3.1585732538600397 ^ x12 x54 + cos x21 + 0.365382 x13 + ~ + 0 x20 + + 0 0 + 0 1.5729403267948965 * + + + 0 0 + 0 0 + + 0 0 + 0 x3 sqrt + + 0 0 + 0 x48 - + / - + + 0 0 + 0 0.9867622178470573 - - 7169.463400 x100 * x17 11181230.000000 + acos tanh x23 + + 0 0 + 0 57.67636600070402 sin + + + 0 0 + 0 -2.884980 + + 0 0 + 0 x50 - sqrt ^ ^ + 0 x17 + 0 x59 + + 0 x71 ~ x87 ^ + + + 0 0 + 0 0 + + 0 0 + 0 0.9910929232006058 * + + 0 0 + 0 -4.0405169999999995 * - x61 x101 + 0 -0.04344899097047564")} /*seed expressions*/,
                validation /*whether to exit right after computing the score for the seed expression (default `false`)*/,
                random_seed /*value for random seed, < 0 means it will be set to RANDOM_SEED if RANDOM_SEED > 0 else with std::mt19937*/,
                0.0 /*T_min*/,
                0.0 /*T_max*/,
                [](double ratio, double t_val) -> double {return 0.9;} /*Temperature update `T = std::max(T_min, r*T)`, where `r` is the return-value of this function, `ratio` is defined as `T_min / T_max`, and `t_val` is the current time, where 1 time-step = 1 applied simulated-annealing perturbation */,
                "" /*file to save SNE values in each equation in the differential equation system; if empty, data not saved but outputted to screen*/,
                false /*where or not to complete the trees of each sr-expression after a new best expression-vec is found*/,
                "sub_tree" /*perturbation option: either "sub_array", "n_random", or (default) "sub_tree"*/);
        }
    }

    void WierdTrackFitterTest(int random_seed, const char* algorithm, double time)
    {
        double threshold = 0.0;
        unsigned int num_threads = 0;
        int track_idx = 2;
        constexpr const char* file_path[] =
        {
            "/Users/edwardfinkelstein/SDSU_UCI/WhitesonResearch/TrackProject/stubborn_track_csvs/v1_noiseless_69000event100000003-hits_Z.csv",
            "/Users/edwardfinkelstein/SDSU_UCI/WhitesonResearch/TrackProject/stubborn_track_csvs/event1000000039-hits_X.csv",
            "/Users/edwardfinkelstein/SDSU_UCI/WhitesonResearch/TrackProject/stubborn_track_csvs/v20260122_163839__train10_test10__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01event100000003-hits_Z.csv"
        };
        constexpr int sizes[] = {61, 85, 169};
        const std::vector<std::string> seed_exprs =
        {
            std::vector<std::string>
            {
                "* * / * 2 6.392626 + x0 -0.061502 - - 0.985492 x0 ^ x0 0.285370 ^ ^ * x0 0.962481 ^ x0 -0.798924 cos ^ 8.423473 x0",
                "-15.251 x0 tanh 9.222 x0 * sin ^ *",
                "-34.520199 9.030267 sqrt x0 -1.453420 ^ - sech *",
                "-34.520199 9.030267 sqrt x0 -1.453420 ^ - sech * 4 x0 0.077712 / cos ~ * +",
                "* -34.520199 sech - sqrt 9.030267 ^ x0 -1.453420",
                "32.963733 0.498148 7.438650 x0 * cos ^ * -0.629287 x0 11.182829 * 2.156483 - sech + *",
                "0.507194 -5.130599 x0 -7.361629 * cos + ^ -0.622545 -2.263896 -11.663861 x0 * - sech + * 4 2.492631 x0 - ^ cos + 0.981453 x0 + -72.499292 ^ +",
                "0.509134 -5.151904 x0 7.366820 * cos + ^ -0.621943 -2.278212 11.746326 x0 * + sech + * 4 2.492631 x0 - ^ cos + 0.981511 x0 + -72.822877 ^ + 0.010065 -0.481231 x0 + / -",
                "0.508292 -5.130599 x0 7.306336 * cos + ^ -0.622545 -2.263896 -11.663861 x0 * - sech + * 4 2.492631 x0 - ^ cos + 0.987333 x0 + -72.499292 ^ + x0 sin -3.515159 exp / cos +",
                "",
            }[8],
            std::vector<std::string>
            {
                "31.554215 18.997395 16 x0 * cos ^ -",
                "x0 -0.449935 + x0 arccos x0 0.499351 ^ * x0 x0 arcsin ^ arccos sech - /"
            }[1],
            std::vector<std::string>
            {
                "2.345027 x0 x0 sqrt cos 2.266180070913597 - / +",
                "0.832126 x0 x0 sqrt cos 2 - / * -21.615306 4.207354924039483 168.000000 x0 + sqrt * cos * -",
                "-0.452316 x0 0.558241 x0 sqrt cos ^ / * 23.928732 4.207354924039483 168.000000 x0 + sqrt * cos * + 3.467233 -21.559683 1.014850 x0 ^ cos tanh * - +",
                "-0.432626 x0 1.882588 x0 sqrt cos ^ * * 25.974640 4.207354924039483 168.220938 x0 + sqrt * cos * + 2.565390 -22.862463 1.014421 x0 ^ cos tanh * - + 7.407921 -21.205194 2 x0 ln ^ - sin * +",
                "-0.436897 x0 0.542974 x0 sqrt cos ^ / * -25.464032 4.207354924039483 168.220938 x0 + sqrt * cos * - 2.526539 -23.049172 1.014421 x0 ^ cos tanh * - + -7.157784 -2.448741 2 x0 ln ^ + cos * - x0 0.595393 ^ 118.95633426995997 x0 1.000996 ^ - / +",
                "-0.440023 x0 0.542656 x0 sqrt cos ^ / * -25.646620 4.207354924039483 168.220938 x0 + sqrt * cos * - 2.850168 23.229391 1.014421 x0 ^ cos tanh * + + 7.688486 -49.540434 2 x0 ln ^ + cos * - x0 0.576049 ^ 118.95633426995997 x0 1.000996 ^ - / + 5.056729 x0 0.988960 exp / - cos -4.366545 * -",
                "-0.435995 x0 0.537501 x0 sqrt cos ^ / * 25.633881 4.207354924039483 168.220938 x0 + sqrt * cos * + 2.659658 23.229391 1.014344 x0 ^ cos tanh * + + 7.991940 -0.867100 2 x0 ln ^ + sin * + x0 0.579037 ^ 118.95633426995997 x0 1.000973 ^ - / + -1.180394 x0 2.685784865116654 / - cos 4.122059 * + 1.502449 x0 0.7371027432716666 ^ * cos arcsin +",
                "-1.435159 x0 1.044739 x0 sqrt cos ^ ^ * -26.235216 4.207354924039483 168.000000 x0 + sqrt * cos * - x0 -25.311785 1.014267 x0 ^ cos tanh * - + 8.292410 8.438624 x0 0.6931471805599453 ^ + sin * - -103.794895 x0 + 118.95633426995997 x0 1.000946 ^ - / + 1.126893 x0 2.685784865116654 / + cos -4.426079 * - 1.502449 x0 0.7371027432716666 ^ * cos arcsin + -1.940061 103.614820 x0 - 0.2658022288340797 * cos * exp +",
                "-1.435127 x0 1.044677 x0 sqrt cos ^ ^ * -26.250286 4.207354924039483 168.000000 x0 + sqrt * cos * - x0 -25.143824 1.014267 x0 ^ cos tanh * - + 8.212401 1.007622 x0 0.6931471805599453 ^ - sin * - -104.520203 x0 + 118.95633426995997 x0 0.995431 / - / + 1.137694 x0 2.685784865116654 / + cos -4.486179 * - 1.502449 x0 0.7371027432716666 ^ * cos arcsin + 1.952246 -2.721564 x0 - 0.2658022288340797 * cos * exp + x0 -0.45018598229727835 * sin 0.5 x0 tanh - / +",
                "-1.435127 x0 1.044676 x0 sqrt cos ^ ^ * -26.289403 4.207354924039483 168.000000 x0 + sqrt * cos * - x0 -25.114706 1.014267 x0 ^ cos tanh * - + 8.289811 1.008218 x0 0.6931471805599453 ^ - sin * - -104.995860 x0 + 118.95633426995997 x0 1.000949 ^ - / + 1.130615 x0 2.685784865116654 / + cos -4.486179 * - 1.502449 x0 0.7371027432716666 ^ * cos arcsin + 1.942679 -2.597723 x0 - 0.2658022288340797 * cos * exp + x0 -0.45018598229727835 * sin 0.466439 ~ / + 2 1.222537 / x0 1.034203 2.718281828459045 - / cos * +",
                "-1.434598 x0 1.044655 x0 sqrt cos ^ ^ * 26.204516 4.207354924039483 168.000000 x0 + sqrt * cos * + x0 25.178137 1.014267 x0 ^ cos tanh * + + -8.318077 -4.128074 x0 0.6931471805599453 ^ + sin * + -104.910201 x0 + 118.95633426995997 x0 0.995676 / - / + 0.985580 x0 2.685784865116654 / * sin 4.600156 * - 1.502449 x0 0.7371027432716666 ^ * cos arcsin + 1.967990 3.355986 x0 - 0.2658022288340797 * sin * exp + x0 -0.45018598229727835 * sin 1.502677 sech / - 1.003001 168.000000 ^ x0 1.030905 2.718281828459045 - / cos * +",
                "-1.434557 x0 1.044505 x0 sqrt cos ^ ^ * -26.210051 4.207354924039483 168.000000 x0 + sqrt * cos * - x0 25.172898 1.014294 x0 ^ cos tanh * + + -8.254968 0.983557 x0 0.6931471805599453 ^ - sin * + -105.189336 x0 + 118.95633426995997 x0 1.004240 * - / + 0.985180 x0 2.685784865116654 / * sin -4.586794 * + 1.502449 x0 0.7371027432716666 ^ * cos arcsin + 1.949297 97.939188 x0 - 0.2658022288340797 * sin * exp + x0 -0.45018598229727835 * sin -0.4107263935558514 / + 1.722899 x0 -1.6880058284590451 / cos * + x0 sech 0.696976831813758 x0 * cos + +",
                "-1.438588 x0 1.043900 x0 sqrt cos ^ ^ * -26.218935 4.207354924039483 168.000000 x0 + sqrt * cos * - x0 24.750491 1.014294 x0 ^ cos tanh * + + 8.151847 1.023804 x0 0.6931471805599453 ^ - sin * - -106.088260 x0 + 118.95633426995997 x0 0.995287 / - / + 0.985199 x0 2.685784865116654 / * sin -4.586794 * + 1.502449 x0 0.997872 ^ ^ log cos + -1.995201 44.248750 x0 + 0.2658022288340797 * sin * exp + x0 -0.45018598229727835 * sin -0.4107263935558514 / + 1.497965 x0 -1.6880058284590451 / cos * + x0 sech 0.696976831813758 x0 * cos + + 0.031806 tanh 4 x0 0.7615941559557649 - * sin / -",
                "-2.446553 x0 1.026091 x0 sqrt cos ^ ^ * -26.239310 4.207354924039483 168.000000 x0 + sqrt * cos * - x0 25.089302 1.014294 x0 ^ cos tanh * + + -7.997338 -1.057546 x0 0.6931471805599453 ^ + sin * - -106.322846 x0 + 118.95633426995997 x0 0.995131 / - / + 0.985374 x0 2.685784865116654 / * sin -4.973426 * + 1.502449 x0 0.997305 ^ ^ log cos + -2.066466 20.589308 x0 + 0.2658022288340797 * sin * exp + x0 -0.45018598229727835 * sin -0.4107263935558514 / + 1.609716 x0 -1.6880058284590451 / cos * + x0 0.696976831813758 x0 * cos + + 0.033090 4 x0 0.7615941559557649 - * sin / - 3.751510 1.052177 x0 0.993021 ^ ^ + sqrt sin +",
                "-2.446317 x0 1.026171 x0 sqrt cos ^ ^ * -26.267023 4.207354924039483 168.000000 x0 + sqrt * cos * - x0 25.114484 1.014280 x0 ^ cos tanh * + + -8.094348 -3.652141 x0 0.6931471805599453 ^ - cos * - -106.639258 x0 + 118.95633426995997 x0 0.995137 / - / + 0.985554 x0 2.685784865116654 / * sin 5.048220 * - 1.502449 x0 0.997721 ^ ^ log cos + 2.081938 2.962750 x0 - 0.2658022288340797 * sin * exp + x0 -0.45018598229727835 * sin -2.744077 * + -1.635574 x0 -1.6880058284590451 / cos * - x0 0.696976831813758 x0 * cos + + 0.031400 4 x0 0.7615941559557649 - * sin / - 4.063501 1.052177 x0 0.992804 ^ ^ + sqrt sin + 1.027196 ~ 2 x0 0.8414709848078965 ^ * * cos +",
                "-2.446627 x0 1.026127 x0 sqrt cos ^ ^ * -26.155498 4.207354924039483 168.000000 x0 + sqrt * cos * - x0 -25.093806 1.014280 x0 ^ cos tanh * - + 8.001965 -0.515339 x0 0.6931471805599453 ^ - cos * - -106.697598 x0 + 118.95633426995997 x0 1.004942 * - / + x0 0.985907 2.685784865116654 / * sin 5.048220 * - 1.502449 x0 0.997906 ^ ^ log cos + 2.070110 2.852405 x0 - 0.2658022288340797 * sin * exp + x0 -0.45018598229727835 * sin 2.698598 * - -1.678895 x0 -1.6880058284590451 / cos * - x0 0.696976831813758 x0 * cos + + 0.032015 4 x0 0.7615941559557649 - * sin / - 3.218607 1.052177 x0 0.992804 ^ ^ + sqrt sin + 1.027706 ~ 2 x0 0.8414709848078965 ^ * * cos + 168.000000 0.012165 11.175638 x0 54.598150033144236 - + / * +",
                "-2.444296 x0 1.026203 x0 sqrt cos ^ ^ * -26.199496 4.207354924039483 168.000000 x0 + sqrt * cos * - x0 25.019477 1.014280 x0 ^ cos tanh * + + -8.058567 -2.633602 x0 0.6931471805599453 ^ + cos * - 11.696530 118.95633426995997 x0 0.995059 / - / + x0 0.3670833851233197 * sin 4.957468 * - 1.502449 x0 0.997889 ^ ^ log cos + -2.060603 44.498356 x0 + 0.2658022288340797 * sin * exp + x0 -0.45018598229727835 * sin -2.701240 * + -1.599485 x0 -1.6880058284590451 / cos * - x0 0.696976831813758 x0 * cos + + 0.032499 4 x0 0.7615941559557649 - * sin / - 2.702079 1.052177 x0 0.992888 ^ ^ + sqrt sin + 1.027621 2 x0 0.8414709848078965 ^ * * cos + 1.225249 2.020931 11.175638 x0 54.598150033144236 - + / - - -6.130043 1.218107 x0 + x0 6.804936 - cos / / -",
                "+ + - + + - + + + + + + - - + - * -2.443340 ^ x0 ^ 1.026195 cos sqrt x0 * -26.141684 cos * 4.207354924039483 sqrt + 168.000000 x0 + x0 * 25.068585 tanh cos ^ 1.014280 x0 * -8.087407 cos + -2.638495 ^ x0 0.6931471805599453 / -11.661316 - 118.95633426995997 + x0 0.579601 * sin * x0 0.3670833851233197 -4.949110 cos log ^ 1.502449 ^ x0 0.998111 exp * 2.027208 sin * - 2.782294 x0 0.2658022288340797 * sin * x0 -0.45018598229727835 -2.666797 * 1.509721 cos / x0 -1.6880058284590451 + x0 cos * 0.696976831813758 x0 / 0.032277 sin * 4 - x0 0.7615941559557649 sin sqrt + 2.923343 ^ 1.052177 ^ x0 0.992888 cos * 1.027655 * 2 ^ x0 0.8414709848078965 - 1.244610 / 2.026644 + 11.175638 - x0 54.598150033144236 / -6.551556 / + 1.214443 x0 cos - x0 -2.647537 / -0.036732 asin cos sqrt + 1.132285 x0",
                "+ + + - + + - + + + + + + - - + - * -2.443340 * x0 * 1.026195 cos sqrt x0 * -26.141684 cos * 4.207354924039483 sqrt + 168.000000 x0 + x0 * 25.068585 tanh cos * 1.014280 x0 * -8.087407 cos + -2.638495 * x0 0.6931471805599453 * -11.661316 - 118.95633426995997 + x0 0.579601 * sin * x0 0.3670833851233197 -4.949110 cos tanh * 1.502449 * x0 0.998111 cos * 2.027208 sin * - 2.782294 x0 0.2658022288340797 * sin * x0 -0.45018598229727835 -2.666797 * 1.509721 cos * x0 -1.6880058284590451 + x0 cos * 0.696976831813758 x0 * 0.032277 sin * 4 - x0 0.7615941559557649 sin sqrt + 2.923343 * 1.052177 * x0 0.992888 cos * 1.027655 * 2 * x0 0.8414709848078965 - 1.244610 * 2.026644 + 11.175638 - x0 54.598150033144236 * -6.551556 * + 1.214443 x0 cos - x0 -2.647537 * -0.036732 asin cos sqrt + 1.132285 x0 tanh cos * * x0 * 2 1.576287 1.113573",
                "+ * sin * + x0 sin sqrt x0 0.13599420224810638 ~ + * 0.223351 x0 -39.031218 * * x0 0.272925 - -1.884260 sin * 1.096031 sqrt x0",
                "x0 x0 sqrt sin + 0.13599420224810638 * sin 29.034195 -0.000009 x0 * x0 x0 * * + * x0 0.270243 * -1.900608 1.095518 x0 sqrt * sin - * + 7.504397 x0 0.18450196567500599 * cos * -",
                "x0 x0 sqrt sin + 0.13599420224810638 * sin 29.040990 -0.000009 x0 * x0 x0 * * + * x0 0.271124 * -1.899330 1.094607 x0 sqrt * sin - * + 7.605615 x0 0.18450196567500599 * cos * - 0.2787780894020867 0.999693 x0 2 - * * cos asin +",
                "x0 x0 sqrt sin + 0.13599420224810638 * sin 28.886312 -0.000008 x0 * x0 x0 * * + * x0 0.272853 * -1.891366 1.094294 x0 sqrt * sin - * + 7.617770 x0 0.18450196567500599 * cos * - 0.2787780894020867 0.999693 x0 2 - * * cos asin + x0 sqrt sqrt x0 -0.36787944117144233 * sin * +",
                "x0 x0 sqrt sin + 0.13599420224810638 * sin 27.854452 0.000008 x0 * x0 x0 * * - * x0 0.272853 * 1.094294 x0 sqrt * sin ~ * + -8.396368 x0 0.18450196567500599 * cos * + 0.2787780894020867 x0 2 - * cos asin + 4.530500 x0 -0.36787944117144233 * sin * + 15.425325 x0 x0 cos + 0.033472 ~ x0 sech sqrt + * * +",
                "x0 x0 sqrt sin + 0.13599420224810638 * sin 27.391414 x0 -0.000007 * x0 x0 * * + * 0.321608 x0 * 1.092289 x0 sqrt * sin sin * - 8.015060 x0 0.18450196567500599 * cos * - 0.2787780894020867 x0 0.898195 * * cos arcsin + 5.064484 x0 -0.36787944117144233 * sin * + 7.763419 x0 x0 0.879127 + + -0.033499 x0 sech sqrt + * * + 0.058354 x0 * x0 0.2627831798005332 * sin * -",
                "x0 x0 sqrt sin + 0.13599420224810638 * sin 27.320094 0.000007 x0 * x0 x0 * * - * -0.323439 x0 * 1.092289 x0 sqrt * sin sin * + 8.034637 x0 0.18450196567500599 * cos * - 0.2787780894020867 x0 0.898195 * * cos arcsin + 5.075383 x0 -0.36787944117144233 * sin * + 7.763419 x0 x0 1 + + -0.033530 x0 sech sqrt + * * + -0.059082 x0 * x0 0.2627831798005332 * sin * + x0 0.5175124998053154 * -0.862119 + cos arcsin -",
                "x0 x0 sqrt sin + 0.13599420224810638 * sin 27.320094 0.000007 x0 * x0 x0 * * - * -0.323439 x0 * 1.092289 x0 sqrt * sin sin * + 8.034637 x0 0.18450196567500599 * cos * - 0.2787780894020867 x0 0.898195 * * cos arcsin + 5.075383 x0 -0.36787944117144233 * sin * + 7.763419 x0 x0 1 + + -0.033530 x0 sech sqrt + * * + -0.059082 x0 * x0 0.2627831798005332 * sin * + x0 0.5175124998053154 * -0.862119 + cos arcsin - 0.4515827052894548 x0 x0 tanh - * sin 2.799731 * +",
            }.back()
        };
        assert(track_idx < static_cast<int>(seed_exprs.size()));
        std::cout << "seed_exprs[" << track_idx << "] = {" << seed_exprs[track_idx] << "}\n";
        Eigen::MatrixXd data = load_csv(file_path[track_idx], sizes[track_idx], 2, false /*no header in these `file_path` files*/);
        std::cout << "data = " << data << '\n';
        if (strcmp(algorithm, "RandomSearch") == 0)
        {
            RandomSearch(WierdTrackFitter /*differential equation to solve*/,
                2 /*number of equations in differential equation system*/,
                data /*data used to solve differential equation*/,
                std::vector<int>{3, 2} /*fixed depths of generated solution*/,
                "postfix" /*expression representation*/,
                0 /*num_consts_diff: number of constants in differential equation*/,
                "LevenbergMarquardt" /*fit method if expression contains const tokens*/,
                5 /*number of fit iterations*/,
                "naive_numerical" /*method for computing the gradient*/,
                true /*cache*/,
                time /*time to run the algorithm in seconds*/,
                num_threads /*num threads*/,
                true /*`const_tokens`: whether to include const tokens {0, 1, 2, 4}*/,
                threshold /*threshold for which solutions cannot be constant*/,
                true /*whether to include or not to include constant tokens in the generated expressions, independent of the num_consts_diff tokens in the differential equation you are trying to solve*/,
                 1 /*number of data columns that constitute labels and not independent variables/features*/,
                 false /*whether or not to include ALL of the features in all of the generated expressions*/,
                 {} /*custom features that the SR-found equations are required to contain*/,
                 "WierdTrackSR.txt", // "" /*filename to save current best expression found (instead of outputting them to standard out)*/,
                 std::vector<int>{9, 7} /*optional max-sizes of each of the expressions in the generated solution*/,
                 {/*split(seed_exprs[track_idx])*/} /*function-vector to be added to each funtion-vector found by symbolic-regressor in each iteration; logic is user-implemented*/,
                 false /*whether or not to evaluate the expression as a directed-acylclic graph (dag); maybe useful if many repeated strucures present in diffeq*/,
                 1000000 /*`print_every` number of expressions generated before thread prints to standard out*/,
                 false /*whether to explicitly print out the result of plugging in the best found expression into the system being solved*/,
                 std::vector<std::string>{"exp", "ln", "log", "^", "/"} /*operators to restrict in the search*/);
        }
        else
        {
            SimulatedAnnealing(WierdTrackFitter /*differential equation to solve*/,
                2 /*number of equations in differential equation system*/,
                data /*data used to solve differential equation*/,
                std::vector<int>{14} /*fixed depths of generated solution*/,
                "postfix" /*expression representation*/,
                0 /*num_consts_diff: number of constants in differential equation*/,
                "LevenbergMarquardt" /*fit method if expression contains const tokens*/,
                5 /*number of fit iterations*/,
                "naive_numerical" /*method for computing the gradient*/,
                true /*cache*/,
                time /*time to run the algorithm in seconds*/,
                num_threads /*num threads*/,
                true /*`const_tokens`: whether to include const tokens {0, 1, 2, 4}*/,
                threshold /*threshold for which solutions cannot be constant*/,
                true /*whether to include or not to include constant tokens in the generated expressions, independent of the num_consts_diff tokens in the differential equation you are trying to solve*/,
                false, /*Whether to simplify the ORIGINAL expression on every iteration (perturbation) of the seed expression vector; if false a copy is maintained so that simplification on this->pieces can still happen*/
                1 /*number of data columns that constitute labels and not independent variables/features*/,
                true /*whether or not to include ALL of the features in all of the generated expressions*/,
                {} /*custom features that the SR-found equations are required to contain*/,
                "WierdTrackSR.txt", // "" /*filename to save current best expression found (instead of outputting them to standard out)*/
                std::vector<int>{92} /*optional max-sizes of each of the expressions in the generated solution*/,
                {split(seed_exprs[track_idx])} /*function-vector to be added to each funtion-vector found by symbolic-regressor in each iteration; logic is user-implemented*/,
                false /*whether or not to evaluate the expression as a directed-acylclic graph (dag); maybe useful if many repeated strucures present in diffeq*/,
                1000000 /*`print_every` number of expressions generated before thread prints to standard out*/,
                false /*whether to explicitly print out the result of plugging in the best found expression into the system being solved*/,
                std::vector<std::string>{"exp", "ln", "log", "^", "/"} /*operators to restrict in the search*/,
                {split(seed_exprs[track_idx])} /*seed expressions*/,
                (num_threads == 1) /*whether to exit right after computing the score for the seed expression (default `false`)*/,
                random_seed /*value for random seed, < 0 means it will be set to RANDOM_SEED if RANDOM_SEED > 0 else with std::mt19937*/,
                0.0 /*T_min*/,
                0.0 /*T_max*/,
                [](double ratio, double t_val) -> double {return 0.9;} /*Temperature update `T = std::max(T_min, r*T)`, where `r` is the return-value of this function, `ratio` is defined as `T_min / T_max`, and `t_val` is the current time, where 1 time-step = 1 applied simulated-annealing perturbation */,
                "WierdTrackSR.txt" /*file to save SNE values in each equation in the differential equation system; if empty, data not saved but outputted to screen*/,
                false /*where or not to complete the trees of each sr-expression after a new best expression-vec is found*/,
                "sub_tree" /*perturbation option: either "sub_array", "n_random", or (default) "sub_tree"*/);
        }
    }
};

int get_random_seed(int argc, char *argv[])
{
    int random_seed = RANDOM_SEED;
    if (argc == 2)
    {
        try
        {
            random_seed = std::stoi(argv[1]);
        }
        catch (std::exception& e)
        {
            std::cout << "Error when doing `std::stoi(argv[1])`,\nmessage = "
            << e.what() << '\n';
            exit(1);
        }
    }
    return random_seed;
}

enum class ProblemOption
{
    SwiftHohenberg,
    VortexRadialProfile,
    SolitonWaveFengEq14and15Laser,
    WildfireSpreadTS,
    InPaintWildfireSpreadTS,
    WierdTrackFitter
};


int main(int argc, char *argv[])
{
    int random_seed = get_random_seed(argc, argv);
    constexpr const char* algorithm = "SimulatedAnnealing";
    constexpr double time = 6000000.;
    constexpr bool test_complete = false;
    printf("Random seed set to %d%s", random_seed, std::string(2, '\n').c_str());
    
    if (test_complete)
    {
        Board x(InPaintWildfireSpreadTS, 0, true, std::vector<int>{1}, "postfix", 0, "LevenbergMarquardt", 1, "naive_numerical", Eigen::MatrixXd{{1, 2}, {3, 4}});
        x.pieces.resize(1);
        
        x.expression_type = "postfix";
        x.pieces[0] = {"x30", "x24", "s", "*", "*", "x30", "tau", "+", "/", "1", "1", "f", "~", "exp", "-", "/", "1", "1", "1", "f", "~", "exp", "-", "/", "-", "*", "*", "x28", "∂f/∂(x100)", "*", "x29", "∂f/∂(x101)", "*", "+", "*"};
        std::cout << "test before = " << x.pieces[0] << '\n';
        std::cout << "test depth before = " << x.getRPNdepth(x.pieces[0], 0).first << '\n';
        x.pieces[0] = x.complete_tree(x.pieces[0], 0);
        std::cout << "test after = " << x.pieces[0] << '\n';
        std::cout << "test depth after = " << x.getRPNdepth(x.pieces[0], 0).first << '\n';

        puts("");puts("");
        
        x.expression_type = "prefix";
        x.pieces[0] = {"*", "*", "/", "*", "x30", "*", "x24", "s", "+", "x30", "tau", "*", "/", "1", "-", "1", "exp", "~", "f", "-", "1", "/", "1", "-", "1", "exp", "~", "f", "+", "*", "x28", "∂f/∂(x100)", "*", "x29", "∂f/∂(x101)"};
        std::cout << "test before = " << x.pieces[0] << '\n';
        std::cout << "test depth before = " << x.getPNdepth(x.pieces[0], 0).first << '\n';
        x.pieces[0] = x.complete_tree(x.pieces[0], 0);
        std::cout << "test after = " << x.pieces[0] << '\n';
        std::cout << "test depth after = " << x.getPNdepth(x.pieces[0], 0).first << '\n';

        
        puts("");puts("");
        
        x.expression_type = "postfix";
        x.pieces[0] = {"10.28319", "0.010000", "x0", "+", "^", "2.714063472005533e-13", "*", "4.692820413780688e-06", "x1", "~", "*", "0.7049172460634555", "+", "+", "0.999329299739067", "0.010000", "x1", "+", "sin", "*", "0.9989466681769272", "x0", "sin", "*", "*", "-", "6.28319", "1", "x1", "+", "+", "-10.85907152907618", "/", "0.019659199307306502", "*", "x0", "x0", "2", "+", "/", "x0", "0.010000", "+", "6.333189999999999", "+", "^", "0.09367884443582758", "+", "+", "-"};
        std::cout << "test before = " << x.pieces[0] << '\n';
        std::cout << "test depth before = " << x.getRPNdepth(x.pieces[0], 0).first << '\n';
        x.pieces = x.complete_tree(x.pieces);
        std::cout << "test after = " << x.pieces[0] << '\n';
        std::cout << "test depth after = " << x.getRPNdepth(x.pieces[0], 0).first << '\n';
        exit(1);
    }
    
    ProblemOption choice = ProblemOption::WierdTrackFitter;
    switch (choice)
    {
        case ProblemOption::SwiftHohenberg:
            ExampleProblems::SwiftHohenbergTest(random_seed, algorithm, time);
            break;
        case ProblemOption::WildfireSpreadTS:
            ExampleProblems::WildfireSpreadTSTest(random_seed, algorithm, time);
            break;
        case ProblemOption::InPaintWildfireSpreadTS:
            ExampleProblems::InPaintWildfireSpreadTSTest(random_seed, algorithm, time);
            break;
        case ProblemOption::VortexRadialProfile:
            ExampleProblems::VortexRadialProfileTest(random_seed, algorithm, time);
            break;
        case ProblemOption::SolitonWaveFengEq14and15Laser:
            ExampleProblems::SolitonWaveFengEq14and15LaserTest(random_seed, algorithm, time);
            break;
        case ProblemOption::WierdTrackFitter:
            ExampleProblems::WierdTrackFitterTest(random_seed, algorithm, time);
            break;
        default:
            break;
    }
    
    return 0;
}

//git push --set-upstream origin PrefixPostfixSymbolicDifferentiator
/*

Case 1:
    g++ -Wall -std=c++20 -o PrefixPostfixMultiThreadDiffSimplifySR_Nd_double PrefixPostfixMultiThreadDiffSimplifySR_Nd_double.cpp -O2 -I/opt/homebrew/opt/eigen/include/eigen3 -I/opt/homebrew/opt/eigen/include/eigen3 -I/Users/edwardfinkelstein/LBFGSpp -L/opt/homebrew/Cellar/boost/1.84.0 -I/opt/homebrew/Cellar/boost/1.84.0/include -march=native
    g++ -Wall -std=c++20 -o PrefixPostfixMultiThreadDiffSimplifySR_Nd_double PrefixPostfixMultiThreadDiffSimplifySR_Nd_double.cpp -g -I/opt/homebrew/opt/eigen/include/eigen3 -I/opt/homebrew/opt/eigen/include/eigen3 -I/Users/edwardfinkelstein/LBFGSpp -L/opt/homebrew/Cellar/boost/1.84.0 -I/opt/homebrew/Cellar/boost/1.84.0/include -march=native

Case 2:
    Compile: Make sure you did $env:Path += ";C:\msys64\ucrt64\bin\" by inspecting $env:Path, then do the two lines below
    g++.exe -O2 -std=c++1z -IC:\Users\finkelsteine\test_codes\LBFGSpp\include -IC:\Users\finkelsteine\test_codes\boost_1_88_0 -IC:\Users\finkelsteine\test_codes\eigen\unsupported -IC:\Users\finkelsteine\test_codes\eigen\ -c C:\Users\finkelsteine\test_codes\hello_with_numbers_double.cpp -o C:\Users\finkelsteine\test_codes\hello_with_numbers_double.o -Wall
    g++.exe  -o C:\Users\finkelsteine\test_codes\hello_with_numbers_double.exe C:\Users\finkelsteine\test_codes\hello_with_numbers_double.o  -O2


    To run this file in Windows PowerShell, MAKE SURE ";C:\msys64\ucrt64\bin\" is in $env:Path
    (by doing $env:Path, and, if it's not there, do $env:Path += ";C:\msys64\ucrt64\bin\"),
    then do:
    & 'C:\Program Files (x86)\CodeBlocks\cb_console_runner.exe' .\hello_with_numbers_double.exe
    or
    .\hello_with_numbers_double.exe

    To unzip file: Expand-Archive -Path "C:\Users\finkelsteine\test_codes\boost_1_88_0.zip" -DestinationPath "C:\Users\finkelsteine\test_codes"
    To count how many instances of a string (in this case "stof" occur in a file (in this case `hello.cpp`):
     - (Get-Content -Path "C:\Users\finkelsteine\test_codes\hello.cpp" | Select-String -Pattern "stof").Count
    To launch Python: C:\Users\finkelsteine\AppData\Local\Programs\Python\Launcher\py.exe
    To get the diff between two files: Compare-Object (Get-Content -Path "C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp") (Get-Content -Path "C:\Users\finkelsteine\test_codes\hello_with_numbers.txt")
    To change the path variable, do $env:Path="newpath"
    To install with pip: C:\Users\finkelsteine\AppData\Local\Programs\Python\Launcher\py.exe -m pip install plotdigitizer

Case 3:
    g++ -Wall -std=c++20 -o PrefixPostfixMultiThreadDiffSimplifySR_Nd_double PrefixPostfixMultiThreadDiffSimplifySR_Nd_double.cpp -O2 -I/opt/homebrew/opt/eigen/include/eigen3 -I/opt/homebrew/opt/eigen/include/eigen3 -I/Users/edwardfinkelstein/LBFGSpp -L/opt/homebrew/Cellar/boost/1.84.0 -I/opt/homebrew/Cellar/boost/1.84.0/include -march=native
    g++ -Wall -std=c++20 -o PrefixPostfixMultiThreadDiffSimplifySR_Nd_double PrefixPostfixMultiThreadDiffSimplifySR_Nd_double.cpp -g -I/opt/homebrew/opt/eigen/include/eigen3 -I/opt/homebrew/opt/eigen/include/eigen3 -I/Users/edwardfinkelstein/LBFGSpp -L/opt/homebrew/Cellar/boost/1.84.0 -I/opt/homebrew/Cellar/boost/1.84.0/include -march=native

 
g++ -Wall -std=c++20 -o PrefixPostfixMultiThreadDiffSimplifySR_Nd_double PrefixPostfixMultiThreadDiffSimplifySR_Nd_double.cpp -O2 -I/usr/include/eigen3 -I./LBFGSpp/include -I./boost/1.84.0/include -march=native
 
*/


//a half built garden


//g++ -Wall -std=c++20 -o PrefixPostfixMultiThreadDiffSimplifySR_Nd_double PrefixPostfixMultiThreadDiffSimplifySR_Nd_double.cpp -O2 -I/usr/include/eigen3 -I./LBFGSpp/include -I./boost/1.84.0/include -march=native

