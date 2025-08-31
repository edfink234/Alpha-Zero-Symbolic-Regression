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
#include <cfloat>
#include <cassert>
#include <thread>
#include <mutex>
#include <atomic>
//#include <latch>
#include <tuple>
#include <functional>
//#include <numbers>
#include <LBFGS.h>
#include <LBFGSB.h>

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/AutoDiff>
#include <boost/unordered/concurrent_flat_map.hpp>

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
float Stof(const std::string& param)
{
    try
    {
        float val = std::stof(param);
        return val;
    }
    catch (const std::out_of_range&)
    {
        if (!param.empty() && param[0] == '-')
        {
            return -std::numeric_limits<float>::infinity();
        }
        else
        {
            return std::numeric_limits<float>::infinity();
        }
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
bool isFloat(const std::string& s)
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

// Function to create a matrix with linspace columns. std::vector<float> min and
// std::vector<float> max must have size == cols
Eigen::MatrixXf createLinspaceMatrix(int rows, int cols, std::vector<float> min_vec, std::vector<float> max_vec)
{
    assert((cols == static_cast<int>(min_vec.size())) && (cols == static_cast<int>(max_vec.size())));
    Eigen::MatrixXf mat(rows, cols);
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
std::vector<float> linspace(float min_val, float max_val, int num_points)
{
    std::vector<float> linspaced(num_points);
    float step = (max_val - min_val) / (num_points - 1);
    for (int i = 0; i < num_points; ++i)
    {
        linspaced[i] = min_val + i * step;
    }
    return linspaced;
}

Eigen::MatrixXf createMeshgridWithLambda(int rows, int cols, std::vector<float> min_vec, std::vector<float> max_vec, const std::function<float(const Eigen::RowVectorXf&)>& lambda)
{
    assert((cols == static_cast<int>(min_vec.size())) && (cols == static_cast<int>(max_vec.size())));

    // Create linspaces for each variable (column)
    std::vector<std::vector<float>> linspaces;
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
    Eigen::MatrixXf matrix(total_combinations, cols + 1);

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


Eigen::MatrixXf addColumnWithLambda(const Eigen::MatrixXf& matrix, const std::function<float(const Eigen::RowVectorXf&)>& lambda) {
    // Get the number of rows and columns of the input matrix
    int rows = matrix.rows();
    int cols = matrix.cols();

    // Create a new matrix with an additional column
    Eigen::MatrixXf newMatrix(rows, cols + 1);

    // Copy the original matrix into the new matrix (without the last column)
    newMatrix.block(0, 0, rows, cols) = matrix;

    // Apply the lambda function to each row and store the result in the last column
    for (int i = 0; i < rows; ++i) {
        newMatrix(i, cols) = lambda(matrix.row(i));
    }

    return newMatrix;
}

Eigen::MatrixXf createMeshgridVectors(int rows, int cols, std::vector<float> min_vec, std::vector<float> max_vec)
{
    assert((cols == static_cast<int>(min_vec.size())) && (cols == static_cast<int>(max_vec.size())));

    // Create linspaces for each variable (column)
    std::vector<std::vector<float>> linspaces;
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
    Eigen::MatrixXf matrix(total_combinations, cols);

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

Eigen::MatrixXf hstack(const Eigen::MatrixXf& mat1, const Eigen::MatrixXf& mat2)
{
    Eigen::MatrixXf result(mat1.rows(), mat1.cols() + mat2.cols());
    result << mat1, mat2; // Concatenate horizontally
    return result;
}

int trueMod(int N, int M)
{
    return ((N % M) + M) % M;
};

bool isInvalid(float x)
{
    return (std::isnan(x) || std::isinf(x));
}

float Variance(const Eigen::VectorXf& vec)
{
    return (vec.array() - vec.mean()).square().sum() / vec.size();
}

Eigen::VectorXf VarianceVec(const Eigen::VectorXf& vec)
{
    return (vec.array() - vec.mean()).square() / vec.size();
}

std::vector<Eigen::VectorXf> Variance(const std::vector<Eigen::VectorXf>& vec)
{
    size_t sz = vec.size();
    std::vector<Eigen::VectorXf> temp(sz);

    for (size_t idx = 0; idx < sz; idx++)
    {
        temp[idx] = VarianceVec(vec[idx]);
    }
    return temp;
}

float VarianceSum(const std::vector<Eigen::VectorXf>& vec)
{
    size_t sz = vec.size();
    float temp = 0.0f;

    for (size_t idx = 0; idx < sz; idx++)
    {
        temp += Variance(vec[idx]);
    }
    return temp;
}

/*
||=== Build file: "no target" in "no project" (compiler: unknown) ===|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In function 'bool isZero(const Eigen::VectorXf&, float)':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|415|warning: comparison of integer expressions of different signedness: 'size_t' {aka 'long long unsigned int'} and 'Eigen::EigenBase<Eigen::Matrix<float, -1, 1> >::Index' {aka 'long long int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In function 'bool isZero(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::Matrix<float, -1, 1> >, -1>&, float)':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|431|warning: comparison of integer expressions of different signedness: 'size_t' {aka 'long long unsigned int'} and 'Eigen::EigenBase<Eigen::Matrix<Eigen::AutoDiffScalar<Eigen::Matrix<float, -1, 1> >, -1, 1, 0, -1, 1> >::Index' {aka 'long long int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In function 'bool isConstant(const Eigen::VectorXf&, float)':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|451|warning: comparison of integer expressions of different signedness: 'size_t' {aka 'long long unsigned int'} and 'Eigen::EigenBase<Eigen::Matrix<float, -1, 1> >::Index' {aka 'long long int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In function 'bool isConstant(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::Matrix<float, -1, 1> >, -1>&, float)':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|467|warning: comparison of integer expressions of different signedness: 'size_t' {aka 'long long unsigned int'} and 'Eigen::EigenBase<Eigen::Matrix<Eigen::AutoDiffScalar<Eigen::Matrix<float, -1, 1> >, -1, 1, 0, -1, 1> >::Index' {aka 'long long int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In constructor 'Board::Board(std::vector<std::vector<std::__cxx11::basic_string<char> > > (*)(Board&), size_t, bool, const std::vector<int>&, const std::string&, size_t, std::string, int, std::string, const Eigen::MatrixXf&, bool, bool, bool, float, bool, bool, int)':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|725|warning: comparison of integer expressions of different signedness: 'int' and 'size_t' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In member function 'std::string Board::print_expression_params()':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|861|warning: comparison of integer expressions of different signedness: 'size_t' {aka 'long long unsigned int'} and 'Eigen::EigenBase<Eigen::Matrix<float, -1, 1> >::Index' {aka 'long long int'} [-Wsign-compare]|
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
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In member function 'float Board::operator()(Eigen::VectorXf&, Eigen::VectorXf&)':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|3487|warning: comparison of integer expressions of different signedness: 'size_t' {aka 'long long unsigned int'} and 'Eigen::EigenBase<Eigen::Matrix<float, -1, 1> >::Index' {aka 'long long int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|3488|warning: comparison of integer expressions of different signedness: 'size_t' {aka 'long long unsigned int'} and 'int' [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|3489|warning: comparison of integer expressions of different signedness: 'Eigen::EigenBase<Eigen::Matrix<float, -1, 1> >::Index' {aka 'long long int'} and 'size_t' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In member function 'float Board::fitFunctionToData()':|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|3657|warning: comparison of integer expressions of different signedness: 'int' and 'std::vector<std::vector<std::__cxx11::basic_string<char> > >::size_type' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|3734|warning: comparison of integer expressions of different signedness: 'int' and 'std::vector<std::vector<std::__cxx11::basic_string<char> > >::size_type' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|3820|warning: comparison of integer expressions of different signedness: 'int' and 'std::vector<std::vector<std::__cxx11::basic_string<char> > >::size_type' {aka 'long long unsigned int'} [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp||In member function 'float Board::complete_status(int, bool)':|
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
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|7686|warning: comparison of integer expressions of different signedness: 'boost::unordered::concurrent_flat_map<std::__cxx11::basic_string<char>, Eigen::Matrix<float, -1, 1> >::size_type' {aka 'long long unsigned int'} and 'int' [-Wsign-compare]|
C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp|7714|warning: comparison of integer expressions of different signedness: 'int' and 'std::vector<std::vector<std::__cxx11::basic_string<char> > >::size_type' {aka 'long long unsigned int'} [-Wsign-compare]|
||=== Build finished: 0 error(s), 36 warning(s) (0 minute(s), 38 second(s)) ===|

*/

template<typename Derived>
typename Derived::Scalar median( Eigen::DenseBase<Derived>& d )
{
    auto r {d.reshaped()};
    std::sort(r.begin(), r.end());
    return (r.size() % 2 == 0) ?
            r.segment((r.size()-2)/2, 2).mean() :
            r(r.size()/2);
}

template<typename Derived>
typename Derived::Scalar median(const Eigen::DenseBase<Derived>& d)
{
    typename Derived::PlainObject m {d.replicate(1,1)};
    return median(m);
}

bool isZero(const Eigen::VectorXf& vec, float tolerance = 1e-5f)
{
    if (vec.size() <= 1)
    {
        return true; // A vector with 0 or 1 element is trivially constant
    }
    if (vec.array().isNaN().any() || vec.array().isInf().any())
    {
        return true; // Return true if any NaN is present so it'll be weeded out
    }
    for (size_t i = 0; i < vec.size(); ++i)
    {
        if (isInvalid(vec[i]))
        {
            return true; // Return true if any NaN is present in values
        }
    }
    return ((vec.array().abs().maxCoeff() <= tolerance) && (median(vec) <= tolerance));
}

bool isZero(const Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>& vec, float tolerance = 1e-5f)
{
    if (vec.size() <= 1)
    {
        return true; // A vector with 0 or 1 element is trivially constant
    }
    for (size_t i = 0; i < vec.size(); ++i)
    {
        if (isInvalid(vec[i].value()))
        {
            return true; // Return true if any NaN is present in values
        }
    }
    return ((vec.array().abs().maxCoeff() <= tolerance) && (median(vec) <= tolerance));
}

bool isConstant(const Eigen::VectorXf& vec, float tolerance = 1e-5f)
{
    if (vec.size() <= 1)
    {
        return true; // A vector with 0 or 1 element is trivially constant
    }
    if (vec.array().isNaN().any() || vec.array().isInf().any())
    {
        return true; // Return true if any NaN is present so it'll be weeded out
    }
    for (size_t i = 0; i < vec.size(); ++i)
    {
        if (isInvalid(vec[i]))
        {
            return true; // Return true if any NaN is present in values
        }
    }
    return (Variance(vec) <= tolerance);
}

bool isConstant(const Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>& vec, float tolerance = 1e-5f)
{
    if (vec.size() <= 1)
    {
        return true; // A vector with 0 or 1 element is trivially constant
    }
    for (size_t i = 0; i < vec.size(); ++i)
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
    Eigen::MatrixXf data;
    std::unordered_map<std::string, Eigen::VectorXf> features;
    std::vector<Eigen::VectorXf> rows;
    long num_columns, num_rows;

public:

    Data() = default; //so we can have a static Data attribute

    // Assignment operator
    Data& operator=(const Eigen::MatrixXf& theData)
    {
        this->data = theData;
        this->num_columns = data.cols();
        this->num_rows = data.rows();
        for (long i = 0; i < this->num_columns; i++) //for each column
        {
            this->features["x"+std::to_string(i)] = Eigen::VectorXf(this->num_rows);
            for (long j = 0; j < this->num_rows; j++)
            {
                this->features["x"+std::to_string(i)](j) = this->data(j,i);
            }
        }
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

    const Eigen::VectorXf& operator[] (int i){return rows[i];}
    const Eigen::VectorXf& operator[] (const std::string& i)
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

float MSE(const Eigen::VectorXf& actual)
{
    return actual.squaredNorm();
}

float MSE(const std::vector<Eigen::VectorXf>& actual)
{
    float temp = 0.0f;
    for (size_t i = 0; i < actual.size(); i++)
    {
        temp += actual[i].squaredNorm();
    }

    return temp;
}

float MSE(const std::vector<Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>>& actual)
{
    float temp = 0.0f;
    size_t count = 0;

    for (const auto& vec : actual)
    {
        for (int i = 0; i < vec.size(); ++i)
        {
            // Access the value of the AutoDiffScalar element
            temp += vec[i].value() * vec[i].value();
        }
        ++count;
    }

    return count > 0 ? temp / count : FLT_MAX;
}

float MSE(const Eigen::VectorXf& actual, const Eigen::VectorXf& predicted)
{
    if (actual.size() != predicted.size())
    {
        throw std::invalid_argument("Vectors must be of the same size");
    }
    return (actual - predicted).squaredNorm();
}

Eigen::AutoDiffScalar<Eigen::VectorXf> MSE(const Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>& actual)
{
    return actual.squaredNorm();
}

float loss_func(const Eigen::VectorXf& actual)
{
    return (1.0f/(1.0f+MSE(actual)));
}

float loss_func(const std::vector<Eigen::VectorXf>& actual)
{
    float mse = 0.0f;
    for (size_t i = 0; i < actual.size(); i++)
    {
        mse += MSE(actual[i]);
    }
    return (1.0f/(1.0f+mse));
}

float loss_func(const Eigen::VectorXf& actual, const Eigen::VectorXf& predicted)
{
    return (1.0f/(1.0f+MSE(actual, predicted)));
}

//performs the transformation "const{>=num_consts_diff}" -> "const"
//on each such token in pieces
void reset_const_token_labels(std::vector<std::vector<std::string>>& pieces, size_t num_consts_diff)
{
    for (std::vector<std::string>& x_expr: pieces)
    {
        for (std::string& token: x_expr)
        {
            if (token.compare(0, 5, "const") == 0)
            {
                std::string int_suffix = token.substr(5);
                if (int_suffix.size())
                {
                    int int_suffix_num = std::stoi(int_suffix);
                    if (int_suffix_num >= num_consts_diff) //then it's a const that belongs to pieces -> reset it
                    {
                        token = "const";
                    }
                    else
                    {
                        //Below we test that token is of the form `const{0 <= num < num_consts_diff}`
                        assert(((0 <= int_suffix_num) && (int_suffix_num < num_consts_diff)));
                    }
                }
            }
        }
    }
};

struct Board
{
    static boost::concurrent_flat_map<std::string, Eigen::VectorXf> inline expression_dict;
    static constexpr size_t max_expression_dict_sz = 100000000; //one-hundred million
    static std::atomic<float> inline fit_time = 0.0;

    static constexpr float K = 0.0884956f;
    static constexpr float phi_1 = 2.8f;
    static constexpr float phi_2 = 1.3f;
    static int inline __num_features;
    static std::vector<std::string> inline __input_vars;
    static std::vector<std::string> inline __unary_operators;
    static std::vector<std::string> inline __binary_operators;
    static size_t inline num_unary_ops;
    static size_t inline num_binary_ops;
    static size_t inline num_leaf_operands;
    static std::unordered_set<std::string> inline __unary_operators_uset;
    static std::unordered_set<std::string> inline __binary_operators_uset;
    static std::vector<std::string> inline __operators;
    static std::vector<std::string> inline __other_tokens;
    static std::vector<std::string> inline __tokens;
    Eigen::VectorXf params; //store the parameters of the expression of the current episode after it's completed
    static Data inline data;

    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<float> vel_dist, pos_dist;
    static std::uniform_int_distribution<int> inline unary_dist; // A random integer generator which generates an index corresponding to a unary operator
    static std::uniform_int_distribution<int> inline binary_dist; // A random integer generator which generates an index corresponding to a binary operator
    static std::uniform_int_distribution<int> inline leaf_dist; // A random integer generator which generates an index corresponding to an operand

    static int inline action_size;
    static std::once_flag inline initialization_flag;  // Flag for std::call_once
    static std::unordered_map<std::string, std::pair<std::string, std::string>> inline feature_mins_maxes;

    size_t reserve_amount;
    int num_fit_iter;
    int num_objectives;
    float MSE_curr;
    std::string fit_method;
    std::string fit_grad_method;

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
    //TODO: MAYBE SEE IF YOU CAN CHANGE `std::string` to `const char*` here?
    std::vector<std::vector<std::string>> pieces, temp_pieces; // Create the empty expression list and a backup
    std::vector<std::string> derivat;// Vector to store the derivative.
    bool visualize_exploration, is_primary;
    std::vector<std::vector<std::string>> (*diffeq)(Board&); //differential equation we want to solve
    size_t num_diff_eqns; //number of equations in the system `diffeq`
    std::vector<std::vector<std::string>> diffeq_result;
    float isConstTol;
    bool simplify_original;

    Board(std::vector<std::vector<std::string>> (*diffeq)(Board&), size_t num_diff_eqns, bool primary = true, const std::vector<int>& depth = {},
          const std::string& expression_type = "prefix", size_t num_consts_diff = 0, std::string fitMethod = "LevenbergMarquardt", int numFitIter = 1,
          std::string fitGradMethod = "naive_numerical", const Eigen::MatrixXf& theData = {}, bool visualize_exploration = false, bool cache = false,
          bool const_tokens = false, float isConstTol = 1e-1f, bool use_const_pieces = false, bool simplifyOriginal = true,
          int numDataCols = 0) :
        gen{rd()}, vel_dist{-1.0f, 1.0f}, pos_dist{0.0f, 1.0f}, num_fit_iter{numFitIter}, fit_method{fitMethod}, fit_grad_method{fitGradMethod}, n{depth},
        is_primary{primary}, simplify_original{simplifyOriginal}
    {
        assert(n.size());
        this->num_objectives = n.size();
        size_t max_n = n[0];
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
        printf("this->diffeq_result = %lu\n", diffeq_result.size());
        this->num_diff_eqns = num_diff_eqns;
        this->isConstTol = isConstTol;

        if (is_primary)
        {
            std::call_once(initialization_flag, [&]()
            {
                Board::data = theData;
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
                if (this->use_const_pieces) //first add "const" if requested
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
                Board::action_size = Board::__tokens.size();

                Board::una_bin_leaf_legal_moves_dict.clear();
                if (const_tokens)
                {
                    //Then include fixed constants, the non-differential equation optimizable constant `const` (if use_const_pieces == true),
                    //AND `this->num_consts_diff` optimizable constant tokens (const0, const1, ..., const{this->num_consts_diff}) that can be optimized.
                    Board::una_bin_leaf_legal_moves_dict[true][true][true] = Board::__tokens;
                }

                else
                {
                    Board::una_bin_leaf_legal_moves_dict[true][true][true] = Board::__operators;
                }

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
                    if (!const_tokens) //without constants (optimizable or not), the only leaf nodes will be the input variables
                    {
                        Board::una_bin_leaf_legal_moves_dict[true][true][true].push_back(i);
                    }
                }
                if (const_tokens)
                {
                    for (const std::string &i: Board::__other_tokens)
                    {
                        Board::una_bin_leaf_legal_moves_dict[true][false][true].push_back(i); //1
                        Board::una_bin_leaf_legal_moves_dict[false][true][true].push_back(i); //2
                        Board::una_bin_leaf_legal_moves_dict[false][false][true].push_back(i); //3
                    }
                }
                Board::num_unary_ops = Board::__unary_operators.size();
                Board::num_binary_ops = Board::__binary_operators.size();
                Board::num_leaf_operands = Board::una_bin_leaf_legal_moves_dict[false][false][true].size();
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
        for (size_t i = this->num_consts_diff; i < this->params.size(); i++)
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
        if (index < Board::__tokens.size())
        {
            return Board::__tokens[index];
        }
        throw std::out_of_range("Index out of range");
    }

    int __num_binary_ops(int i) const
    {
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

            if (new_expression[first_arg_idx_high] == "0") //+/- x 0 -> x
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

            else if (new_expression[first_arg_idx_low] == "0")
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
            if (new_expression[first_arg_idx_high] == "0") //* x 0 -> 0 (because, since prefix operators come at the beginning, if the beginning of the second argument of '*' is 0, then the whole second argument MUST be 0, therefore the expression reduces to * x 0, which is 0)
            {
                //puts("hi 239");
                new_expression[op_idx] = "0"; //change '*' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "0") //* 0 x -> 0
            {
                //puts("hi 245");
                new_expression[op_idx] = "0"; //change '*' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_high] == "1") //* x 1 -> x (because, since prefix operators come at the beginning, if the beginning of the second argument of '*' is 1, then the whole second argument MUST be 1, therefore the expression reduces to * x 1, which is 1)
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
            else if (new_expression[first_arg_idx_low] == "1") //* 1 x -> x
            {
                //puts("hi 265");
                new_expression.erase(new_expression.begin() + op_idx, new_expression.begin() + op_idx + 2); //erase the '*' and the '1'
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
            //TODO: There's an issue with how / is being handled here..., if the left or right sub-tree (represented by the symbol `x`) hasn't been simplified; it might be 0, so there's a possibility that the result of / x 0 could actually be nan as well, same goes for / 0 x
            if ((new_expression[first_arg_idx_low] == "0") && (new_expression[first_arg_idx_high] == "0")) // / 0 0 -> nan
            {
                //puts("hi 290");
                new_expression[op_idx] = "nan"; //change '/' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_high] == "0") // / x 0 -> inf (because, since prefix operators come at the beginning, if the beginning of the second argument of '/' is 0, then the whole second argument MUST be 0, therefore the expression reduces to / x 0, which is 0)
            {
                //puts("hi 282");
                new_expression[op_idx] = (new_expression[first_arg_idx_low] != "~") ? "inf": "-inf"; //change '/' to 'inf' or '-inf'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "0") // / 0 x -> 0
            {
                //puts("hi 295");
                new_expression[op_idx] = "0"; //change '/' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_high] == "1") // / x 1 -> x (because, since prefix operators come at the beginning, if the beginning of the second argument of '/' is 1, then the whole second argument MUST be 1, therefore the expression reduces to / x 1, which is 1)
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
            new_expression.push_back(expression[low]); // /
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // / x
            int first_arg_idx_high = new_expression.size();
            graspSimplifyPrefixHelper(expression, temp+1, temp+1+grasp[temp+1], grasp, new_expression, true); // / x y
            //int second_arg_idx_high = new_expression.size();
            //int step;
            if (new_expression[first_arg_idx_high] == "0") //^ x 0 -> 1 (because, since prefix operators come at the beginning, if the beginning of the second argument of '^' is 0, then the whole second argument MUST be 0, therefore the expression reduces to ^ x 0, which is 1)
            {
                //puts("hi 334");
                new_expression[op_idx] = "1"; //change '^' to '1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "0") // ^ 0 x -> 0 (x > 0 assumed)
            {
                //puts("hi 340");
                new_expression[op_idx] = "0"; //change '^' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_high] == "1") // ^ x 1 -> x (because, since prefix operators come at the beginning, if the beginning of the second argument of '^' is 1, then the whole second argument MUST be 1, therefore the expression reduces to ^ x 1, which is 1)
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
            else if (new_expression[first_arg_idx_low] == "1") // ^ 1 x -> 1
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
            if (new_expression[first_arg_idx_low] == "0") // cos 0 -> 1
            {
                //puts("hi 374");
                new_expression[op_idx] = "1"; //change 'cos' to '1'
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
            if (new_expression[first_arg_idx_low] == "0") // sin 0 -> 0
            {
                //puts("hi 388");
                new_expression[op_idx] = "0"; //change 'sin' to '0'
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
            if (new_expression[first_arg_idx_low] == "0") // tanh 0 -> 0
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
            if (new_expression[first_arg_idx_low] == "0") // sech 0 -> 1
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
            if (new_expression[first_arg_idx_low] == "0") // ~ 0 -> 0
            {
    //            puts("hi 466");
                new_expression[op_idx] = "0"; //change '~' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            //TODO: Uncomment and test this!
    //        else if (new_expression[first_arg_idx_low] == "inf") // ~ inf -> -inf
    //        {
    //            //puts("hi 487");
    //            new_expression[op_idx] = "-inf"; //change '~' to '-inf'
    //            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
    //        }
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

    void simplifyPN_Helper(std::vector<std::string>& expression)
    {
        bool simplified = true;
        bool isFloat1, isFloat2, isConst1, isConst2;
        while (simplified)
        {
            simplified = false;
            if (expression.size() > 1)
            {
                for (size_t i = 0; i < expression.size() - 1; i++)
                {
                    if (is_binary(expression[i]))
                    {
                        isFloat1 = isFloat(expression[i+1]);
                        isFloat2 = isFloat(expression[i+2]);

                        if (isFloat1 && isFloat2)
                        {
                            if (expression[i] == "+")
                            {
                                expression[i] = simplifyString(std::to_string(Stof(expression[i+1]) + Stof(expression[i+2])));
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i] == "-")
                            {
                                expression[i] = simplifyString(std::to_string(Stof(expression[i+1]) - Stof(expression[i+2])));
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i] == "*")
                            {
                                expression[i] = simplifyString(std::to_string(Stof(expression[i+1]) * Stof(expression[i+2])));
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i] == "/")
                            {
                                expression[i] = simplifyString(std::to_string(Stof(expression[i+1]) / Stof(expression[i+2])));
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i] == "^")
                            {
                                expression[i] = simplifyString(std::to_string(std::powf(Stof(expression[i+1]), Stof(expression[i+2]))));
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                        }

                        isConst1 = is_const(expression[i+1]);
                        isConst2 = is_const(expression[i+2]);

                        if ((isConst1 && isConst2) && ((expression[i+1].find("nan") != std::string::npos) || (expression[i+2].find("nan") != std::string::npos))) //binary_op nan x = binary_op x nan = nan
                        {
                            //puts("hi 570");
                            expression[i] = "nan";
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "-")
                        {
                            if ((isConst1 && isConst2) && (expression[i+1] == expression[i+2])) //- x x => 0
                            {
                                expression[i] = "0";
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i+1] == "0") //- 0 x -> ~ x
                            {
                                expression[i] = "~";
                                expression.erase(expression.begin() + i + 1);
                                simplified = true;
                                break;
                            }
                            else if (expression[i+2] == "0" && isConst1) //- x 0 -> x
                            {
                                expression[i] = expression[i+1];
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                        }
                        else if (expression[i] == "*")
                        {
                            if (expression[i+1] == "0" && isConst2) //* 0 x -> 0
                            {
                                //puts("hi 131");
                                expression[i] = "0";
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i+2] == "0" && isConst1) //* x 0 -> 0
                            {
                                //puts("hi 139");
                                expression[i] = "0";
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i+1] == "1" && isConst2) //* 1 x -> x
                            {
                                //puts("hi 147");
                                expression[i] = expression[i+2];
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i+2] == "1" && isConst1) //* x 1 -> x
                            {
                                //puts("hi 155");
                                expression[i] = expression[i+1];
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                        }
                        else if (expression[i] == "+")
                        {
                            if (expression[i+1] == "0" && isConst2) //+ 0 x -> x
                            {
                                //puts("hi 167");
                                expression[i] = expression[i+2];
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i+2] == "0" && isConst1) //+ x 0 -> x
                            {
                                //puts("hi 175");
                                expression[i] = expression[i+1];
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                        }
                        else if (expression[i] == "/")
                        {
                            if (expression[i+1] == "0" && isConst2) // / 0 x -> 0
                            {
                                //puts("hi 187");
                                expression[i] = "0";
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i+2] == "1" && isConst1) // / x 1 -> x
                            {
                                //puts("hi 195");
                                expression[i] = expression[i+1];
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (isConst1 && isConst2 && (expression[i+1] == expression[i+2])) // / x x -> 1
                            {
                                //puts("hi 203");
                                expression[i] = "1";
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                        }
                        else if (expression[i] == "^")
                        {
                            if (expression[i+2] == "0" && isConst1) // ^ x 0 -> 1
                            {
                                //puts("hi 223");
                                expression[i] = "1";
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i+1] == "0" && isConst2) // ^ 0 x -> 0 (x > 0)
                            {
                                //puts("hi 215");
                                expression[i] = "0";
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i+1] == "1" && isConst2) // ^ 1 x -> 1
                            {
                                //puts("hi 231");
                                expression[i] = "1";
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i+2] == "1" && isConst1) // ^ x 1 -> x
                            {
                                //puts("hi 239");
                                expression[i] = expression[i+1];
                                expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                                simplified = true;
                                break;
                            }
                        }
                    }

                    else if (is_unary(expression[i]) && isFloat(expression[i+1]))
                    {
                        if (expression[i] == "cos")
                        {
                            expression[i] = simplifyString(std::to_string(cos(Stof(expression[i+1]))));
                            expression.erase(expression.begin() + i + 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "~")
                        {
                            expression[i] = simplifyString(std::to_string(-(Stof(expression[i+1]))));
                            expression.erase(expression.begin() + i + 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "sin")
                        {
                            expression[i] = simplifyString(std::to_string(sin(Stof(expression[i+1]))));
                            expression.erase(expression.begin() + i + 1);
                            simplified = true;
                            break;
                        }
                        else if ((expression[i] == "ln") || (expression[i] == "log"))
                        {
                            expression[i] = simplifyString(std::to_string(log(Stof(expression[i+1])))); // Natural log (ln)
                            expression.erase(expression.begin() + i + 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "asin" || expression[i] == "arcsin")
                        {
                            expression[i] = simplifyString(std::to_string(asin(Stof(expression[i+1]))));
                            expression.erase(expression.begin() + i + 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "acos" || expression[i] == "arccos")
                        {
                            expression[i] = simplifyString(std::to_string(acos(Stof(expression[i+1]))));
                            expression.erase(expression.begin() + i + 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "exp")
                        {
                            expression[i] = simplifyString(std::to_string(exp(Stof(expression[i+1]))));
                            expression.erase(expression.begin() + i + 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "sech")
                        {
                            expression[i] = simplifyString(std::to_string(1 / cosh(Stof(expression[i+1])))); // sech(x) = 1 / cosh(x)
                            expression.erase(expression.begin() + i + 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "tanh")
                        {
                            expression[i] = simplifyString(std::to_string(tanh(Stof(expression[i+1]))));
                            expression.erase(expression.begin() + i + 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "sqrt")
                        {
                            expression[i] = simplifyString(std::to_string(sqrt(Stof(expression[i+1]))));
                            expression.erase(expression.begin() + i + 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "abs")
                        {
                            expression[i] = simplifyString(std::to_string(abs(Stof(expression[i+1]))));
                            expression.erase(expression.begin() + i + 1);
                            simplified = true;
                            break;
                        }
                    }

                    else if (is_unary(expression[i]))
                    {
                        if (expression[i] == "~" && expression[i+1] == "~")
                        {
                            expression.erase(expression.begin() + i, expression.begin() + i + 2); // Remove elements at i and i + 1
                            simplified = true;
                            break;
                        }
                        //TODO: Add 0 ~ -> 0
                        //TODO: Add inf ~ -> -inf
                        else if (expression[i] == "exp" && (expression[i+1] == "ln" || expression[i+1] == "log"))
                        {
                            //puts("hi 361");
                            expression.erase(expression.begin() + i, expression.begin() + i + 2); // Remove elements at i and i + 1
                            simplified = true;
                            break;
                        }
                        else if (expression[i+1] == "exp" && (expression[i] == "ln" || expression[i] == "log"))
                        {
                            //puts("hi 369");
                            expression.erase(expression.begin() + i, expression.begin() + i + 2); // Remove elements at i and i + 1
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "cos" && (expression[i+1] == "acos" || expression[i+1] == "arccos"))
                        {
                            //puts("hi 403");
                            expression.erase(expression.begin() + i, expression.begin() + i + 2); // Remove elements at i and i + 1
                            simplified = true;
                            break;
                        }
                        else if ((expression[i] == "cos") && (expression[i+1] == "~")) //cos(-x) = cos(x)
                        {
                            //puts("hi 708");
                            expression.erase(expression.begin() + i + 1); // Remove the '~'
                            simplified = true;
                            break;
                        }
                        else if (expression[i+1] == "cos" && (expression[i] == "acos" || expression[i] == "arccos"))
                        {
                            //puts("hi 411");
                            expression.erase(expression.begin() + i, expression.begin() + i + 2); // Remove elements at i and i + 1
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "sin" && (expression[i+1] == "asin" || expression[i+1] == "arcsin"))
                        {
                            //puts("hi 419");
                            expression.erase(expression.begin() + i, expression.begin() + i + 2); // Remove elements at i and i + 1
                            simplified = true;
                            break;
                        }
                        else if (expression[i+1] == "sin" && (expression[i] == "asin" || expression[i] == "arcsin"))
                        {
                            //puts("hi 427");
                            expression.erase(expression.begin() + i, expression.begin() + i + 2); // Remove elements at i and i + 1
                            simplified = true;
                            break;
                        }
                    }
                }
            }
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

            if (new_expression.back() == "0") // x 0 +/- -> x
            {
                //puts("hi 181");
                new_expression.pop_back();
            }

            else if (new_expression[first_arg_idx_high - 1] == "0") // 0 x +/- -> x +/-
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

            if (new_expression.back() == "0") // x 0 * -> 0 (because, since postfix operators come at the end, if the end of the second argument of '*' is 0, then the whole second argument MUST be 0, therefore the expression reduces to x 0 *, which is 0)
            {
                //puts("hi 235");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (new_expression[first_arg_idx_high - 1] == "0") //0 x * -> 0 (because, since postfix operators come at the end, if the end of the first argument of '*' is 0, then the whole second argument MUST be 0, therefore the expression reduces to 0 x *, which is 0)
            {
                //puts("hi 241");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (new_expression.back() == "1") // x 1 * -> x (because, since postfix operators come at the end, if the end of the second argument of '*' is 1, then the whole second argument MUST be 1, therefore the expression reduces to x 1 *, which is x)
            {
                //puts("hi 247");
                new_expression.pop_back(); //erase the '1'
            }
            else if (new_expression[first_arg_idx_high - 1] == "1") //1 x * -> x (because, since postfix operators come at the end, if the end of the first argument of '*' is 1, then the whole first argument MUST be 1, therefore the expression reduces to 1 x *, which is x)
            {
                //puts("hi 252");
                new_expression.erase(new_expression.begin() + first_arg_idx_high - 1); //erase the '1'
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
            //TODO: There's an issue with how / is being handled here..., if the left or right sub-tree (represented by the symbol `x`) hasn't been simplified; it might be 0, so there's a possibility that the result of x 0 / could actually be nan as well, same goes for 0 x /
            if ((new_expression.back() == "0") && (new_expression[first_arg_idx_high - 1] == "0")) // 0 0 / -> nan
            {
                //puts("hi 279");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (new_expression.back() == "0") // x 0 / -> inf (because, since postfix operators come at the end, if the end of the second argument of '/' is 0, then the whole second argument MUST be 0, therefore the expression reduces to x 0 /, which is inf)
            {
                //puts("hi 280");
                new_expression[first_arg_idx_low] = (new_expression[first_arg_idx_high - 1] == "~") ? "-inf" : "inf";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (new_expression[first_arg_idx_high - 1] == "0") //0 x / -> 0 (because, since postfix operators come at the end, if the end of the first argument of '/' is 0, then the whole second argument MUST be 0, therefore the expression reduces to 0 x /, which is 0)
            {
                //puts("hi 286");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (new_expression.back() == "1") // x 1 / -> x (because, since postfix operators come at the end, if the end of the second argument of '/' is 1, then the whole second argument MUST be 1, therefore the expression reduces to x 1 /, which is x)
            {
                //puts("hi 292");
                new_expression.pop_back(); //erase the '1'
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

            if (new_expression.back() == "0") // x 0 ^ -> 1 (because, since postfix operators come at the end, if the end of the second argument of '^' is 0, then the whole second argument MUST be 0, therefore the expression reduces to x 0 ^, which is 1)
            {
                //puts("hi 318");
                new_expression[first_arg_idx_low] = "1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (new_expression[first_arg_idx_high - 1] == "0") //0 x ^ -> 0 (because, since postfix operators come at the end, if the end of the first argument of '^' is 0, then the whole second argument MUST be 0, therefore the expression reduces to 0 x ^, which is 0) (assume x > 0)
            {
                //puts("hi 324");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (new_expression.back() == "1") // x 1 ^ -> x (because, since postfix operators come at the end, if the end of the second argument of '^' is 1, then the whole second argument MUST be 1, therefore the expression reduces to x 1 ^, which is x)
            {
                //puts("hi 330");
                new_expression.pop_back(); //erase the '1'
            }
            else if (new_expression[first_arg_idx_high - 1] == "1") //1 x ^ -> 1 (because, since postfix operators come at the end, if the end of the first argument of '^' is 1, then the whole second argument MUST be 1, therefore the expression reduces to 1 x ^, which is 1)
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
            if (new_expression.back() == "0") // 0 cos -> 1 (because, since postfix operators come at the end, if the end of the argument of 'cos' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 cos, which is 1)
            {
                //puts("hi 350");
                new_expression[first_arg_idx_low] = "1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
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
            if (new_expression.back() == "0") // 0 sin -> 0 (because, since postfix operators come at the end, if the end of the argument of 'sin' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 sin, which is 0)
            {
                //puts("hi 365");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
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
            if (new_expression.back() == "0") // 0 tanh -> 0 (because, since postfix operators come at the end, if the end of the argument of 'tanh' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 tanh, which is 0)
            {
                //puts("hi 380");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (new_expression.back() == "inf") // inf tanh -> 1 (because, since postfix operators come at the end, if the end of the argument of 'tanh' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf tanh, which is 1)
            {
                //puts("hi 386");
                new_expression[first_arg_idx_low] = "1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (new_expression.back() == "-inf") // -inf tanh -> -1 (because, since postfix operators come at the end, if the end of the argument of 'tanh' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf tanh, which is -1)
            {
                //puts("hi 392");
                new_expression[first_arg_idx_low] = "-1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ tanh -> -1 (because, since postfix operators come at the end, if the end of the argument of 'tanh' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf tanh, which is -1)
            {
                //puts("hi 392");
                new_expression[first_arg_idx_low] = "-1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
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
            if (new_expression.back() == "0") // 0 sech -> 1 (because, since postfix operators come at the end, if the end of the argument of 'sech' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 sech, which is 1)
            {
                //puts("hi 395");
                new_expression[first_arg_idx_low] = "1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (new_expression.back() == "inf") // inf sech -> 0 (because, since postfix operators come at the end, if the end of the argument of 'sech' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf sech, which is 0)
            {
    //            puts("hi 419");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (new_expression.back() == "-inf") // -inf sech -> 0 (because, since postfix operators come at the end, if the end of the argument of 'sech' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf sech, which is 0)
            {
                //puts("hi 425");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ sech -> 0 (because, since postfix operators come at the end, if the end of the argument of 'sech' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf sech, which is 0)
            {
                //puts("hi 431");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
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
            if (new_expression.back() == "0") // 0 ~ -> 0 (because, since postfix operators come at the end, if the end of the argument of '~' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 ~, which is 0)
            {
    //            puts("hi 445");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            //TODO: Uncomment and test this!
    //        if (new_expression.back() == "inf") // inf ~ -> -inf (because, since postfix operators come at the end, if the end of the argument of '~' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf ~, which is -inf)
    //        {
    ////            puts("hi 445");
    //            new_expression[first_arg_idx_low] = "-inf";
    //            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
    //        }
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

    void simplifyRPN_Helper(std::vector<std::string>& expression)
    {
        bool simplified = true;
        bool isFloat1, isFloat2, isConst1, isConst2;
        while (simplified)
        {
            simplified = false;
            if (expression.size() > 1)
            {
                for (size_t i = 1; i < expression.size(); i++)
                {
                    if (is_binary(expression[i]))
                    {
                        isFloat1 = isFloat(expression[i-1]);
                        isFloat2 = isFloat(expression[i-2]);

                        if (isFloat1 && isFloat2)
                        {
                            if (expression[i] == "+")
                            {
                                expression[i] = simplifyString(std::to_string(Stof(expression[i-2]) + Stof(expression[i-1])));
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i] == "-")
                            {
                                expression[i] = simplifyString(std::to_string(Stof(expression[i-2]) - Stof(expression[i-1])));
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i] == "*")
                            {
                                expression[i] = simplifyString(std::to_string(Stof(expression[i-2]) * Stof(expression[i-1])));
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i] == "/")
                            {
                                expression[i] = simplifyString(std::to_string(Stof(expression[i-2]) / Stof(expression[i-1])));
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i] == "^")
                            {
                                expression[i] = simplifyString(std::to_string(std::powf(Stof(expression[i-2]), Stof(expression[i-1]))));
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                        }

                        isConst1 = is_const(expression[i-1]);
                        isConst2 = is_const(expression[i-2]);

                        if ((isConst1 && isConst2) && ((expression[i-1].find("nan") != std::string::npos) || (expression[i-2].find("nan") != std::string::npos))) //x nan binary_op = nan x binary_op = nan
                        {
                            //puts("hi 549");
                            expression[i] = "nan";
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "-")
                        {
                            if ((isConst1 && isConst2) && (expression[i-1] == expression[i-2])) //x x - => 0
                            {
                                expression[i] = "0";
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i-2] == "0" && isConst1) //"0 x -" -> "x ~"
                            {
                                expression[i] = "~";
                                expression.erase(expression.begin() + i - 2);
                                simplified = true;
                                break;
                            }
                            else if (expression[i-1] == "0") //"x 0 -" -> "x"
                            {
                                expression[i] = expression[i-2];
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                        }

                        else if (expression[i] == "*")
                        {
                            if (expression[i-2] == "0" && isConst1) //"0 x *" -> "0"
                            {
                                //puts("hi 131");
                                expression[i] = "0";
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i-1] == "0" && isConst2) //"x 0 *" -> "0"
                            {
                                //puts("hi 139");
                                expression[i] = "0";
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i-2] == "1" && isConst1) //"1 x *" -> "x"
                            {
                                //puts("hi 147");
                                expression[i] = expression[i-1];
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i-1] == "1" && isConst2) //"x 1 *" -> "x"
                            {
                                //puts("hi 155");
                                expression[i] = expression[i-2];
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                        }

                        else if (expression[i] == "+")
                        {
                            if (expression[i-2] == "0" && isConst1) //"0 x +" -> "x"
                            {
                                //puts("hi 167");
                                expression[i] = expression[i-1];
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i-1] == "0" && isConst2) //"x 0 +" -> "x"
                            {
                                //puts("hi 175");
                                expression[i] = expression[i-2];
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                        }

                        else if (expression[i] == "/")
                        {
                            if (expression[i-2] == "0" && isConst1) // "0 x /" -> "0"
                            {
                                //puts("hi 187");
                                expression[i] = "0";
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i-1] == "1" && isConst2) // "x 1 /" -> "x"
                            {
                                //puts("hi 195");
                                expression[i] = expression[i-2];
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (isConst1 && isConst2 && (expression[i-1] == expression[i-2])) // "x x /" -> "1"
                            {
                                //puts("hi 203");
                                expression[i] = "1";
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                        }

                        else if (expression[i] == "^")
                        {
                            if (expression[i-1] == "0" && isConst2) // "x 0 ^" -> "1"
                            {
                                //puts("hi 223");
                                expression[i] = "1";
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i-2] == "0" && isConst1) // "0 x ^" -> "0" (x > 0)
                            {
                                //puts("hi 215");
                                expression[i] = "0";
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i-2] == "1" && isConst1) // "1 x ^" -> "1"
                            {
                                //puts("hi 231");
                                expression[i] = "1";
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                            else if (expression[i-1] == "1" && isConst2) // "x 1 ^" -> "x"
                            {
                                //puts("hi 239");
                                expression[i] = expression[i-2];
                                expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                                simplified = true;
                                break;
                            }
                        }
                    }

                    else if (is_unary(expression[i]) && isFloat(expression[i-1]))
                    {
                        if (expression[i] == "cos")
                        {
                            expression[i] = simplifyString(std::to_string(cos(Stof(expression[i-1]))));
                            expression.erase(expression.begin() + i - 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "~")
                        {
                            expression[i] = simplifyString(std::to_string(-(Stof(expression[i-1]))));
                            expression.erase(expression.begin() + i - 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "sin")
                        {
                            expression[i] = simplifyString(std::to_string(sin(Stof(expression[i-1]))));
                            expression.erase(expression.begin() + i - 1);
                            simplified = true;
                            break;
                        }
                        else if ((expression[i] == "ln") || (expression[i] == "log"))
                        {
                            expression[i] = simplifyString(std::to_string(log(Stof(expression[i-1])))); // Natural log (ln)
                            expression.erase(expression.begin() + i - 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "asin" || expression[i] == "arcsin")
                        {
                            expression[i] = simplifyString(std::to_string(asin(Stof(expression[i-1]))));
                            expression.erase(expression.begin() + i - 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "acos" || expression[i] == "arccos")
                        {
                            expression[i] = simplifyString(std::to_string(acos(Stof(expression[i-1]))));
                            expression.erase(expression.begin() + i - 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "exp")
                        {
                            expression[i] = simplifyString(std::to_string(exp(Stof(expression[i-1]))));
                            expression.erase(expression.begin() + i - 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "sech")
                        {
                            expression[i] = simplifyString(std::to_string(1 / cosh(Stof(expression[i-1])))); // sech(x) = 1 / cosh(x)
                            expression.erase(expression.begin() + i - 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "tanh")
                        {
                            expression[i] = simplifyString(std::to_string(tanh(Stof(expression[i-1]))));
                            expression.erase(expression.begin() + i - 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "sqrt")
                        {
                            expression[i] = simplifyString(std::to_string(sqrt(Stof(expression[i-1]))));
                            expression.erase(expression.begin() + i - 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "abs")
                        {
                            expression[i] = simplifyString(std::to_string(abs(Stof(expression[i-1]))));
                            expression.erase(expression.begin() + i - 1);
                            simplified = true;
                            break;
                        }
                    }

                    else if (is_unary(expression[i]))
                    {
                        if (expression[i] == "~" && expression[i-1] == "~")
                        {
                            expression.erase(expression.begin() + i - 1, expression.begin() + i + 1); // Remove elements at i - 1 and i
                            simplified = true;
                            break;
                        }
                        //TODO: Add 0 ~ -> 0
                        else if (expression[i] == "exp" && (expression[i-1] == "ln" || expression[i-1] == "log"))
                        {
                            //puts("hi 360");
                            expression.erase(expression.begin() + i - 1, expression.begin() + i + 1); // Remove elements at i - 1 and i
                            simplified = true;
                            break;
                        }
                        else if (expression[i-1] == "exp" && (expression[i] == "ln" || expression[i] == "log"))
                        {
                            //puts("hi 368");
                            expression.erase(expression.begin() + i - 1, expression.begin() + i + 1); // Remove elements at i - 1 and i
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "cos" && (expression[i-1] == "acos" || expression[i-1] == "arccos"))
                        {
                            //puts("hi 408");
                            expression.erase(expression.begin() + i - 1, expression.begin() + i + 1); // Remove elements at i - 1 and i
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "cos" && expression[i-1] == "~") //cos(-x) -> cos(x)
                        {
                            //puts("hi 688");
                            expression.erase(expression.begin() + i - 1); // Remove the '~'
                            simplified = true;
                            break;
                        }
                        else if (expression[i-1] == "cos" && (expression[i] == "acos" || expression[i] == "arccos"))
                        {
                            //puts("hi 416");
                            expression.erase(expression.begin() + i - 1, expression.begin() + i + 1); // Remove elements at i - 1 and i
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "sin" && (expression[i-1] == "asin" || expression[i-1] == "arcsin"))
                        {
                            //puts("hi 424");
                            expression.erase(expression.begin() + i - 1, expression.begin() + i + 1); // Remove elements at i - 1 and i
                            simplified = true;
                            break;
                        }
                        else if (expression[i-1] == "sin" && (expression[i] == "asin" || expression[i] == "arcsin"))
                        {
                            //puts("hi 432");
                            expression.erase(expression.begin() + i - 1, expression.begin() + i + 1); // Remove elements at i - 1 and i
                            simplified = true;
                            break;
                        }
                    }
                }
            }
        }
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
        assert(this->stack.size() > idx);

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
                    assert(this->stack.size() > idx);
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

    std::vector<std::string> get_legal_moves(int idx)
    {
        assert(n.size() > idx);
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

        for (int i = (is_prefix ? (static_cast<int>(pieces[idx].size()) - 1) : 0); (is_prefix ? (i >= 0) : (i < static_cast<int>(pieces[idx].size()))); (is_prefix ? (i--) : (i++)))
        {
            token = pieces[idx][i];
//            puts(("\ntoken = "+token+"\n").c_str());
            if (is_const(token)) // leaf
            {
                if (token.compare(0, 5, "const") == 0 && show_consts)
                {
                    stack.push(std::to_string((this->params)(std::stoi(token.substr(5)))));
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
        size_t sz = pieces.size() - 1;
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
        temp.reserve(2*pieces[idx].size());
        size_t sz = pieces[idx].size() - 1;
        for (size_t i = 0; i <= sz; i++)
        {
            token = pieces[idx][i];

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
        size_t sz = pieces.size() - 1;
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
        size_t sz = pieces.size() - 1;
        for (int jdx = 0; jdx < sz; jdx++)
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
        size_t sz = pieces.size() - 1;
        for (int jdx = 0; jdx < sz; jdx++)
        {
            temp += expression(pieces[jdx], show_consts) + ", ";
        }
        temp += expression(pieces[sz], show_consts);
        return temp;
    }

    float expression_evaluator(const Eigen::VectorXf& params, const std::vector<std::string>& pieces, float t) const
    {
        std::stack<float> stack;
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
                    stack.push(0.0f);
                }
                else if (token == "1")
                {
                    stack.push(1.0f);
                }
                else if (token == "2")
                {
                    stack.push(2.0f);
                }
                else if (token == "4")
                {
                    stack.push(4.0f);
                }
                else if (isFloat(token))
                {
                    stack.push(Stof(token));
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
                    float temp = stack.top();
                    stack.pop();
                    stack.push(cos(temp));
                }
                else if (token == "exp")
                {
                    float temp = stack.top();
                    stack.pop();
                    stack.push(exp(temp));
                }
                else if (token == "sqrt")
                {
                    float temp = stack.top();
                    stack.pop();
                    stack.push(sqrt(temp));
                }
                else if (token == "sin")
                {
                    float temp = stack.top();
                    stack.pop();
                    stack.push(sin(temp));
                }
                else if (token == "asin" || token == "arcsin")
                {
                    float temp = stack.top();
                    stack.pop();
                    stack.push(asin(temp));
                }
                else if (token == "log" || token == "ln")
                {
                    float temp = stack.top();
                    stack.pop();
                    stack.push(log(temp));
                }
                else if (token == "tanh")
                {
                    float temp = stack.top();
                    stack.pop();
                    stack.push(tanh(temp));
                }
                else if (token == "sech")
                {
                    float temp = stack.top();
                    stack.pop();
                    stack.push(1.0f/cosh(temp));
                }
                else if (token == "acos" || token == "arccos")
                {
                    float temp = stack.top();
                    stack.pop();
                    stack.push(acos(temp));
                }
                else if (token == "~") //unary minus
                {
                    float temp = stack.top();
                    stack.pop();
                    stack.push(-temp);
                }
                else if (token == "abs") //unary abs
                {
                    float temp = stack.top();
                    stack.pop();
                    stack.push(abs(temp));
                }
            }
            else // binary operator
            {
                float left_operand = stack.top();
                stack.pop();
                float right_operand = stack.top();
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

    Eigen::VectorXf expression_evaluator(const Eigen::VectorXf& params, const std::vector<std::string>& pieces) const
    {
        std::stack<Eigen::VectorXf> stack;
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
                    if (temp_idx >= params.size())
                    {
                        throw std::runtime_error("\ntemp_idx = "+std::to_string(temp_idx)
                                                 +"\nparams.size() = "+std::to_string(params.size())
                                                 +"\nnum_consts = "+std::to_string(this->__num_consts())
                                                 +"\nBoard::expression_dict.size() = "+std::to_string(Board::expression_dict.size()));
                    }
                    stack.push(Eigen::VectorXf::Ones(Board::data.numRows())*params(temp_idx));
                }
                else if (token == "0")
                {
                    stack.push(Eigen::VectorXf::Zero(Board::data.numRows()));
                }
                else if (token == "1")
                {
                    stack.push(Eigen::VectorXf::Ones(Board::data.numRows()));
                }
                else if (token == "2")
                {
                    stack.push(Eigen::VectorXf::Ones(Board::data.numRows())*2.0f);
                }
                else if (token == "4")
                {
                    stack.push(Eigen::VectorXf::Ones(Board::data.numRows())*4.0f);
                }
                else if (isFloat(token))
                {
                    stack.push(Eigen::VectorXf::Ones(Board::data.numRows())*Stof(token));
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
                    Eigen::VectorXf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().cos());
                }
                else if (token == "exp")
                {
                    Eigen::VectorXf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().exp());
                }
                else if (token == "sqrt")
                {
                    Eigen::VectorXf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().sqrt());
                }
                else if (token == "sin")
                {
                    Eigen::VectorXf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().sin());
                }
                else if (token == "asin" || token == "arcsin")
                {
                    Eigen::VectorXf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().asin());
                }
                else if (token == "log" || token == "ln")
                {
                    Eigen::VectorXf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().log());
                }
                else if (token == "tanh")
                {
                    Eigen::VectorXf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().tanh());
                }
                else if (token == "sech")
                {
                    Eigen::VectorXf temp = stack.top();
                    stack.pop();
                    stack.push(1/temp.array().cosh());
                }
                else if (token == "acos" || token == "arccos")
                {
                    Eigen::VectorXf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().acos());
                }
                else if (token == "~") //unary minus
                {
                    Eigen::VectorXf temp = stack.top();
                    stack.pop();
                    stack.push(-temp.array());
                }
                else if (token == "abs") //unary abs
                {
                    Eigen::VectorXf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().cwiseAbs());
                }
            }
            else // binary operator
            {
                Eigen::VectorXf left_operand = stack.top();
                stack.pop();
                Eigen::VectorXf right_operand = stack.top();
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

    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> expression_evaluator(const std::vector<Eigen::AutoDiffScalar<Eigen::VectorXf>>& parameters, const std::vector<std::string>& pieces) const
    {
        std::stack<Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>> stack;
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
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>::Constant(Board::data.numRows(), parameters[std::stoi(token.substr(5))]));
                }
                else if (token == "0")
                {
                    //                    std::cout << "\nparameters[" << const_count << "] = " << parameters[const_count].value() << '\n';
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>::Constant(Board::data.numRows(), 0.0f));
                }
                else if (token == "1")
                {
                    //                    std::cout << "\nparameters[" << const_count << "] = " << parameters[const_count].value() << '\n';
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>::Constant(Board::data.numRows(), 1.0f));
                }
                else if (token == "2")
                {
                    //                    std::cout << "\nparameters[" << const_count << "] = " << parameters[const_count].value() << '\n';
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>::Constant(Board::data.numRows(), 2.0f));
                }
                else if (token == "4")
                {
                    //                    std::cout << "\nparameters[" << const_count << "] = " << parameters[const_count].value() << '\n';
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>::Constant(Board::data.numRows(), 4.0f));
                }
                else if (isFloat(token))
                {
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>::Constant(Board::data.numRows(), Stof(token)));
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
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().cos());
                }
                else if (token == "exp")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().exp());
                }
                else if (token == "sqrt")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().sqrt());
                }
                else if (token == "sin")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().sin());
                }
                else if (token == "asin" || token == "arcsin")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().asin());
                }
                else if (token == "log" || token == "ln")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().log());
                }
                else if (token == "tanh")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().tanh());
                }
                else if (token == "sech")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(1/temp.array().cosh());
                }
                else if (token == "acos" || token == "arccos")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().acos());
                }
                else if (token == "~") //unary minus
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(-temp.array());
                }
                else if (token == "abs") //unary minus
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().cwiseAbs());
                }
            }
            else // binary operator
            {
                Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> left_operand = stack.top();
                stack.pop();
                Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> right_operand = stack.top();
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

    std::vector<Eigen::VectorXf> expression_evaluator(const Eigen::VectorXf& params, const std::vector<std::vector<std::string>>& pieces) const
    {
        size_t sz = pieces.size();
        std::vector<Eigen::VectorXf> temp(sz);

        for (size_t idx = 0; idx < sz; idx++)
        {
            temp[idx] = expression_evaluator(params, pieces[idx]);
        }
        return temp;
    }

    std::vector<Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>> expression_evaluator(const std::vector<Eigen::AutoDiffScalar<Eigen::VectorXf>>& params, const std::vector<std::vector<std::string>>& pieces) const
    {
        std::vector<Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>> temp;
        size_t sz = pieces.size();
        temp.reserve(sz);
        for (size_t idx = 0; idx < sz; idx++)
        {
            temp.push_back(expression_evaluator(params, pieces[idx]));
        }
        return temp;
    }

    Eigen::AutoDiffScalar<Eigen::VectorXf> grad_func(std::vector<Eigen::AutoDiffScalar<Eigen::VectorXf>>& inputs)
    {
        return MSE(expression_evaluator(inputs, this->diffeq_result));
    }

    /*
     x: parameter vector: (x_0, x_1, ..., x_{x.size()-1})
     g: gradient evaluated at x: (g_0(x_0), g_1(x_1), ..., g_{g.size()-1}(x_{x.size()-1}))
     */
    float operator()(Eigen::VectorXf& x, Eigen::VectorXf& grad)
    {
        if (this->fit_method == "LBFGS" || this->fit_method == "LBFGSB")
        {
            assert(grad.size() == x.size());
            float grad_piece_prefactor = this->isConstTol/(Board::data.numRows()*this->num_objectives);
            float mse = MSE(expression_evaluator(x, this->diffeq_result));
            if (this->fit_grad_method == "naive_numerical")
            {
                float low_b, temp, low_inv_var_b;
                for (int i = 0; i < x.size(); i++) //finite differences wrt x evaluated at the current values x(i)
                {
                    //https://stackoverflow.com/a/38855586/18255427
                    temp = x(i);
                    x(i) -= 0.00001f;
                    low_inv_var_b = grad_piece_prefactor;
                    if (low_inv_var_b)
                    {
                        low_inv_var_b /= VarianceSum(this->expression_evaluator(x, this->pieces)); //larger variance in SR expressions -> smaller penalty
                    }
                    low_b = MSE(expression_evaluator(x, this->diffeq_result)) + low_inv_var_b;
                    x(i) = temp + 0.00001f;
                    low_inv_var_b = grad_piece_prefactor;
                    if (low_inv_var_b)
                    {
                        low_inv_var_b /= VarianceSum(this->expression_evaluator(x, this->pieces)); //larger variance in SR expressions -> smaller penalty
                    }
                    grad(i) = ((MSE(expression_evaluator(x, this->diffeq_result)) + low_inv_var_b) - low_b) / 0.00002f;
                    x(i) = temp;
                }
            }

            else if (this->fit_grad_method == "autodiff")
            {
                size_t sz = x.size();
                std::vector<Eigen::AutoDiffScalar<Eigen::VectorXf>> inputs(sz);
                inputs.reserve(sz);
                for (size_t i = 0; i < sz; i++)
                {
                    inputs[i].value() = x(i);
                    inputs[i].derivatives() = Eigen::VectorXf::Unit(sz, i);
                }
                grad = grad_func(inputs).derivatives();
            }
            return mse;
        }
        else if (this->fit_method == "LevenbergMarquardt")
        {
//            if (Board::expression_dict.size() >= Board::max_expression_dict_sz)
//            {
//                std::scoped_lock str_lock(Board::thread_locker);
//                puts(("NOW 3326 Board::expression_dict.size() >= Board::max_expression_dict_sz\n(params.size() == num_consts) is "
//                     +std::to_string(this->params.size() == this->__num_consts())).c_str());
//            }
            auto temp = this->expression_evaluator(x, this->diffeq_result); //std::vector<Eigen::VectorXf>
            std::vector<Eigen::VectorXf> expr_eval_var;
            if (this->isConstTol)
            {
                expr_eval_var = Variance(this->expression_evaluator(x, this->pieces)); //std::vector<Eigen::VectorXf>
            }
            size_t num_cols = temp.size(), num_rows = temp[0].size();
            size_t num_piece_cols = expr_eval_var.size();
            size_t num_piece_vals = num_piece_cols*num_rows;
            float grad_piece_prefactor = (this->isConstTol) ? (this->isConstTol/(num_piece_vals)) : 0.0f;
            size_t total_cols = num_cols + num_piece_cols;
            assert((!expr_eval_var.size()) || (num_rows == expr_eval_var[0].size()));
            assert(num_piece_cols == ((this->isConstTol) ? this->num_objectives : 0));
            assert(grad.size() == num_rows*total_cols);
            for (size_t kdx = 0; kdx < num_rows; kdx++) //for each row
            {
                for (size_t ldx = 0; ldx < num_cols; ldx++) //first loop over each differential equation value
                {
                    grad(kdx*total_cols + ldx) = temp[ldx][kdx]; // Assign values directly
                }
                if (this->isConstTol)
                {
                    for (size_t ldx = num_cols, mdx = 0; ldx < total_cols; ldx++, mdx++) //then loop over each SR-expression value
                    {
                        grad(kdx*total_cols + ldx) = (grad_piece_prefactor / expr_eval_var[mdx][kdx]);
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
        return 0.0f;
    }
    bool LBFGS()
    {
        bool improved = false;
        auto start_time = Clock::now();
        LBFGSpp::LBFGSParam<float> param;
        param.epsilon = 1e-6;
        param.max_iterations = this->num_fit_iter;
        //https://lbfgspp.statr.me/doc/LineSearchBacktracking_8h_source.html
        LBFGSpp::LBFGSSolver<float, LBFGSpp::LineSearchMoreThuente> solver(param); //LineSearchBacktracking, LineSearchBracketing, LineSearchMoreThuente, LineSearchNocedalWright
        float fx;

        Eigen::VectorXf eigenVec = this->params;
        float mse = MSE(expression_evaluator(this->params, this->diffeq_result));
        try
        {
            solver.minimize((*this), eigenVec, fx);
        }
        catch (std::runtime_error& e){}
        catch (std::invalid_argument& e){}

        //printf("mse = %f -> fx = %f\n", mse, fx);
        if (fx < mse)
        {
            //printf("mse = %f -> fx = %f\n", mse, fx);
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
        LBFGSpp::LBFGSBParam<float> param;
        param.epsilon = 1e-6;
        param.max_iterations = this->num_fit_iter;
        //https://lbfgspp.statr.me/doc/LineSearchBacktracking_8h_source.html
        LBFGSpp::LBFGSBSolver<float> solver(param); //LineSearchBacktracking, LineSearchBracketing, LineSearchMoreThuente, LineSearchNocedalWright
        float fx;

        Eigen::VectorXf eigenVec = this->params;
        float mse = MSE(expression_evaluator(this->params, this->diffeq_result));
        try
        {
            solver.minimize((*this), eigenVec, fx, Eigen::VectorXf::Constant(eigenVec.size(), -std::numeric_limits<float>::infinity()), Eigen::VectorXf::Constant(eigenVec.size(), std::numeric_limits<float>::infinity()));
            //solver.minimize((*this), eigenVec, fx, Eigen::VectorXf::Constant(eigenVec.size(), -10.f), Eigen::VectorXf::Constant(eigenVec.size(), 10.f));
        }
        catch (std::runtime_error& e){}
        catch (std::invalid_argument& e){}
        catch (std::logic_error& e){}

        //printf("mse = %f -> fx = %f\n", mse, fx);
        if (fx < mse)
        {
            //printf("mse = %f -> fx = %f\n", mse, fx);
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

    int df(Eigen::VectorXf &x, Eigen::MatrixXf &fjac)
    {
        float epsilon, temp;
        epsilon = 1e-5f;

        for (int i = 0; i < x.size(); i++)
        {
            //Eigen::VectorXf xPlus(x);
            //xPlus(i) += epsilon;
            //
            //Eigen::VectorXf xMinus(x);
            //xMinus(i) -= epsilon;
            //x(i) -= epsilon;

            temp = x(i);

            x(i) = temp + epsilon;
            Eigen::VectorXf fvecPlus(values());
            operator()(x, fvecPlus);

            x(i) = temp - epsilon;
            Eigen::VectorXf fvecMinus(values());
            operator()(x, fvecMinus);

            fjac.block(0, i, values(), 1) = std::move((fvecPlus - fvecMinus) / (2.0f * epsilon));

            x(i) = temp;
        }
        return 0;
    }

    bool LevenbergMarquardt()
    {
        bool improved = false;
        auto start_time = Clock::now();
        Eigen::LevenbergMarquardt<decltype(*this), float> lm(*this);
        float score_before = MSE(expression_evaluator(this->params, this->diffeq_result));
        lm.parameters.maxfev = this->num_fit_iter;
        //std::cout << "ftol (Cost function change) = " << lm.parameters.ftol << '\n';
        //std::cout << "xtol (Parameters change) = " << lm.parameters.xtol << '\n';
        lm.minimize(this->params);
        float score_after = MSE(expression_evaluator(this->params, this->diffeq_result));
        if (score_after < score_before)
        {
            //printf("score_before = %f -> score_after = %f\n", score_before, score_after);
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
        for (int jdx = 0; jdx < this->pieces.size(); jdx++) //loops over each generated symbolic expression
        {
            if (isConstant(expression_evaluator(this->params, this->pieces[jdx]), this->isConstTol))
            {
                return false;
            }
        }
        return true;
    }

    float fitFunctionToData()
    {
        float score = 0.0f;
        bool depends_symb_on_x0 = false;
        for (int jdx = 0; jdx < this->pieces.size(); jdx++) //loops over each generated symbolic expression
        {
            //This block below checks if `this->pieces[jdx]` depends on `x0`.
            {
                depends_symb_on_x0 = false;
                for (const auto& piece: this->pieces[jdx])
                {
                    if (piece != "x0")
                    {
                        for (int i = 0; i < static_cast<int>(piece.size())-2; i++)
                        {
                            //checks if next 3 characters are 'n', 'a', 'n' or 'i', 'n', 'f'
                            if (((piece[i] == 'n') && (piece[i+1] == 'a') && (piece[i+2] == 'n')) ||
                                ((piece[i] == 'i') && (piece[i+1] == 'n') && (piece[i+2] == 'f')))
                            {
                                this->MSE_curr = FLT_MAX;
                                return score;
                            }
                            //checks if next 3 characters are 'x', '0', * or *, 'x', '0'
                            else if (((piece[i] == 'x') && (piece[i+1] == '0')) ||
                                    ((piece[i+1] == 'x') && (piece[i+2] == '0')))
                            {
                                depends_symb_on_x0 = true;
                            }
                        }
                    }
                    else
                    {
                        depends_symb_on_x0 = true;
                    }
                }
                //We reject the solution if it doesn't depend on `x0`
                if (!depends_symb_on_x0)
                {
                    this->MSE_curr = FLT_MAX;
                    return score;
                }
            }
            Eigen::VectorXf expression_eval = expression_evaluator(this->params, this->pieces[jdx]);
            if (/*(Board::__num_features == 1) && */isConstant(expression_eval, this->isConstTol)) //Ignore the trivial solution (1-d functions)!
            {
                this->MSE_curr = FLT_MAX;
                return score;
            }
            else if (Board::__num_features > 1)
            {
                std::vector<int> grasp;
                for (const std::string& i: Board::__input_vars)
                {
                    //Below, we're checking if the independent variable `i` is present in the expression `this->pieces[jdx]`.
                    //We checked "x0" above; hence, the `(i != "x0")` bit.
                    if ((i != "x0") && std::find(this->pieces[jdx].begin(), this->pieces[jdx].end(), i) == this->pieces[jdx].end())
                    {
                        //then `this->pieces[jdx]` does not depend on `i` so it is a trivial expression -> get out of dodge!
                        this->MSE_curr = FLT_MAX;
                        return score;
                    }
                    //If the variable `i` is found, we then test the derivative wrt `i` to check if it's 0 within `this->isConstTol` tolerance.
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
                        this->MSE_curr = FLT_MAX;
                        return score;
                    }
                }
            }
        }
        if (this->params.size())
        {
            this->diffeq_result = diffeq(*this);
            assert(this->diffeq_result.size() == this->num_diff_eqns);
            for (int jdx = 0; jdx < this->diffeq_result.size(); jdx++)
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
            Eigen::VectorXf temp_vec; //need to have a back-up vector in case `improved == false` so we can get the score of the expression we just built.

            if (improved && this->passesConstantThreshold()) //If improved, update the expression_dict with this->params, TODO: add a check here in addition to `improved` to also make sure `this->isConstTol` wasn't violated in the process of optimizing
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
            std::vector<Eigen::VectorXf> expression_eval = expression_evaluator(temp_vec, this->diffeq_result);
            score = loss_func(expression_eval[0]);
//            std::cout << "expression values for (" << this->_to_infix(this->diffeq_result[0], false) << ") = " << hstack(Board::data["x0"], expression_eval[0]) << "\nand params = " << temp_vec << "\nand pieces[0] = " << this->_to_infix(this->pieces[0], false) << '\n';
            if (isInvalid(score))
            {
                this->MSE_curr = FLT_MAX;
                return 0.0f;
            }
            else
            {
                assert(score >= 0.0f);
            }

            this->MSE_curr = (1.0f/score) - 1.0f;
            float temp;
            for (size_t jdx = 1; jdx < expression_eval.size(); ++jdx)
            {
                temp = loss_func(expression_eval[jdx]);

                if (isInvalid(temp))
                {
                    this->MSE_curr = FLT_MAX;
                    return 0.0f;
                }
                else
                {
                    assert(temp >= 0.0f);
                }
                score += temp;
                this->MSE_curr += (1.0f/temp) - 1.0f;
            }
            this->params = temp_vec; //copy `temp_vec` back into `this->params` for displaying purposes
        }
        else
        {
            this->diffeq_result = diffeq(*this);
            assert(this->diffeq_result.size() == this->num_diff_eqns);
            score = 0.0f;
            float temp;
            this->MSE_curr = 0.0f;
            for (int jdx = 0; jdx < this->diffeq_result.size(); jdx++)
            {
                ((this->expression_type == "prefix") ? simplifyPN(this->diffeq_result[jdx]) : simplifyRPN(this->diffeq_result[jdx]));
                auto temp_data = expression_evaluator(this->params, this->diffeq_result[jdx]);
                temp = loss_func(temp_data);
                if (isInvalid(temp))
                {
                    this->MSE_curr = FLT_MAX;
                    return 0.0f;
                }
                score += temp;
                this->MSE_curr += ((1.0f/temp) - 1.0f);
            }
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
    float complete_status(int idx, bool cache = true)
    {
        assert(this->stack.size() > idx);
        assert(n.size() > idx);
        if (this->pieces[idx].empty())
        {
            this->stack[idx] = std::vector<int>();
            this->idx[idx] = 0;
            if (this->expression_type == "prefix")
            {
                this->depth[idx] = 0, this->num_binary[idx] = 0, this->num_leaves[idx] = 0;
            }
        }
        assert(this->stack.size() > idx);
        auto [depth, complete] =  ((this->expression_type == "prefix") ? getPNdepth(pieces[idx], idx, 0 /*start*/, 0 /*stop*/, this->cache && cache /*cache*/, true /*modify*/) : getRPNdepth(pieces[idx], idx, 0 /*start*/, 0 /*stop*/, this->cache && cache /*cache*/, true /*modify*/)); //structured binding :)
        if (!complete || depth < this->n[idx]) //Expression not complete
        {
            return -1.0f;
        }
        else if (idx < this->pieces.size() - 1)
        {
            return 0.0f;
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
                    for (int jdx = 0; jdx < this->pieces.size(); jdx++)
                    {
                        ((this->expression_type == "prefix") ? simplifyPN(this->pieces[jdx]) : simplifyRPN(this->pieces[jdx])); //simplify expression
                    }
                }
                else
                {
                    this->temp_pieces = this->pieces;
                    for (int jdx = 0; jdx < this->pieces.size(); jdx++)
                    {
                        ((this->expression_type == "prefix") ? simplifyPN(this->pieces[jdx]) : simplifyRPN(this->pieces[jdx])); //simplify expression
                    }
                    reset_const_token_labels(this->pieces, this->num_consts_diff);
                }
                if (this->num_consts_diff || this->use_const_pieces) //If I have tokens that need to be optimized
                {
                    this->expression_string.clear();
                    this->expression_string.reserve(8*pieces.size());

                    for (int jdx = 0; jdx < this->pieces.size(); jdx++)
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
                        try //MAYBE: Might be able to remove this try-catch block itf.
                        {
                            if (Board::expression_dict.size() < Board::max_expression_dict_sz) //if the capacity of the shared dict has not been exceeded.
                            {
                                Board::expression_dict.insert_or_assign(this->expression_string, Eigen::VectorXf());
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
                        for (int jdx = 0; jdx < this->pieces.size(); jdx++)
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
                                        std::string int_suffix = token.substr(5);
                                        int temp_idx = std::stoi(int_suffix);
                                        if (temp_idx >= this->num_consts_diff)
                                        {
                                            throw std::runtime_error("Somehow, there's a const token with a suffix integer equal to " + std::to_string(temp_idx));
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
                float res = fitFunctionToData();
                if (!this->simplify_original) //nah there's a bug here with consts
                {
                    this->pieces = this->temp_pieces;
                }
                return res;
            }
            return 0.0f;
        }
    }
    const Eigen::VectorXf& operator[] (int i)
    {
        return Board::data[i];
    }
    const Eigen::VectorXf& operator[] (const std::string& i)
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
    //depth-n sub-expression in the expression individual
    void get_indices(std::vector<std::pair<int, int>>& sub_exprs, std::vector<std::string>& individual, int idx)
    {
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
            else if (this->n[idx] == 0) //depth-0 sub-trees are leaf-nodes
            {
                sub_exprs.push_back(std::make_pair(k, k));
                continue;
            }

            auto [start, stop] = std::make_pair( std::min(k, ptr_GB), std::max(k, ptr_GB));
            auto [depth, complete] =  ((expression_type == "prefix") ? getPNdepth(individual, idx, start, stop+1, false /*cache*/) : getRPNdepth(individual, idx, start, stop+1, false /*cache*/));

            if (complete && (depth == this->n[idx]))
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
std::vector<std::vector<std::string>> SolitonWaveFengEq14and15Laser(Board& x)
{
    //std::vector<std::vector<std::string>> results(10); //2 equations for the ODE, 4 equations for the boundary conditions, 2 equations for symmetry of a(u) and n respectively, 2 equations for data
    std::vector<std::vector<std::string>> results(9); //2 equations for the ODE, 4 equations for the boundary conditions, 1 equation for symmetry of n respectively, 2 equations for data
    for (int i = 0; i < results.size(); i++){results[i].reserve(100);}
    std::vector<std::string> temp, temp_prime;
    temp.reserve(100);
    temp_prime.reserve(100);
    std::vector<int> grasp;
    /*
      For parameters commented-out below (first 2 equations only):
       - Best score = 0.997519, MSE = 2.17869e+27
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
        - Best score = 5.1745, MSE = 3.46556
        - Best expression = arccos((0.000004 ^ sech(x0))), exp(~(sin(cos(x0))))
        - Best expression (original format) = 0.000004 x0 sech ^ arccos, x0 cos sin ~ exp

      For parameters below (first 8 equations)
        - Best score = 7.99935, MSE = 0.000654101
        - Best expression = (1 / (-10.594090 / arcsin(sech(x0)))), cos((sech(x0) - (cos(1) - sech(x0))))
        - Best expression (original format) = 1 -10.594090 x0 sech arcsin / /, x0 sech 1 cos x0 sech - - cos

      For parameters below (all 10 equations)
        - Best score = 9.99802, MSE = 0.00198078
        - Best expression = (cos(1.559132) * sech((x0 * 0.774245))), sech((sech(x0) * (0.157922 ^ cos(4))))
        - Best expression (original format) = 1.559132 cos x0 0.774245 * sech *, x0 sech 0.157922 4 cos ^ * sech

      For parameters below (first 6 equations and last 3 equations WITH `const0`)
        - Best score = 9.99802, MSE = 0.00198078
        - Best expression = (cos(1.559132) * sech((x0 * 0.774245))), sech((sech(x0) * (0.157922 ^ cos(4))))
        - Best expression (original format) = 1.559132 cos x0 0.774245 * sech *, x0 sech 0.157922 4 cos ^ * sech

     For parameters below (same configuration as the one right above but adding `10 *` at the end of last equation and changing threshold `this->isConstTol` to 0.001 instead of 0 and changing `^ 2` to `abs` in last 3 equations
        - Best score = 8.94096, MSE = 0.0602338
        - Best expression = (tanh(tanh(sech(x0))) / ((4 + -10.434288) - (tanh(x0) / exp(1)))), sech((sqrt(10.466281) * (sech(x0) ^ tanh(2))))
        - Best expression (original format) = x0 sech tanh tanh 4 -10.434288 + x0 tanh 1 exp / - /, 10.466281 sqrt x0 sech 2 tanh ^ * sech
        - Best differential equation parameters = {(const0, 5.22145)}
        - Best expression parameters = {}

    */
    constexpr const char* rho = "0.000544662309"; // 1/1836, Figs 10-11 caption, https://www.bing.com/search?q=9.1e-31%2F%201.67e-27%20&qs=n&form=QBRE&sp=-1&ghc=1&lq=0&pq=9.1e-31%2F%201.67e-27%20&sc=0-18&sk=&cvid=09FAD78B6CC6414E98D1BED802D49D1C
    constexpr const char* omega_squared_factor_for_omega_0_point_8_omega_pe = "0.64"; //0.8^2 ω_{pe} = 0.64 ω_{pe}, Figs 10-11 caption "ω = 0.8*ω_{pe}", ω_{pe} = (4*pi*(n_e=n)*(q_e^2))/(m_e), see "III. PROPAGATION MODES"
    constexpr const char* omega_squared_factor = omega_squared_factor_for_omega_0_point_8_omega_pe;
    constexpr const char* cs_squared_factor_for_rho_i_1_over_1836_v_te_0_point_05c_v_ti_0_point_001c = "2.36165577e-6"; //Figure 10: (((.05*.05)/1836) + (.001*.001)), https://www.bing.com/search?q=(((.05*.05)%2F1836)%20%2B%20(.001*.001))&qs=n&form=QBRE&sp=-1&lq=0&pq=(((.05*.05)%2F1836)%20%2B%20(.001*.001))&sc=1-32&sk=&cvid=C563F2D3B3AB42BAA438BCD2FAF6F307&ajf=10
    constexpr const char* cs_squared = cs_squared_factor_for_rho_i_1_over_1836_v_te_0_point_05c_v_ti_0_point_001c;
    constexpr const char* alpha_0 = "0.4";
    constexpr const char* alpha = alpha_0;
//    constexpr const char* one_plus_rho_i_times_alpha_for_rho_i_1_over_1836_alpha_0 = "1.0002178649237472767"; //1 + ρ_i*0.4 = 1.0002178649237472767
//    constexpr const char* one_plus_rho_i_times_alpha = one_plus_rho_i_times_alpha_for_rho_i_1_over_1836_alpha_0; //1 + ρ_i*α
    //constexpr const char* const0 = "4.06461";
    constexpr const char* const0 = "const0";
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
        for (const std::string& i: temp) //u(ξ_min) tanh 1 u(ξ_min) sech / α const0 * - *
        {
            if (i == "x0")
            {
                results[2].push_back(x.feature_mins_maxes[i].first);
            }
            else if (i == alpha)
            {
                results[2].push_back(alpha);
                results[2].push_back(const0);
                results[2].push_back("*");
            }
            else
            {
                results[2].push_back(i);
            }
        }
        assert((2+temp.size()) == results[2].size()); //sanity check
        //u(ξ_max) tanh 1 u(ξ_max) sech / α const0 * - *
        for (const std::string& i: temp) // u(ξ_max) tanh 1 u(ξ_max) sech / α const0 * - *
        {
            if (i == "x0")
            {
                results[3].push_back(x.feature_mins_maxes[i].second);
            }
            else if (i == alpha)
            {
                results[3].push_back(alpha);
                results[3].push_back(const0);
                results[3].push_back("*");
            }
            else
            {
                results[3].push_back(i);
            }
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
        for (const std::string& i: temp) // u tanh 1 u sech / α const0 * - *
        {
            results[8].push_back(i);
            if (i == alpha)
            {
                results[8].push_back(const0);
                results[8].push_back("*");
            }
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

std::vector<std::vector<std::string>> VortexRadialProfile(Board& x)
{
    std::vector<std::vector<std::string>> results;
    std::vector<std::string> result;
    result.reserve(100);
    std::vector<int> grasp;
    std::vector<std::string> R_prime;
    std::string mu = "1";
    std::string S = "1";
    std::string infty = std::to_string(FLT_MAX);

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

{x0: r, x1: θ}
{x.pieces[0]: f}
*/
std::vector<std::vector<std::string>> SwiftHohenberg(Board& x)
{
    //TODO: Probably wanna make these containers you need here attributes of x so you're not potentially allocating memory from scratch (save for `x.pieces`) every time you wanna build the system
    std::vector<std::vector<std::string>> results;
    std::vector<std::string> result, dfdr1, d2fdr2, d3fdr3, d4fdr4,
    dfdtheta1, d2fdtheta2, d3fdtheta3, d4fdtheta4,
    df3dtheta2dr1;
    result.reserve(500);
    std::vector<int> grasp;
    std::vector<std::string> R_prime;
    std::string mu = "1";
    std::string nu = "1";
    std::string infty = std::to_string(FLT_MAX);

    if (x.expression_type == "prefix")
    {
        throw std::invalid_argument("Prefix not implemented yet for this SolitonWaveFengEq14and15Laser function!");
    }
    else if (x.expression_type == "postfix")
    {
        //μ f * ν f * f * f f f * * - + f - 2 ∂^2f/∂r^2 * - ∂^4f/∂r^4 - 2 ∂^3f/∂r^3 * ∂^2f/∂r^2 r / + (∂f/∂r) r r * / - (∂^3f/∂θ^2∂r) r r * / 2 ∂^2f/∂r^2 * r r * r * / - 2 ∂f/∂r * + + r / - 2 ∂^4f/∂θ^2∂r^2 * ∂^3f/∂θ^2∂r r / + (∂^4f/∂θ^4) r r * / + 2 ∂^2f/∂r^2 * - 2 ∂^2f/∂θ^2 * + r r * / - 2 r r * r * / ∂f/∂r 2 ∂^3f/∂θ^2∂r * - 3 r / ∂^2f/∂θ^2 * + * -
        result.push_back(mu); // μ
        for (const std::string& i: x.pieces[0]) // f
        {
            result.push_back(i);
        }
        result.push_back("*"); // *
        result.push_back(nu); // ν
        for (const std::string& i: x.pieces[0]) // f
        {
            result.push_back(i);
        }
        result.push_back("*"); // *
        for (const std::string& i: x.pieces[0]) // f
        {
            result.push_back(i);
        }
        result.push_back("*"); // *
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
        result.push_back("*"); // *
        result.push_back("*"); // *
        result.push_back("-"); // -
        result.push_back("+"); // +
        for (const std::string& i: x.pieces[0]) // f
        {
            result.push_back(i);
        }
        result.push_back("-"); // -
        result.push_back("2"); // 2
        x.derivePostfix(0, x.pieces[0].size()-1, "x0", x.pieces[0], grasp);
        dfdr1 = x.derivat;
        x.derivePostfix(0, dfdr1.size()-1, "x0", dfdr1, grasp);
        d2fdr2 = x.derivat;
        for (const std::string& i: d2fdr2) // ∂^2f/∂r^2
        {
            result.push_back(i);
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
        for (const std::string& i: d2fdr2) // ∂^2f/∂r^2
        {
            result.push_back(i);
        }
        result.push_back("x0"); // r
        result.push_back("/"); // /
        result.push_back("+"); // +
        for (const std::string& i: dfdr1) // ∂f/∂r
        {
            result.push_back(i);
        }
        result.push_back("x0"); // r
        result.push_back("x0"); // r
        result.push_back("*"); // *
        result.push_back("/"); // /
        result.push_back("-"); // -
        x.derivePostfix(0, x.pieces[0].size()-1, "x1", x.pieces[0], grasp);
        dfdtheta1 = x.derivat;
        x.derivePostfix(0, dfdtheta1.size()-1, "x1", dfdtheta1, grasp);
        d2fdtheta2 = x.derivat;
        x.derivePostfix(0, d2fdtheta2.size()-1, "x0", d2fdtheta2, grasp);
        df3dtheta2dr1 = x.derivat;
        for (const std::string& i: df3dtheta2dr1) // ∂^3f/∂θ^2∂r
        {
            result.push_back(i);
        }
        result.push_back("x0"); // r
        result.push_back("x0"); // r
        result.push_back("*"); // *
        result.push_back("/"); // /
        result.push_back("2"); // 2
        for (const std::string& i: d2fdr2) // ∂^2f/∂r^2
        {
            result.push_back(i);
        }
        result.push_back("*"); // *
        result.push_back("x0"); // r
        result.push_back("x0"); // r
        result.push_back("*"); // *
        result.push_back("x0"); // r
        result.push_back("*"); // *
        result.push_back("/"); // /
        result.push_back("-"); // -
        result.push_back("2"); // 2
        for (const std::string& i: dfdr1) // ∂f/∂r
        {
            result.push_back(i);
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
        for (const std::string& i: df3dtheta2dr1) // ∂^3f/∂θ^2∂r
        {
            result.push_back(i);
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
        result.push_back("x0"); // r
        result.push_back("x0"); // r
        result.push_back("*"); // *
        result.push_back("/"); // /
        result.push_back("+"); // +
        result.push_back("2"); // 2
        for (const std::string& i: d2fdr2) // ∂^2f/∂r^2
        {
            result.push_back(i);
        }
        result.push_back("*"); // *
        result.push_back("-"); // -
        result.push_back("2"); // 2
        for (const std::string& i: d2fdtheta2) // ∂^2f/∂θ^2
        {
            result.push_back(i);
        }
        result.push_back("*"); // *
        result.push_back("+"); // +
        result.push_back("x0"); // r
        result.push_back("x0"); // r
        result.push_back("*"); // *
        result.push_back("/"); // /
        result.push_back("-"); // -
        result.push_back("2"); // 2
        result.push_back("x0"); // r
        result.push_back("x0"); // r
        result.push_back("*"); // *
        result.push_back("x0"); // r
        result.push_back("*"); // *
        result.push_back("/"); // /
        for (const std::string& i: dfdr1) // ∂f/∂r
        {
            result.push_back(i);
        }
        result.push_back("2"); // 2
        for (const std::string& i: df3dtheta2dr1) // ∂^3f/∂θ^2∂r
        {
            result.push_back(i);
        }
        result.push_back("*"); // *
        result.push_back("-"); // -
        result.push_back("3"); // 3
        result.push_back("x0"); // r
        result.push_back("/"); // /
        for (const std::string& i: d2fdtheta2) // ∂^2f/∂θ^2
        {
            result.push_back(i);
        }
        result.push_back("*"); // *
        result.push_back("+"); // +
        result.push_back("*"); // *
        result.push_back("-"); // -
    }
    results.push_back(result);
    return results;
}

//x0 -> x, x1 -> y, x2 -> t
std::vector<std::vector<std::string>> TwoDAdvectionDiffusion_1(Board& x)
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
std::vector<std::vector<std::string>> TwoDAdvectionDiffusion_2(Board& x)
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
std::vector<std::vector<std::string>> sech_squared_trial(Board& x)
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
void SimulatedAnnealing(std::vector<std::vector<std::string>> (*diffeq)(Board&), size_t num_diff_eqns, const Eigen::MatrixXf& data, const std::vector<int>& depth, const std::string expression_type = "prefix", size_t num_consts_diff = 0, const std::string method = "LevenbergMarquardt", const int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", const bool cache = true, const double time = 120.0 /*time to run the algorithm in seconds*/, unsigned int num_threads = 0, bool const_tokens = false, float isConstTol = 1e-1f, bool use_const_pieces = false, bool simplifyOriginal = false, int numDataCols = 0, const std::vector<std::vector<std::string>>& seed_expressions = {}, bool exit_early = false)
{

    if (num_threads == 0)
    {
        unsigned int temp = std::thread::hardware_concurrency();
        num_threads = ((temp <= 1) ? 1 : temp);
        printf("num_threads = %u\n", num_threads);
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
    std::atomic<float> max_score{0.0};
    std::atomic<float> best_MSE{FLT_MAX};
    std::string best_expression, orig_expression, best_expr_result, orig_expr_result;

    auto start_time = Clock::now();

    /*
     Inside of thread:
     */
    auto func = [&diffeq, &num_diff_eqns, &depth, &expression_type, &num_consts_diff, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &best_expr_result, &orig_expr_result, &const_tokens, &isConstTol, &use_const_pieces, &simplifyOriginal, &numDataCols, &seed_expressions, &exit_early, &best_MSE]()
    {
        std::random_device rand_dev;
        std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
        Board x(diffeq, num_diff_eqns, true, depth, expression_type, num_consts_diff, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol, use_const_pieces, simplifyOriginal, numDataCols);
        sync_point.arrive_and_wait();
        Board secondary(diffeq, num_diff_eqns, false, std::vector<int>(depth.size(), 0), expression_type, num_consts_diff, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol, use_const_pieces, simplifyOriginal, numDataCols); //For perturbations
        assert(secondary.pieces.size() == secondary.n.size());
        assert(secondary.pieces.size() == x.pieces.size());
        assert(secondary.pieces.size() == x.n.size());
        float score = 0.0f, check_point_score = 0.0f;

        std::vector<std::vector<std::string>> current(depth.size());
        std::vector<std::pair<int, int>> sub_exprs;
        std::vector<std::string> temp_legal_moves;
        size_t piece_to_perturb_idx; //Used in the case of a depth-0 perturbation
        std::string piece_to_perturb; piece_to_perturb.reserve(10); //Used in the case of a depth-0 perturbation
        std::string piece_to_replace_with; piece_to_replace_with.reserve(10); //Used in the case of a depth-0 perturbation
        std::vector<std::uniform_int_distribution<int>> rand_depth_dists(depth.size());
        std::vector<int> rand_depths(depth.size());

        size_t temp_sz;
//        std::string expression, orig_expression, best_expression;
        constexpr float T_max = 0.1f;
        constexpr float T_min = 0.012f;
        constexpr float ratio = T_min/T_max;
        float T = T_max;

        auto P = [&](float delta)
        {
            return exp(delta/T);
        };

        auto updateScore = [&](float r = 1.0f)
        {
//            assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).first == x.n);
//            assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).second);
            if ((score > max_score) || (x.pos_dist(generator) < P(score-max_score)) || (exit_early))
            {
                current = x.pieces; //update current expression
                if ((score > max_score) || (exit_early))
                {
                    max_score = score;
                    std::scoped_lock str_lock(Board::thread_locker);
                    best_MSE = x.MSE_curr;
                    best_expression = x._to_infix();
                    orig_expression = x.expression();
                    best_expr_result = x._to_infix(x.diffeq_result);
                    orig_expr_result = x.expression(x.diffeq_result);
                    std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
                    std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
                    std::cout << "Best score = " << score << ", MSE = " << best_MSE << '\n';
                    std::cout << "Best expression = " << best_expression << '\n';
                    std::cout << "Best expression (original format) = " << orig_expression << '\n';
                    std::cout << "Best differential equation parameters = " << x.print_diff_params() << '\n';
                    std::cout << "Best expression parameters = " << x.print_expression_params() << '\n';
                    std::cout << "Total system result = " << best_expr_result << '\n';
                    std::cout << "Total system result (original format) = " << orig_expr_result << '\n';
                }
                //TODO: Output the MSE for each equation in the system you are trying to solve!
            }
            else
            {
                x.pieces = current; //reset perturbed state to current state
            }
            T = r*T;
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
                            if (int_suffix_num >= x.num_consts_diff) //then it's a const that belongs to pieces -> reset it
                            {
                                token = "const";
                            }
                            else
                            {
                                //Below we test that token is of the form `const{0 <= num < x.num_consts_diff}`
                                assert(((0 <= int_suffix_num) && (int_suffix_num < x.num_consts_diff)));
                            }
                        }
                    }
                }
            }
        };

        //`n` is the vector of mutation-tree depths that dictate how each expression in `x.pieces` will be perturbed, and `i` is the time index
        auto Perturbation = [&](const std::vector<int>& n, float i)
        {
            //Step 1: Generate a random depth-n sub-expression `secondary_one.pieces`
            sub_exprs.clear();
            secondary.n = n;
            for (int jdx = 0; jdx < secondary.num_objectives; jdx++)
            {
                //Pre-validation: clears and sanity check
                secondary.pieces[jdx].clear();
                sub_exprs.clear();
                assert(secondary.n[jdx] <= x.n[jdx]);
                //Step 1a: check for the special case of a depth-0 (i.e. 1 operand) perturbation
                if (n[jdx] == 0)
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
                        {
                            secondary.pieces[jdx].emplace_back(temp_legal_moves[distribution(generator)]); //make the randomly chosen valid move
                        }
                    }
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
                        //Step 2b: Else, start by identifying the starting and stopping index pairs of all depth-n sub-expressions
                        //in `x.pieces[jdx]` and store them in an std::vector<std::pair<int, int>> called `sub_exprs`.
                        secondary.get_indices(sub_exprs, x.pieces[jdx], jdx);
                        if (!sub_exprs.size())
                        {
                            throw std::runtime_error("sub_exprs empty, x.n["+std::to_string(jdx)+"] = "+std::to_string(x.n[jdx])
                                                     + ", secondary.n[" +std::to_string(jdx)+ "] = " +std::to_string(secondary.n[jdx]));
                        }
                        else if (secondary.n[jdx] > x.n[jdx])
                        {
                            throw std::runtime_error("(secondary.n[jdx] > x.n[jdx]), x.n["+std::to_string(jdx)+"] = "+std::to_string(x.n[jdx])
                                                     + ", secondary.n[" +std::to_string(jdx)+ "] = " +std::to_string(secondary.n[jdx]));
                        }
                        //Step 3: Generate a uniform int from 0 to sub_exprs.size() - 1 called `pert_ind`
                        std::uniform_int_distribution<int> distribution(0, sub_exprs.size() - 1);
                        int pert_ind = distribution(generator);

                        //Step 4: Substitute sub_exprs_1[pert_ind] in x.pieces[jdx] with secondary_one.pieces[jdx]
                        auto start = x.pieces[jdx].begin() + sub_exprs[pert_ind].first;
                        auto end = x.pieces[jdx].begin() + std::min(sub_exprs[pert_ind].second, static_cast<int>(x.pieces[jdx].size()));
                        x.pieces[jdx].erase(start, end+1);
                        x.pieces[jdx].insert(start, secondary.pieces[jdx].begin(), secondary.pieces[jdx].end()); //could be a move operation: secondary.pieces doesn't need to be in a defined state after this->params, or erase+insert -> replace?
                        auto depth_and_completion = ((x.expression_type == "prefix") ? x.getPNdepth(x.pieces[jdx], jdx) : x.getRPNdepth(x.pieces[jdx], jdx));
                        if (depth_and_completion.first != x.n[jdx])
                        {
                            throw std::runtime_error("(depth_and_completion.first != x.n[jdx]), x.n["+std::to_string(jdx)+"] = "+std::to_string(x.n[jdx])
                                                     + ", secondary.n[" +std::to_string(jdx)+ "] = " +std::to_string(secondary.n[jdx])
                                                     + ", depth_and_completion.first = "+std::to_string(depth_and_completion.first));
                        }
                    }
                }
                //Step 5: Reset const token labels in pieces
                if (x.use_const_pieces)
                {
                    reset_const_token_labels();
                }
                //Step 6: Evaluate the new mutated `x.pieces` and update score if needed
                score = x.complete_status(x.pieces.size() - 1, false);
                if (score < 0.0f)
                {
                    throw(std::runtime_error("score = "+std::to_string(score)));
                }
                updateScore(pow(ratio, 1.0f/(i+1.0f)));
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
                    {
                        x.pieces[jdx].emplace_back(temp_legal_moves[distribution(generator)]); //make the randomly chosen valid move
                        current[jdx].push_back(x.pieces[jdx].back());
                    }
                }
                auto depth_and_completion = ((x.expression_type == "prefix") ? x.getPNdepth(x.pieces[jdx], jdx) : x.getRPNdepth(x.pieces[jdx], jdx));
                assert(depth_and_completion.first <= x.n[jdx]);
                assert(depth_and_completion.second);
                x.n[jdx] = depth_and_completion.first;
                rand_depth_dists[jdx] = std::uniform_int_distribution<int>(0, depth_and_completion.first);
                rand_depths[jdx] = rand_depth_dists[jdx](generator);
            }
        }
        else
        {
            assert(x.pieces.size() == x.num_objectives);
            for (int jdx = 0; jdx < x.num_objectives; jdx++)
            {
                x.pieces[jdx] = seed_expressions[jdx];
                auto depth_and_completion = ((x.expression_type == "prefix") ? x.getPNdepth(x.pieces[jdx], jdx) : x.getRPNdepth(x.pieces[jdx], jdx));
                if (depth_and_completion.first > x.n[jdx])
                {
                    throw std::runtime_error("Seed expression depth of x.pieces["+std::to_string(jdx)+"] = "
                                             +std::to_string(depth_and_completion.first));
                }
                assert(depth_and_completion.second);
                x.n[jdx] = depth_and_completion.first;
                rand_depth_dists[jdx] = std::uniform_int_distribution<int>(0, depth_and_completion.first);
            }
            score = x.complete_status(x.pieces.size() - 1, false);
            std::cout << "score = " << score << '\n';
        }
        reset_const_token_labels();
        updateScore();
        if (exit_early)
        {
            exit(1);
        }
        for (float i = 0; (timeElapsedSince(start_time) < time); i++)
        {
            if (i && (static_cast<int>(i)%50000 == 0))
            {
//                std::cout << "Unique expressions = " << Board::expression_dict.size() << '\n';
                if (check_point_score == max_score)
                {
                    T = std::min(T*10.0f, T_max);
                }
                else
                {
                    T = std::max(T/10.0f, T_min);
                }
                check_point_score = max_score;
            }
            for (int jdx = 0; jdx < x.num_objectives; jdx++)
            {
                rand_depths[jdx] = rand_depth_dists[jdx](generator);
            }
            Perturbation(rand_depths, i);
        }
    };
    //Starting the threads each with a separate version of `func`
    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads[i] = std::thread(func); //TODO: (maybe) provide a depth argument to func to specify if different threads should focus on different depth expressions (and modify the search functions accordingly)?
    }

    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads[i].join();
    }

    std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
    std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
    std::cout << "Best score = " << max_score << ", MSE = " << best_MSE << '\n';
    std::cout << "Best expression = " << best_expression << '\n';
    std::cout << "Best expression (original format) = " << orig_expression << '\n';
    std::cout << "Best diff result = " << best_expr_result << '\n';
    std::cout << "Best expression (original format) = " << orig_expr_result << '\n';
}
////
//////https://arxiv.org/abs/2310.06609
//void GP(std::vector<std::string> (*diffeq)(Board&), size_t num_diff_eqns, const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true, double time = 120, unsigned int num_threads = 0, bool const_tokens = false, float isConstTol = 1e-1f)
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
//    std::atomic<float> max_score{0.0};
//    std::atomic<float> best_MSE{FLT_MAX};
//    std::string best_expression, orig_expression, best_expr_result, orig_expr_result;
//
//    auto start_time = Clock::now();
//
//    /*
//     Inside of thread:
//     */
//
//    auto func = [&diffeq, &depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &best_expr_result, &orig_expr_result, &const_tokens, &isConstTol, &best_MSE]()
//    {
//        std::random_device rand_dev;
//        std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
//        Board x(diffeq, true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol);
//        sync_point.arrive_and_wait();
//        Board secondary_one(diffeq, false, (depth > 0) ? depth-1 : 0, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol), secondary_two(diffeq, false, (depth > 0) ? depth-1 : 0, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol); //For crossover and mutations
//        float score = 0.0f, mut_prob = 0.8f, rand_mut_cross;
//        constexpr int init_population = 2000;
//        std::vector<std::pair<std::vector<std::string>, float>> individuals;
//        std::pair<std::vector<std::string>, float> individual_1, individual_2;
//        std::vector<std::pair<int, int>> sub_exprs_1, sub_exprs_2;
//        individuals.reserve(2*init_population);
//        std::vector<std::string> temp_legal_moves;
//        std::uniform_int_distribution<int> rand_depth_dist(0, x.n - 1), selector_dist(0, init_population - 1);
//        int rand_depth, rand_individual_idx_1, rand_individual_idx_2;
//        std::uniform_real_distribution<float> rand_mut_cross_dist(0.0f, 1.0f);
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
//                best_MSE = x.MSE_curr;
//                best_expression = x._to_infix();
//                orig_expression = x.expression();
//                best_expr_result = x._to_infix(x.diffeq_result);
//                orig_expr_result = x.expression(x.diffeq_result);
//                std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
//                std::cout << "Best score = " << score << ", MSE = " << best_MSE << '\n';
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
//            [](std::pair<std::vector<std::string>, float>& individual_1, std::pair<std::vector<std::string>, float>& individual_2)
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
//    std::cout << "Best score = " << max_score << ", MSE = " << best_MSE << '\n';
//    std::cout << "Best expression = " << best_expression << '\n';
//    std::cout << "Best expression (original format) = " << orig_expression << '\n';
//    std::cout << "Best diff result = " << best_expr_result << '\n';
//    std::cout << "Best expression (original format) = " << orig_expr_result << '\n';
//}
//
//void PSO(std::vector<std::string> (*diffeq)(Board&), size_t num_diff_eqns, const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true, double time = 120, unsigned int num_threads = 0, bool const_tokens = false, float isConstTol = 1e-1f)
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
//    std::atomic<float> max_score{0.0};
//    std::atomic<float> best_MSE{FLT_MAX};
//    std::string best_expression, orig_expression, best_expr_result, orig_expr_result;
//
//    auto start_time = Clock::now();
//
//    /*
//     Inside of thread:
//     */
//
//    auto func = [&diffeq, &depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &best_expr_result, &orig_expr_result, &const_tokens, &isConstTol, &best_MSE]()
//    {
//        std::random_device rand_dev;
//        std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
//        Board x(diffeq, true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol);
//
//        sync_point.arrive_and_wait();
//        float score = 0, check_point_score = 0;
//        std::vector<std::string> temp_legal_moves;
//
//        size_t temp_sz;
//    //    std::string expression, orig_expression, best_expression;
//
//        /*
//         For this setup, we don't know a-priori the number of particles, so we generate them and their corresponding velocities as needed
//         */
//        std::vector<float> particle_positions, best_positions, v, curr_positions;
//        particle_positions.reserve(x.reserve_amount); //stores record of all current particle position indices
//        best_positions.reserve(x.reserve_amount); //indices corresponding to best pieces
//        curr_positions.reserve(x.reserve_amount); //indices corresponding to x.pieces
//        v.reserve(x.reserve_amount); //stores record of all current particle velocities
//        float rp, rg, new_v, c = 0.0f;
//        int c_count = 0;
//        std::unordered_map<float, std::unordered_map<int, int>> Nsa;
//        std::unordered_map<float, std::unordered_map<int, float>> Psa;
//        std::unordered_map<int, float> p_i_vals, p_i;
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
//                    std::uniform_real_distribution<float> temp(-c_count, c_count);
//    //                std::cout << "c: " << c << " -> ";
//                    c = temp(generator);
//    //                std::cout << c << '\n';
//                }
//                else
//                {
//    //                std::cout << "c: " << c << " -> ";
//                    c = 0.0f; //if new best found, reset c and try to exploit the new best
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
//                v[i] = copysign(std::min(new_v, FLT_MAX), new_v);
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
//                best_MSE = x.MSE_curr;
//                best_expression = x._to_infix();
//                orig_expression = x.expression();
//                best_expr_result = x._to_infix(x.diffeq_result);
//                orig_expr_result = x.expression(x.diffeq_result);
//                std::cout << "Best score = " << score << ", MSE = " << best_MSE << '\n';
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
//    std::cout << "Best score = " << max_score << ", MSE = " << best_MSE << '\n';
//    std::cout << "Best expression = " << best_expression << '\n';
//    std::cout << "Best expression (original format) = " << orig_expression << '\n';
//    std::cout << "Best diff result = " << best_expr_result << '\n';
//    std::cout << "Best expression (original format) = " << orig_expr_result << '\n';
//}
//
////https://arxiv.org/abs/2205.13134
//void ConcurrentMCTS(std::vector<std::string> (*diffeq)(Board&), size_t num_diff_eqns, const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true, double time = 120, unsigned int num_threads = 0, bool const_tokens = false, float isConstTol = 1e-1f)
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
//    std::atomic<float> max_score{0.0f};
//    std::atomic<float> best_MSE{FLT_MAX};
//
//    std::string best_expression, orig_expression, best_expr_result, orig_expr_result;
//
//    auto start_time = Clock::now();
//
//    /*
//     Inside of thread:
//     */
//
//    boost::concurrent_flat_map<std::string, boost::concurrent_flat_map<std::string, float>> Qsa;
//    boost::concurrent_flat_map<std::string, boost::concurrent_flat_map<std::string, int>> Nsa;
//    boost::concurrent_flat_map<std::string, int> Ns;
//
//    auto func = [&diffeq, &depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &best_expr_result, &orig_expr_result, &const_tokens, &isConstTol, &best_MSE, &Qsa, &Nsa, &Ns]()
//    {
//        std::random_device rand_dev;
//        std::mt19937 thread_local generator(rand_dev());
//        Board x(diffeq, true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol);
//
//        sync_point.arrive_and_wait();
//        float score = 0.0f, check_point_score = 0.0f, UCT, UCT_best;
//        std::string best_act;
//
//        std::vector<std::string> temp_legal_moves;
//        std::string state;
//
//        float c = 1.4f; //"controls the balance between exploration and exploitation", see equation 2 here: https://web.engr.oregonstate.edu/~afern/classes/cs533/notes/uct.pdf, top of page 8 here: https://arxiv.org/pdf/1402.6028.pdf, first formula in section 4. Experiments here: https://cesa-bianchi.di.unimi.it/Pubblicazioni/ml-02.pdf
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
////                for (float i: temp_legal_moves)
////                {
////                    assert(i >= 0.0f);
////                }
////                    auto start_time = Clock::now();
//                getString();
////                    str_convert_time += timeElapsedSince(start_time);
//                UCT = 0.0f;
//                UCT_best = -FLT_MAX;
//                best_act = temp_legal_moves[0];
//                std::vector<std::string> best_acts;
//                best_acts.reserve(temp_legal_moves.size());
//
//                for (const std::string& a : temp_legal_moves)
//                {
////                    assert(a > -1.0f);
////                    boost::concurrent_flat_map<std::string, boost::concurrent_flat_map<float, float>>
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
//                            float Qsa_s_a;
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
//                               x.second.insert_or_assign(a, 0.0f);
//                            });
//                            Ns.insert_or_assign(state, 0);
//                            best_acts.push_back(a);
//                            UCT = -FLT_MAX;
//                        }
//                    }
//                    else
//                    {
//                        Nsa.insert_or_assign(state, boost::concurrent_flat_map<std::string, int>({{a, 0}}));
//                        Qsa.insert_or_assign(state, boost::concurrent_flat_map<std::string, float>({{a, 0.0f}}));
//                        Ns.insert_or_assign(state, 0);
//                        best_acts.push_back(a);
//                        UCT = -FLT_MAX;
//                    }
//
//                    if (UCT > UCT_best)
//                    {
//                        best_act = a;
//                        UCT_best = UCT;
//                    }
//                }
////                assert(best_acts.size() || (best_act > -1.0f));
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
//                        x.second.insert_or_assign(state_action.second, 0.0f);
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
//                best_MSE = x.MSE_curr;
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
//    std::cout << "Best score = " << max_score << ", MSE = " << best_MSE << '\n';
//    std::cout << "Best expression = " << best_expression << '\n';
//    std::cout << "Best expression (original format) = " << orig_expression << '\n';
//    std::cout << "Best diff result = " << best_expr_result << '\n';
//    std::cout << "Best expression (original format) = " << orig_expr_result << '\n';
//}
//
////https://arxiv.org/abs/2205.13134
//void MCTS(std::vector<std::string> (*diffeq)(Board&), size_t num_diff_eqns, const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true, double time = 120, unsigned int num_threads = 0, bool const_tokens = false, float isConstTol = 1e-1f)
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
//    std::atomic<float> max_score{0.0};
//    std::atomic<float> best_MSE{FLT_MAX};
//    std::string best_expression, orig_expression, best_expr_result, orig_expr_result;
//
//    auto start_time = Clock::now();
//
//    /*
//     Inside of thread:
//     */
//
//    auto func = [&diffeq, &depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &best_expr_result, &orig_expr_result, &const_tokens, &isConstTol, &best_MSE]()
//    {
//        std::random_device rand_dev;
//        std::mt19937 thread_local generator(rand_dev());
//        Board x(diffeq, true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol);
//
//        sync_point.arrive_and_wait();
//        float score = 0.0f, check_point_score = 0.0f, UCT, UCT_best;
//        std::string best_act;
//
//        std::vector<std::string> temp_legal_moves;
//        std::unordered_map<std::string, std::unordered_map<std::string, float>> Qsa, Nsa;
//        std::unordered_map<std::string, float> Ns;
//        std::string state;
//
//        float c = 1.4f; //"controls the balance between exploration and exploitation", see equation 2 here: https://web.engr.oregonstate.edu/~afern/classes/cs533/notes/uct.pdf, top of page 8 here: https://arxiv.org/pdf/1402.6028.pdf, first formula in section 4. Experiments here: https://cesa-bianchi.di.unimi.it/Pubblicazioni/ml-02.pdf
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
//                UCT = 0.0f;
//                UCT_best = -FLT_MAX;
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
//                        UCT = -FLT_MAX;
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
//                best_MSE = x.MSE_curr;
//                best_expression = x._to_infix();
//                orig_expression = x.expression();
//                best_expr_result = x._to_infix(x.diffeq_result);
//                orig_expr_result = x.expression(x.diffeq_result);
//                std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
//                std::cout << "Best score = " << score << ", MSE = " << best_MSE << '\n';
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
//    std::cout << "Best score = " << max_score << ", MSE = " << best_MSE << '\n';
//    std::cout << "Best expression = " << best_expression << '\n';
//    std::cout << "Best expression (original format) = " << orig_expression << '\n';
//    std::cout << "Best diff result = " << best_expr_result << '\n';
//    std::cout << "Best expression (original format) = " << orig_expr_result << '\n';
//}

void RandomSearch(std::vector<std::vector<std::string>> (*diffeq)(Board&), size_t num_diff_eqns, const Eigen::MatrixXf& data, const std::vector<int>& depth, const std::string expression_type = "prefix", size_t num_consts_diff = 0, const std::string method = "LevenbergMarquardt", const int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", const bool cache = true, const double time = 120.0 /*time to run the algorithm in seconds*/, unsigned int num_threads = 0, bool const_tokens = false, float isConstTol = 1e-1f, bool use_const_pieces = false, int numDataCols = 0)
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

    std::atomic<float> max_score{0.0};
    std::atomic<float> best_MSE{FLT_MAX};
    std::string best_expression, orig_expression, best_expr_result, orig_expr_result;

    auto start_time = Clock::now();

    /*
     Inside of thread:
     */

    auto func = [&diffeq, &num_diff_eqns, &depth, &expression_type, &num_consts_diff, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &best_expr_result, &orig_expr_result, &const_tokens, &use_const_pieces, &numDataCols, &isConstTol, &best_MSE]()
    {
        std::random_device rand_dev;
        std::mt19937 thread_local generator(rand_dev()); // Mersenne Twister random number generator

        Board x(diffeq, num_diff_eqns, true, depth, expression_type, num_consts_diff, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol, use_const_pieces, true, numDataCols);

        sync_point.arrive_and_wait();
        float score = 0.0f;
        std::vector<std::string> temp_legal_moves;
        size_t temp_sz;

        int n_count = 0;
//
        while ((timeElapsedSince(start_time) < time))
        {
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

            if (Board::expression_dict.size() > n_count)
            {
                std::scoped_lock str_lock(Board::thread_locker);
                std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
                n_count += 1000000;
            }
            if (score > max_score)
            {
                max_score = score;
                std::scoped_lock str_lock(Board::thread_locker);
                best_MSE = x.MSE_curr;
                best_expression = x._to_infix();
                orig_expression = x.expression();
                best_expr_result = x._to_infix(x.diffeq_result);
                orig_expr_result = x.expression(x.diffeq_result);
                std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
                std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
                std::cout << "Best score = " << score << ", MSE = " << best_MSE << '\n';
                std::cout << "Best expression = " << best_expression << '\n';
                std::cout << "Best expression (original format) = " << orig_expression << '\n';
                std::cout << "Best diff result = " << best_expr_result << '\n';
                std::cout << "Best expression (original format) = " << orig_expr_result << '\n';
                std::cout << "Best differential equation parameters = " << x.print_diff_params() << '\n';
                std::cout << "Best expression parameters = " << x.print_expression_params() << '\n';
                std::cout << "Total system result = " << best_expr_result << '\n';
                std::cout << "Total system result (original format) = " << orig_expr_result << '\n';
            }

            for (int jdx = 0; jdx < x.pieces.size(); jdx++)
            {
                x.pieces[jdx].clear();
            }
        }
    };

    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads[i] = std::thread(func);
    }

    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads[i].join();
    }

    std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
    std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
    std::cout << "Best score = " << max_score << ", MSE = " << best_MSE << '\n';
    std::cout << "Best expression = " << best_expression << '\n';
    std::cout << "Best expression (original format) = " << orig_expression << '\n';
    std::cout << "Best diff result = " << best_expr_result << '\n';
    std::cout << "Best expression (original format) = " << orig_expr_result << '\n';
}

int main()
{
    constexpr double time = 1000000;
//    float threshold = 0.0223f;
//    auto data = createMeshgridVectors(101, 1, {0.0001f}, {10.0f});
//    RandomSearch(VortexRadialProfile /*differential equation to solve*/, 3 /*number of equations in differential equation system*/, data /*data used to solve differential equation*/, std::vector<int>{7} /*fixed depths of generated solution*/, "postfix" /*expression representation*/, 0 /*num_consts_diff: number of constants in differential equation*/, "LevenbergMarquardt" /*fit method if expression contains const tokens*/, 5 /*number of fit iterations*/, "naive_numerical" /*method for computing the gradient*/, true /*cache*/, time /*time to run the algorithm in seconds*/, 0 /*num threads*/, true /*`const_tokens`: whether to include const tokens {0, 1, 2, 4}*/, threshold /*threshold for which solutions cannot be constant*/, true /*whether or not to include constant tokens in the generated expressions, independent of the num_consts_diff tokens in the differential equation you are trying to solve*/, 0 /*number of data columns that constitute labels and not independent variables/features*/);
    auto data1 = createMeshgridVectors(33, 2, {0.0001f, 0.0f}, {10.0f, 6.28319f});
    float threshold = 1.0f;
//    RandomSearch(SwiftHohenberg /*differential equation to solve*/, 1 /*number of equations in differential equation system*/, data1 /*data used to solve differential equation*/, std::vector<int>{4} /*fixed depths of generated solution*/, "postfix" /*expression representation*/, 0 /*num_consts_diff: number of constants in differential equation*/, "LevenbergMarquardt" /*fit method if expression contains const tokens*/, 5 /*number of fit iterations*/, "naive_numerical" /*method for computing the gradient*/, true /*cache*/, time /*time to run the algorithm in seconds*/, 0 /*num threads*/, true /*`const_tokens`: whether to include const tokens {0, 1, 2, 4}*/, threshold /*threshold for which solutions cannot be constant*/, true /*whether or not to include constant tokens in the generated expressions, independent of the num_consts_diff tokens in the differential equation you are trying to solve*/, 0 /*number of data columns that constitute labels and not independent variables/features*/);
    SimulatedAnnealing(SwiftHohenberg /*differential equation to solve*/,
       1 /*number of equations in differential equation system*/,
       data1 /*data used to solve differential equation*/,
       std::vector<int>{6} /*fixed depths of generated solution*/,
       "postfix" /*expression representation*/,
       0 /*num_consts_diff: number of constants in differential equation*/,
       "LevenbergMarquardt" /*fit method if expression contains const tokens*/,
       5 /*number of fit iterations*/,
       "naive_numerical" /*method for computing the gradient*/,
       true /*cache*/,
       time /*time to run the algorithm in seconds*/,
       1 /*num threads*/,
       true /*`const_tokens`: whether to include const tokens {0, 1, 2, 4}*/,
       threshold /*threshold for which solutions cannot be constant*/,
       false /*whether or not to include constant tokens in the generated expressions, independent of the num_consts_diff tokens in the differential equation you are trying to solve*/,
       false /*Whether to simplify the expression on every iteration (perturbation) of the seed expression vector*/,
       0 /*number of data columns that constitute labels and not independent variables/features*/,
       {split("6.283190 10.000000 * ~ exp x1 cos x0 sqrt + ^")} /*seed expressions*/,
       false /*whether to exit right after computing the score for the seed epxression (default `false`)*/);
    
//    Eigen::MatrixXf data(127, 3);
//    data << -10.43428828745382, 0.0012964163037524623, -0.0010561600135938624, -10.317688181744055, 0.0012964163037524623, -0.0010590449646629705, -10.201088076034292, 0.0012964163037524623, -0.001061929915732083, -10.084487970324528, 0.0012964163037524623, -0.0010807860775096723, -9.967887864614763, 0.0012964163037524623, -0.0011318227224482134, -8.91848691322689, 0.0012964163037524623, -0.0010936643774922638, -8.801886807517125, 0.0012964163037524623, -0.0010965493285613778, -8.68528670180736, 0.0012964163037524623, -0.0010994342796304824, -8.568686596097598, 0.0012964163037524623, -0.0010915881967886473, -8.481236516815274, 0.0012964163037524623, -0.0010761774391480077, -7.460985591854841, 0.0012964163037524623, -0.0007317424693321756, -7.344385486145076, 0.0012964163037524623, -0.0006616950327002919, -7.2277853804353125, 0.0012964163037524623, -0.0006260437338381488, -7.111185274725549, 0.0012964163037524623, -0.0005903924349760059, -7.023735195443225, 0.0012964163037524623, -0.0005636539608293984, -6.003484270482792, 0.0012964163037524623, -0.0002904584970633939, -5.886884164773028, 0.0012964163037524623, -0.0002933434481325109, -5.770284059063264, 0.0012964163037524623, -0.00029622839920161436, -5.6536839533535, 0.0012964163037524623, -0.0002634502321286223, -5.566233874071178, 0.0012964163037524623, -0.00014361535924022599, -5.04153339837724, -0.0023812961476865346, 0.0005753938780901555, -4.487682896255862, -0.0023812961476865346, 0.0025630321446362203, -4.371082790546097, -0.0023812961476865346, 0.003170470816595319, -4.254482684836333, -0.004220152373406005, 0.0039049987818705924, -4.13788257912657, -0.0023812961476865346, 0.004744215952619693, -3.1370650051177638, -0.0477397497154341, 0.015162386082458477, -3.0301815748838123, -0.04651384556495444, 0.017571736480597892, -3.0301815748838123, -0.059385839144990904, 0.017571736480597892, -2.9135814691740496, -0.0612246953707104, 0.020200118733113566, -2.9135814691740496, -0.0722578327250274, 0.020200118733113566, -2.826131389891726, -0.0722578327250274, 0.022171405422500352, -2.272280887770348, -0.20097776852539212, 0.03380538107006965, -2.204264159439653, -0.2193663307825871, 0.03703478194098867, -2.1556807820605837, -0.23959374926550153, 0.039341496848788016, -2.1265307556331425, -0.22672175568546504, 0.04072552579346761, -1.864180517786174, -0.3664748288401468, 0.05165624361142156, -1.864180517786174, -0.3811856786459028, 0.05165624361142156, -1.835030491358733, -0.40692966580597567, 0.0529119057852572, -1.7961637894554787, -0.3958965284516587, 0.05212629254610853, -1.7767304385038507, -0.3738302537430248, 0.05316534620334737, -1.7767304385038507, -0.42899594051460965, 0.05316534620334737, -1.7378637366005965, -0.41673689900981303, 0.05538744310479808, -1.5143802006568823, -0.5503604514120964, 0.06859392065655061, -1.5143802006568823, -0.5632324449921329, 0.06859392065655061, -1.456080147802, -0.5797821510236083, 0.07148952152552111, -1.4269301213745589, -0.6092038506351203, 0.07068320834571716, -1.4269301213745589, -0.5650713012178524, 0.07068320834571716, -1.3880634194713046, -0.5944930008293643, 0.07399308257283442, -1.2520299628099139, -0.6643695374067052, 0.07771025027470406, -1.096563155196895, -0.7342460739840461, 0.08765946352014711, -1.0382631023420146, -0.75631234869268, 0.09068147600343156, -1.0188297513903866, -0.7342460739840461, 0.09032323761866505, -0.9605296985355043, -0.763667773595558, 0.09139473856823228, -0.9313796721080632, -0.7765397671755945, 0.0949841086018029, -0.34837914355924404, -0.8813545720416058, 0.1078952792642798, -0.2900790907043618, -0.8923877093959227, 0.1081109856961209, -0.2317790378494795, -0.8850322844930447, 0.1083264085954597, -0.11517893213971497, -0.8887099969444838, 0.10825685757412504, -0.11517893213971497, -0.8997431342988007, 0.10825685757412504, 0.0014211735700477846, -0.8960654218473618, 0.10889242358867447, 0.4095215435542219, -0.8776768595901667, 0.10071037481903938, 0.7301718342560726, -0.8261888852700209, 0.09222130882213057, 0.7593218606835137, -0.8133168916899844, 0.0907023965524053, 0.817621913538396, -0.8004448981099479, 0.09121769317212448, 0.8467719399658353, -0.8151557479157039, 0.09057423883080643, 0.8759219663932765, -0.7857340483041919, 0.08956661658439113, 0.9050719928207176, -0.8004448981099479, 0.08778230764296704, 0.9439386947239718, -0.7710231984984359, 0.08540322905440165, 0.9633720456755999, -0.7894117607556309, 0.08421368976011888, 1.2840223363774506, -0.6423032626980713, 0.07096278110376601, 1.2840223363774506, -0.6551752562781077, 0.07096278110376601, 1.2937390118532637, -0.6239147004408763, 0.07102632037452986, 1.352039064708146, -0.6018484257322423, 0.06793381671873283, 1.371472415659774, -0.6239147004408763, 0.06690298216680045, 1.4297724685146562, -0.5944930008293643, 0.06381047851100341, 1.4297724685146562, -0.6055261381836813, 0.06381047851100341, 1.6046726270793013, -0.4657730650289996, 0.0563485619345576, 1.6435393289825555, -0.45473992767468263, 0.054811139292982716, 1.7018393818374378, -0.43267365296604865, 0.049947734849127005, 1.7212727327890658, -0.45473992767468263, 0.05004382699481444, 1.779572785643948, -0.41796280316029266, 0.04866441212888443, 1.779572785643948, -0.42899594051460965, 0.04866441212888443, 1.954472944208593, -0.28924286735992794, 0.04236496314010472, 1.9933396461118473, -0.27820973000561094, 0.040965085587042566, 2.0710730499183576, -0.256143455296977, 0.0381653304809182, 2.0710730499183576, -0.270854305102733, 0.0381653304809182, 2.0710730499183576, -0.28188744245704994, 0.0381653304809182, 2.109939751821612, -0.2402067013407414, 0.03676545292785605, 2.1876731556281204, -0.23039946813690404, 0.03396569782173175, 2.333423287765326, -0.1715560689138802, 0.030854610209602847, 2.6346402275155505, -0.10167953233653931, 0.021605693132999516, 2.712373631322059, -0.10167953233653931, 0.01973791820879841, 2.7415236577495, -0.08696868253078335, 0.019037502612222985, 2.7998237106043806, -0.0722578327250274, 0.017636671419072176, 2.8289737370318218, -0.08329097007934436, 0.01693625582249675, 3.7326245562824916, -0.013414433502003498, 0.007472265835869461, 3.849224661992256, -0.006059008599125504, 0.006376637533005264, 3.849224661992256, -0.017092145953442495, 0.006376637533005264, 3.9658247677020206, -0.006059008599125504, 0.005281009230141067, 4.082424873411785, -0.006059008599125504, 0.004185380927276871, 4.199024979121546, -0.006059008599125504, 0.0038337565701724525, 4.315625084831311, -0.006059008599125504, 0.002806815055583041, 4.432225190541075, -0.0005424399219670362, 0.0018962403086712788, 4.54882529625084, 0.0012964163037524623, 0.0022326809651833437, 4.665425401960604, 0.0012964163037524623, 0.0014163423316503833, 4.752875481242928, 0.0012964163037524623, 0.001872028283842192, 5.77312640620336, -0.0005424399219670362, -0.0005818385550432711, 5.889726511913125, 0.0012964163037524623, -0.000584723506112388, 6.006326617622889, 0.0012964163037524623, -0.0005876084571814913, 6.12292672333265, 0.0012964163037524623, -0.0006079755495703998, 6.239526829042415, 0.0012964163037524623, -0.0006811290130572356, 7.23062772757541, 0.0012964163037524623, -0.00136232556390265, 7.347227833285174, 0.0012964163037524623, -0.0015817304103852947, 7.463827938994935, 0.0012964163037524623, -0.0018011352568679273, 7.5804280447047, 0.0012964163037524623, -0.001939549782349283, 7.667878123987023, 0.0012964163037524623, -0.0019254043838014701, 8.163428573253519, 0.0012964163037524623, -0.0017486845884493267, 8.688129048947456, 0.0012964163037524623, -0.0015615695110176451, 8.80472915465722, 0.0012964163037524623, -0.001650762588452242, 8.921329260366985, 0.0012964163037524623, -0.001896787283102471, 9.03792936607675, 0.0012964163037524623, -0.001610327152429707, 9.154529471786514, 0.0012964163037524623, -0.0015408317932039549, 9.65007992105301, 0.0012964163037524623, -0.0015530928352476675, 10.145630370319505, 0.0012964163037524623, -0.00156535387729138, 10.26223047602927, 0.0012964163037524623, -0.0015682388283604879, 10.378830581739034, 0.0012964163037524623, -0.0015711237794295918, 10.466280661021358, -0.0023812961476865346, -0.0015732874927314232;
//    std::cout << "data = " << data << '\n';

//    RandomSearch(SolitonWaveFengEq14and15Laser /*differential equation to solve*/, 9 /*number of equations in differential equation system*/, data /*data used to solve differential equation*/, std::vector<int>{4, 4} /*fixed depths of generated solution*/, "postfix" /*expression representation*/, 1 /*num_consts_diff: number of constants in differential equation*/, "LevenbergMarquardt" /*fit method if expression contains const tokens*/, 5 /*number of fit iterations*/, "naive_numerical" /*method for computing the gradient*/, true /*cache*/, time /*time to run the algorithm in seconds*/, 0 /*num threads*/, true /*`const_tokens`: whether to include const tokens {0, 1, 2, 4}*/, threshold /*threshold for which solutions cannot be constant*/, true /*whether or not to include constant tokens in the generated expressions, independent of the num_consts_diff tokens in the differential equation you are trying to solve*/, 2 /*number of data columns that constitute labels and not independent variables/features*/);
//    SimulatedAnnealing(SolitonWaveFengEq14and15Laser /*differential equation to solve*/,
//       9 /*number of equations in differential equation system*/,
//       data /*data used to solve differential equation*/,
//       std::vector<int>{4, 4} /*fixed depths of generated solution*/,
//       "postfix" /*expression representation*/,
//       1 /*num_consts_diff: number of constants in differential equation*/,
//       "LevenbergMarquardt" /*fit method if expression contains const tokens*/,
//       5 /*number of fit iterations*/,
//       "naive_numerical" /*method for computing the gradient*/,
//       true /*cache*/,
//       time /*time to run the algorithm in seconds*/,
//       0 /*num threads*/,
//       true /*`const_tokens`: whether to include const tokens {0, 1, 2, 4}*/,
//       threshold /*threshold for which solutions cannot be constant*/,
//       true /*whether or not to include constant tokens in the generated expressions, independent of the num_consts_diff tokens in the differential equation you are trying to solve*/,
//       false /*Whether to simplify the expression on every iteration (perturbation) of the seed expression vector*/,
//       2 /*number of data columns that constitute labels and not independent variables/features*/,
//       {split("x0 sech tanh tanh 4 -10.434288 + x0 tanh 1 exp / - /"), split("10.466281 sqrt x0 sech 2 tanh ^ * sech")} /*seed expressions*/,
//       false /*whether to exit right after computing the score for the seed epxression (default `false`)*/);

    return 0;
}

//git push --set-upstream origin PrefixPostfixSymbolicDifferentiator

//g++ -Wall -std=c++20 -o PrefixPostfixMultiThreadDiffSimplifySR_Nd PrefixPostfixMultiThreadDiffSimplifySR_Nd.cpp -O2 -I/opt/homebrew/opt/eigen/include/eigen3 -I/opt/homebrew/opt/eigen/include/eigen3 -I/Users/edwardfinkelstein/LBFGSpp -L/opt/homebrew/Cellar/boost/1.84.0 -I/opt/homebrew/Cellar/boost/1.84.0/include -march=native

//g++ -Wall -std=c++20 -o PrefixPostfixMultiThreadDiffSimplifySR_Nd PrefixPostfixMultiThreadDiffSimplifySR_Nd.cpp -g -I/opt/homebrew/opt/eigen/include/eigen3 -I/opt/homebrew/opt/eigen/include/eigen3 -I/Users/edwardfinkelstein/LBFGSpp -L/opt/homebrew/Cellar/boost/1.84.0 -I/opt/homebrew/Cellar/boost/1.84.0/include -march=native

//git push --set-upstream origin PrefixPostfixSymbolicDifferentiator

//C:\msys64\ucrt64\bin\g++.exe -std=c++1z -IC:\Users\finkelsteine\test_codes\eigen\ -IC:\Users\finkelsteine\test_codes\eigen\unsupported -IC:\Users\finkelsteine\test_codes\boost_1_88_0 -IC:\Users\finkelsteine\test_codes\LBFGSpp\include -c C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp -o C:\Users\finkelsteine\test_codes\hello_with_numbers.o
//C:\msys64\ucrt64\bin\g++.exe  -o C:\Users\finkelsteine\test_codes\hello_with_numbers.exe C:\Users\finkelsteine\test_codes\hello_with_numbers.o  -O2
//C:\msys64\ucrt64\bin\g++.exe  -o C:\Users\finkelsteine\test_codes\hello_with_numbers.exe C:\Users\finkelsteine\test_codes\hello_with_numbers.o  -g

//To unzip file: Expand-Archive -Path "C:\Users\finkelsteine\test_codes\boost_1_88_0.zip" -DestinationPath "C:\Users\finkelsteine\test_codes"
//To count how many instances of a string (in this case "stof" occur in a file (in this case `hello.cpp`):
// - (Get-Content -Path "C:\Users\finkelsteine\test_codes\hello.cpp" | Select-String -Pattern "stof").Count
//To launch Python: C:\Users\finkelsteine\AppData\Local\Programs\Python\Launcher\py.exe
//To get the diff between two files: Compare-Object (Get-Content -Path "C:\Users\finkelsteine\test_codes\hello_with_numbers.cpp") (Get-Content -Path "C:\Users\finkelsteine\test_codes\hello_with_numbers.txt")
//To install with pip: C:\Users\finkelsteine\AppData\Local\Programs\Python\Launcher\py.exe -m pip install plotdigitizer
//To change the path variable, do $env:Path="newpath"


//SimulatedAnnealing(SwiftHohenberg /*differential equation to solve*/,
//   1 /*number of equations in differential equation system*/,
//   data1 /*data used to solve differential equation*/,
//   std::vector<int>{4} /*fixed depths of generated solution*/,
//   "postfix" /*expression representation*/,
//   0 /*num_consts_diff: number of constants in differential equation*/,
//   "LevenbergMarquardt" /*fit method if expression contains const tokens*/,
//   5 /*number of fit iterations*/,
//   "naive_numerical" /*method for computing the gradient*/,
//   true /*cache*/,
//   time /*time to run the algorithm in seconds*/,
//   1 /*num threads*/,
//   true /*`const_tokens`: whether to include const tokens {0, 1, 2, 4}*/,
//   threshold /*threshold for which solutions cannot be constant*/,
//   false /*whether or not to include constant tokens in the generated expressions, independent of the num_consts_diff tokens in the differential equation you are trying to solve*/,
//   false, /*Whether to simplify the expression on every iteration (perturbation) of the seed expression vector*/
//   0 /*number of data columns that constitute labels and not independent variables/features*/,
//                   {},//{split("6.283190 10.000000 * ~ exp x1 cos x0 sqrt + ^")} /*seed expressions*/,
//   false /*whether to exit right after computing the score for the seed epxression (default `false`)*/);

//void SimulatedAnnealing(std::vector<std::vector<std::string>> (*diffeq)(Board&), size_t num_diff_eqns, const Eigen::MatrixXd& data, const std::vector<int>& depth, const std::string expression_type = "prefix", size_t num_consts_diff = 0, const std::string method = "LevenbergMarquardt", const int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", const bool cache = true, const double time = 120.0 /*time to run the algorithm in seconds*/, unsigned int num_threads = 0, bool const_tokens = false, double isConstTol = 1e-1f, bool use_const_pieces = false, bool simplifyOriginal = false, int numDataCols = 0, const std::vector<std::vector<std::string>>& seed_expressions = {}, bool exit_early = false)
//auto func = [&diffeq, &num_diff_eqns, &depth, &expression_type, &num_consts_diff, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &best_expr_result, &orig_expr_result, &const_tokens, &isConstTol, &use_const_pieces, &simplifyOriginal, &numDataCols, &seed_expressions, &exit_early, &best_MSE]()
//Board x(diffeq, num_diff_eqns, true, depth, expression_type, num_consts_diff, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol, use_const_pieces, simplifyOriginal, numDataCols);
//sync_point.arrive_and_wait();
//Board secondary(diffeq, num_diff_eqns, false, std::vector<int>(depth.size(), 0), expression_type, num_consts_diff, method, num_fit_iter, fit_grad_method, data, false, cache, const_tokens, isConstTol, use_const_pieces, simplifyOriginal, numDataCols); //For perturbations
//TODO: Hide this file from git commit history so it doesn't show publicly. So find a way to have git back it up but not make it publicly visible like all the other files in the repo.
