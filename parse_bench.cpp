#include <boost/spirit/include/qi.hpp>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm> 

using clk = std::chrono::high_resolution_clock;

static bool parse_double_spirit(const std::string& s, double& out) {
    namespace qi   = boost::spirit::qi;
    namespace ascii= boost::spirit::ascii;

    auto f = s.begin(), l = s.end();
    // Skip leading/trailing ASCII whitespace; require full consumption (eoi).
    // qi::double_ already yields ±inf/NaN where appropriate.
    bool ok = qi::phrase_parse(f, l, qi::double_ >> qi::eoi, ascii::space, out);
    return ok; // f==l guaranteed by eoi
}

static bool parse_double_stod(const std::string& s, double& out) {
    // std::stod throws on error; avoid exceptions in the hot path by prechecking.
    // Quick precheck: require at least one digit; this still lets stod decide.
    bool has_digit = std::any_of(s.begin(), s.end(), ::isdigit);
    if (!has_digit) return false;

    try {
        size_t idx = 0;
        out = std::stod(s, &idx);
        return idx == s.size(); // ensure full consumption
    } catch (...) {
        return false;
    }
}

int main(int argc, char** argv) {
    assert(false);
    // Small corpus of “typical” numeric strings, plus some failures
    std::vector<std::string> samples = {
        "0", "1", "-1", "3.1415926535", "-2.718281828",
        "6.02214076e23", "1e-308", "1.7976931348623157e308",
        "0.0", "-0.0", "+42.0", "  123.456", "7.5e+09",
        "nan", "inf", "-inf",
        "bad", "123abc", "", "  ", "--1"
    };

    // Repeat count (default ~10 million parses total if 500k * 20 samples)
    size_t repeats = (argc > 1) ? std::stoull(argv[1]) : 500'000ULL;

    // Shuffle to reduce branch predictability
    std::mt19937_64 rng(12345);
    std::shuffle(samples.begin(), samples.end(), rng);

    // WARMUP
    volatile double sink = 0.0;
    for (int i = 0; i < 1000; ++i) {
        for (auto& s : samples) { double x; parse_double_spirit(s, x); sink += x; }
        for (auto& s : samples) { double x; parse_double_stod  (s, x); sink += x; }
    }

    // Measure Boost Spirit
    auto t0 = clk::now();
    for (size_t r = 0; r < repeats; ++r) {
        for (auto& s : samples) {
            double x;
            (void)parse_double_spirit(s, x);
            sink += x;
        }
    }
    auto t1 = clk::now();

    // Measure std::stod
    for (int i = 0; i < 500; ++i) { // small spacer to reduce turbo/p-state bias
        asm volatile("" ::: "memory");
    }

    auto t2 = clk::now();
    for (size_t r = 0; r < repeats; ++r) {
        for (auto& s : samples) {
            double x;
            (void)parse_double_stod(s, x);
            sink += x;
        }
    }
    auto t3 = clk::now();

    auto ns_spirit = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    auto ns_stod   = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();

    const size_t total_parses = repeats * samples.size();
    std::cout << "Boost Spirit: " << ns_spirit / 1e6 << " ms  ("
              << (double)ns_spirit / total_parses << " ns/parse)\n";
    std::cout << "std::stod  : " << ns_stod   / 1e6 << " ms  ("
              << (double)ns_stod   / total_parses << " ns/parse)\n";

    // Prevent optimizing everything away
    std::cerr << "sink=" << sink << "\n";
    return 0;
}

//g++ -std=c++20 -O2 -march=native -DNDEBUG parse_bench.cpp -o parse_bench -L/opt/homebrew/Cellar/boost/1.84.0 -I/opt/homebrew/Cellar/boost/1.84.0/include
