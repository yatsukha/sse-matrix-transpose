#include "timer.hpp"
#include "../matrix_util.hpp"

#include <vector>
#include <iostream>
#include <functional>
#include <random>
#include <cmath>
#include <type_traits>
#include <numeric>
#include <locale>

namespace pr {

  using result_type = ::std::vector<::std::pair<::std::chrono::nanoseconds, double>>;
  
  result_type measure(::std::vector<::std::function<void(void)>>& tasks, ::std::size_t repetitions) {

    result_type results(repetitions, result_type::value_type{0, 0.0});
    steady_timer<result_type::value_type::first_type> timer;

    for (::std::size_t r = 0; r < repetitions; ++r) {
      for (::std::size_t i = 0; i < tasks.size(); ++i) {
        timer.start();

        tasks[i]();

        auto time = timer.get();

        results[i].first  += time;
        results[i].second += (time.count() * time.count()) / (long long) (repetitions - 1);
      }
    }

    for (auto&& result : results) {
      result.first /= repetitions;

      result.second -= 
      ((double) repetitions / (repetitions - 1)) 
        * result.first.count() * result.first.count();

      result.second = ::std::sqrt(result.second);
    }


    return results;
  }

  template<typename T,::std::size_t R, ::std::size_t C>
  matrix<T, R, C> random_matrix() {
    matrix<T, R, C> m;

    ::std::mt19937 mt{ ::std::random_device{}() };


    ::std::conditional_t<
      ::std::numeric_limits<T>::is_integer, 
      ::std::uniform_int_distribution<T>,
      ::std::uniform_real_distribution<T>
    > dist(-10'000, 10'000);

    for (::std::size_t i = 0; i < R; ++i)
      for (::std::size_t j = 0; j < C; ++j)
        m[i][j] = dist(mt);

    return m;
  }

}

::std::ostream& operator<<(::std::ostream& out, ::pr::result_type::value_type const& r) {
  out << r.first.count() / 1000.0 << " microseconds, standard deviation: " << r.second;
  return out;
}

#define R 100
#define C 1000
#define TYPE double

int main() noexcept {
  auto m = ::pr::random_matrix<TYPE, R, C>();
  auto t = ::pr::matrix<TYPE, C, R>{};

  ::std::vector<::std::function<void(void)>> tasks{
    [&]{ ::pr::transpose(m, t); },
    [&]{ ::pr::sse_transpose(m, t); }
  };

  auto r = ::pr::measure(tasks, 1000);

  ::std::cout << "naive: " << r[0] << ::std::endl;
  ::std::cout << "sse:   " << r[1] << ::std::endl;
}