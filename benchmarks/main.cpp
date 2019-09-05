#include "timer.hpp"
#include "../matrix_util.hpp"

#include <vector>
#include <iostream>
#include <functional>
#include <random>
#include <cmath>
#include <type_traits>
#include <numeric>

namespace pr {

  using result_type = ::std::vector<::std::pair<::std::chrono::microseconds, double>>;
  
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

int main() noexcept {

  auto m_inv = ::pr::random_matrix<int, 255, (1 << 16) + 3>();

  ::std::vector<::std::function<void(void)>> tasks = {
    [m_inv]{ ::pr::transpose(m_inv); },
    [m_inv]{ ::pr::sse_transpose(m_inv); }
  };

  auto results = ::pr::measure(tasks, 5);

  ::std::cout << "-- column intensive --" << ::std::endl;
  ::std::cout << "regular transpose: " << results[0].first.count() << " " << results[0].second << ::std::endl;
  ::std::cout << "sse transpose:     " << results[1].first.count() << " " << results[1].second << ::std::endl;

  auto m = ::pr::random_matrix<int, (1 << 16) + 3, 255>();

  tasks = {
    [m]{ ::pr::transpose(m); },
    [m]{ ::pr::sse_transpose(m); }
  };

  results = ::pr::measure(tasks, 5);

  ::std::cout << "-- row intensive --" << ::std::endl;
  ::std::cout << "regular transpose: " << results[0].first.count() << " " << results[0].second << ::std::endl;
  ::std::cout << "sse transpose:     " << results[1].first.count() << " " << results[1].second << ::std::endl;
}