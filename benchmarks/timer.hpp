#pragma once

#include <chrono>

namespace pr {
  
  template<typename Clock, typename Duration>
  class timer {
    ::std::chrono::time_point<Clock> current;

    public:
      timer() noexcept
        : current(Clock::now()) {}

      void start() noexcept {
        current = Clock::now();
      }

      Duration get() noexcept {
        return ::std::chrono::duration_cast<Duration>(Clock::now() - current);
      }
  };

  template<typename Duration>
  using steady_timer = timer<::std::chrono::steady_clock, Duration>;

  template<typename Duration>
  using system_timer = timer<::std::chrono::system_clock, Duration>;

}