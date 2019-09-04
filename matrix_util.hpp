#pragma once

#include "matrix.hpp"
#include <emmintrin.h>

namespace pr {

  template<typename T, ::std::size_t R, ::std::size_t C>
  matrix<T, C, R> transpose(matrix<T, R, C> const& m) {
    matrix<T, C, R> transposed;

    for (::std::size_t i = 0; i < R; ++i)
      for (::std::size_t j = 0; j < C; ++j)
        transposed[j][i] = m[i][j];

    return transposed;
  }

  namespace sse_transpose_impl {
    template<typename T, ::std::size_t R, ::std::size_t C>
    matrix<T, C, R> sse_4pack_transpose(matrix<T, R, C> const& m) {
      matrix<T, C, R> transposed;

      ::std::size_t const maxI = (m.rows / 4) * 4;
      ::std::size_t const maxJ = (m.cols / 4) * 4;

      std::cout << m.cols << std::endl;
      std::cout << "m: " << maxI << " " << maxJ << std::endl;

      ::__m128 r0, r1, r2, r3;

      for (::std::size_t i = 0; i < maxI; i += 4) {
        for (::std::size_t j = 0; j < maxJ; j += 4) {
          r0 = _mm_load_ps(reinterpret_cast<float const*>(&m[i    ][j]));
          r1 = _mm_load_ps(reinterpret_cast<float const*>(&m[i + 1][j]));
          r2 = _mm_load_ps(reinterpret_cast<float const*>(&m[i + 2][j]));
          r3 = _mm_load_ps(reinterpret_cast<float const*>(&m[i + 3][j]));

          _MM_TRANSPOSE4_PS(r0, r1, r2, r3);

          _mm_store_ps(reinterpret_cast<float*>(&transposed[j    ][i]), r0);
          _mm_store_ps(reinterpret_cast<float*>(&transposed[j + 1][i]), r1);
          _mm_store_ps(reinterpret_cast<float*>(&transposed[j + 2][i]), r2);
          _mm_store_ps(reinterpret_cast<float*>(&transposed[j + 3][i]), r3);
        }

        for (::std::size_t k = 0; k < 4; ++k)
          for (::std::size_t j = maxJ; j < m.cols; ++j)
            transposed[j][i + k] = m[i + k][j];
      }

      for (::std::size_t i = maxI; i < m.rows; ++i)
        for (::std::size_t j = 0; j < m.cols; ++j)
          transposed[j][i] = m[i][j];

      return transposed;
    }
  }

  template<typename T, ::std::size_t R, ::std::size_t C>
  matrix<T, C, R> sse_transpose(matrix<T, R, C> const& m) {
    constexpr ::std::size_t data_size = sizeof(T);  

    if constexpr (4 == data_size)
      return sse_transpose_impl::sse_4pack_transpose(m);
    /*else if constexpr (8 == data_size)
      return sse_transpose_impl::sse_2pack_transpose(m);
    else if constexpr (16 == data_size)
      return sse_transpose_impl::sse_1pack_transpose(m);
    else
      static_assert(false, "invalid data type for sse transpose");*/
  }

}