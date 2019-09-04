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

      ::std::size_t const max_i = (m.rows / 4) * 4;
      ::std::size_t const max_j = (m.cols / 4) * 4;

      ::__m128 r0, r1, r2, r3;

      for (::std::size_t i = 0; i < max_i; i += 4) {
        for (::std::size_t j = 0; j < max_j; j += 4) {
          r0 = ::_mm_load_ps(reinterpret_cast<float const*>(&m[i    ][j]));
          r1 = ::_mm_load_ps(reinterpret_cast<float const*>(&m[i + 1][j]));
          r2 = ::_mm_load_ps(reinterpret_cast<float const*>(&m[i + 2][j]));
          r3 = ::_mm_load_ps(reinterpret_cast<float const*>(&m[i + 3][j]));

          _MM_TRANSPOSE4_PS(r0, r1, r2, r3);

          ::_mm_store_ps(reinterpret_cast<float*>(&transposed[j    ][i]), r0);
          ::_mm_store_ps(reinterpret_cast<float*>(&transposed[j + 1][i]), r1);
          ::_mm_store_ps(reinterpret_cast<float*>(&transposed[j + 2][i]), r2);
          ::_mm_store_ps(reinterpret_cast<float*>(&transposed[j + 3][i]), r3);
        }

        for (::std::size_t k = 0; k < 4; ++k)
          for (::std::size_t j = max_j; j < m.cols; ++j)
            transposed[j][i + k] = m[i + k][j];
      }

      for (::std::size_t i = max_i; i < m.rows; ++i)
        for (::std::size_t j = 0; j < m.cols; ++j)
          transposed[j][i] = m[i][j];

      return transposed;
    }

    template<typename T, ::std::size_t R, ::std::size_t C>
    matrix<T, C, R> sse_2pack_transpose(matrix<T, R, C> const& m) {
      matrix<T, C, R> transposed;

      ::std::size_t const max_i = (m.rows / 2) * 2;
      ::std::size_t const max_j = (m.cols / 2) * 2;

      ::__m128 r0, r1;

      for (::std::size_t i = 0; i < max_i; i += 2) {
        for (::std::size_t j = 0; j < max_j; j += 2) {
          r0 = ::_mm_load_ps(reinterpret_cast<float const*>(&m[i    ][j]));
          r1 = ::_mm_load_ps(reinterpret_cast<float const*>(&m[i + 1][j]));

          ::_mm_store_ps(reinterpret_cast<float*>(
            &transposed[j    ][i]), 
            ::_mm_shuffle_ps(r1, r0, _MM_SHUFFLE(3, 2, 1, 0))
          );

          ::_mm_store_ps(reinterpret_cast<float*>(
            &transposed[j + 1][i]), 
            ::_mm_shuffle_ps(r1, r0, _MM_SHUFFLE(1, 0, 3, 2))
          );
        }

        // TODO: better loops
        for (::std::size_t k = 0; k < 2; ++k)
          for (::std::size_t j = max_j; j < m.cols; ++j)
            transposed[j][i + k] = m[i + k][j];
      }

      for (::std::size_t i = max_i; i < m.rows; ++i)
        for (::std::size_t j = 0; j < m.cols; ++j)
          transposed[j][i] = m[i][j];

      return transposed;
    }

    template<typename T, ::std::size_t R, ::std::size_t C>
    matrix<T, C, R> sse_1pack_transpose(matrix<T, R, C> const& m) {
      matrix<T, C, R> transposed;

      for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j)
          ::_mm_store_ps(
            reinterpret_cast<float*>(&transposed[j][i]), 
            ::_mm_load_ps(reinterpret_cast<float*>(&m[i][j]))
          );

      return transposed;
    }

  }

  template<typename T, ::std::size_t R, ::std::size_t C>
  matrix<T, C, R> sse_transpose(matrix<T, R, C> const& m) {
    constexpr ::std::size_t data_size = sizeof(T);  

    if constexpr (4 == data_size)
      return sse_transpose_impl::sse_4pack_transpose(m);
    else if constexpr (8 == data_size)
      return sse_transpose_impl::sse_2pack_transpose(m);
    else if constexpr (16 == data_size)
      return sse_transpose_impl::sse_1pack_transpose(m);
    else
      static_assert(false, "invalid data type for sse transpose");
  }

}