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

      for (::std::size_t i = 0; i != max_i; i += 4) {
        for (::std::size_t j = 0; j != max_j; j += 4)  {
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

        for (::std::size_t k = 0; k != 4; ++k)
          for (::std::size_t j = max_j; j != m.cols; ++j)
            transposed[j][i + k] = m[i + k][j];
      }

      for (::std::size_t i = max_i; i != m.rows; ++i)
        for (::std::size_t j = 0; j != m.cols; ++j)
          transposed[j][i] = m[i][j];

      return transposed;
    }

    template<typename T, ::std::size_t R, ::std::size_t C>
    matrix<T, C, R> sse_2pack_transpose(matrix<T, R, C> const& m) {
      matrix<T, C, R> transposed;

      ::__m128d r0, r1;

      ::std::size_t const max_i = (m.rows / 2) * 2;
      ::std::size_t const max_j = (m.cols / 2) * 2;

      if (max_j != m.cols)
        for (::std::size_t i = 0; i < max_i; i += 2) {
          for (::std::size_t j = 0; j < max_j; j += 2) {
            r0 = ::_mm_load_pd(reinterpret_cast<double const*>(&m[i    ][j]));
            r1 = ::_mm_load_pd(reinterpret_cast<double const*>(&m[i + 1][j]));

            ::_mm_store_pd(reinterpret_cast<double*>(
              &transposed[j][i]), 
              ::_mm_shuffle_pd(r0, r1, 0b00)
            );

            ::_mm_store_pd(reinterpret_cast<double*>(
              &transposed[j + 1][i]), 
              ::_mm_shuffle_pd(r0, r1, 0b11)
            );
          }

          for (::std::size_t k = 0; k < 2; ++k)
            for (::std::size_t j = max_j; j < m.cols; ++j)
              transposed[j][i + k] = m[i + k][j];
        }
      else
        for (::std::size_t i = 0; i < max_i; i += 2) {
          for (::std::size_t j = 0; j < max_j; j += 2) {
            r0 = ::_mm_load_pd(reinterpret_cast<double const*>(&m[i    ][j]));
            r1 = ::_mm_load_pd(reinterpret_cast<double const*>(&m[i + 1][j]));

            ::_mm_store_pd(reinterpret_cast<double*>(
              &transposed[j][i]), 
              ::_mm_shuffle_pd(r0, r1, 0b00)
            );

            ::_mm_store_pd(reinterpret_cast<double*>(
              &transposed[j + 1][i]), 
              ::_mm_shuffle_pd(r0, r1, 0b11)
            );
          }
        }

      if (max_i != m.rows)
        for (::std::size_t j = 0; j < m.cols; ++j)
          transposed[j][max_i] = m[max_i][j];

      return transposed;
    }

    template<typename T, ::std::size_t R, ::std::size_t C>
    matrix<T, C, R> sse_1pack_transpose(matrix<T, R, C> const& m) {
      matrix<T, C, R> transposed;

      for (::std::size_t i = 0; i < m.rows; ++i)
        for (::std::size_t j = 0; j < m.cols; ++j)
          ::_mm_store_ps(
            reinterpret_cast<float*>(&transposed[j][i]), 
            ::_mm_load_ps(reinterpret_cast<float const*>(&m[i][j]))
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

  template<typename T, ::std::size_t R, ::std::size_t C>
    matrix<T, C, R> cache_friendly_transpose(matrix<T, R, C> const& m) {
      matrix<T, C, R> transposed;

      ::std::size_t const max_i = (m.rows / 4) * 4;
      ::std::size_t const max_j = (m.cols / 4) * 4;

      for (::std::size_t i = 0; i != max_i; i += 4) {
        for (::std::size_t j = 0; j != max_j; j += 4)
          for (::std::size_t k = 0; k != 4; ++k)
            for (::std::size_t l = 0; l != 4; ++l)
              transposed[j + l][i + k] = m[i + k][j + l];

        for (::std::size_t k = 0; k != 4; ++k)
          for (::std::size_t j = max_j; j < m.cols; ++j)
            transposed[j][i + k] = m[i + k][j];
      }

      for (::std::size_t i = max_i; i != m.rows; ++i)
        for (::std::size_t j = 0; j != m.cols; ++j)
          transposed[j][i] = m[i][j];

      return transposed;
    }

    // reference transpose

    template<typename T, ::std::size_t R, ::std::size_t C>
    void transpose(matrix<T, R, C> const& m, matrix<T, C, R>& transposed) noexcept {
      for (::std::size_t i = 0; i < R; ++i)
        for (::std::size_t j = 0; j < C; ++j)
          transposed[j][i] = m[i][j];
    }

    namespace sse_transpose_impl {
      template<typename T, ::std::size_t R, ::std::size_t C>
      void sse_4pack_transpose(matrix<T, R, C> const& m, matrix<T, C, R>& transposed) noexcept {

        ::std::size_t const max_i = (m.rows / 4) * 4;
        ::std::size_t const max_j = (m.cols / 4) * 4;

        ::__m128 r0, r1, r2, r3;

        for (::std::size_t i = 0; i != max_i; i += 4) {
          for (::std::size_t j = 0; j != max_j; j += 4)  {
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

          for (::std::size_t k = 0; k != 4; ++k)
            for (::std::size_t j = max_j; j != m.cols; ++j)
              transposed[j][i + k] = m[i + k][j];
        }

        for (::std::size_t i = max_i; i != m.rows; ++i)
          for (::std::size_t j = 0; j != m.cols; ++j)
            transposed[j][i] = m[i][j];
      }

      template<typename T, ::std::size_t R, ::std::size_t C>
      void sse_2pack_transpose(matrix<T, R, C> const& m, matrix<T, C, R>& transposed) noexcept {
        ::__m128d r0, r1;

        ::std::size_t const max_i = (m.rows / 2) * 2;
        ::std::size_t const max_j = (m.cols / 2) * 2;

        if (max_j != m.cols)
          for (::std::size_t i = 0; i < max_i; i += 2) {
            for (::std::size_t j = 0; j < max_j; j += 2) {
              r0 = ::_mm_load_pd(reinterpret_cast<double const*>(&m[i    ][j]));
              r1 = ::_mm_load_pd(reinterpret_cast<double const*>(&m[i + 1][j]));

              ::_mm_store_pd(reinterpret_cast<double*>(
                &transposed[j][i]), 
                ::_mm_shuffle_pd(r0, r1, 0b00)
              );

              ::_mm_store_pd(reinterpret_cast<double*>(
                &transposed[j + 1][i]), 
                ::_mm_shuffle_pd(r0, r1, 0b11)
              );
            }

            for (::std::size_t k = 0; k < 2; ++k)
              for (::std::size_t j = max_j; j < m.cols; ++j)
                transposed[j][i + k] = m[i + k][j];
          }
        else
          for (::std::size_t i = 0; i < max_i; i += 2) {
            for (::std::size_t j = 0; j < max_j; j += 2) {
              r0 = ::_mm_load_pd(reinterpret_cast<double const*>(&m[i    ][j]));
              r1 = ::_mm_load_pd(reinterpret_cast<double const*>(&m[i + 1][j]));

              ::_mm_store_pd(reinterpret_cast<double*>(
                &transposed[j][i]), 
                ::_mm_shuffle_pd(r0, r1, 0b00)
              );

              ::_mm_store_pd(reinterpret_cast<double*>(
                &transposed[j + 1][i]), 
                ::_mm_shuffle_pd(r0, r1, 0b11)
              );
            }
          }

        if (max_i != m.rows)
          for (::std::size_t j = 0; j < m.cols; ++j)
            transposed[j][max_i] = m[max_i][j];
      }

      template<typename T, ::std::size_t R, ::std::size_t C>
      void sse_1pack_transpose(matrix<T, R, C> const& m, matrix<T, C, R>& transposed) noexcept {
        for (::std::size_t i = 0; i < m.rows; ++i)
          for (::std::size_t j = 0; j < m.cols; ++j)
            ::_mm_store_ps(
              reinterpret_cast<float*>(&transposed[j][i]), 
              ::_mm_load_ps(reinterpret_cast<float const*>(&m[i][j]))
            );
      }
    }

    template<typename T, ::std::size_t R, ::std::size_t C>
    void sse_transpose(matrix<T, R, C> const& src, matrix<T, C, R>& dest) noexcept {
      constexpr ::std::size_t data_size = sizeof(T);  

      if constexpr (4 == data_size)
        sse_transpose_impl::sse_4pack_transpose(src, dest);
      else if constexpr (8 == data_size)
        sse_transpose_impl::sse_2pack_transpose(src, dest);
      else if constexpr (16 == data_size)
        sse_transpose_impl::sse_1pack_transpose(src, dest);
      else
        static_assert(false, "invalid data type for sse transpose");
    }

}