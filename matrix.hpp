#pragma once

#include <utility>
#include <memory>
#include <ostream>
#include <iomanip>
#include <cstring>
#include <cassert>

// intel sse intrinsics
#include <emmintrin.h>

namespace pr {
  
  // a contigous storage matrix designed to be used with intels mmx/sse1/sse2 intrinsics
  template<typename T, ::std::size_t R, ::std::size_t C>
  class matrix {
    private:
      static_assert(R > 0, "rows must be > 0");
      static_assert(C > 0, "columns must be > 0");

      ::std::size_t constexpr static alignment = 16;
      ::std::size_t const ratio = alignment / sizeof(T);

      ::std::size_t aligned_cols;
      ::std::size_t allocated;

      T* data;

      static_assert(sizeof(T) <= alignment, "too small alignment for data type of T");

    public:
      ::std::size_t const& rows = R;
      ::std::size_t const& cols = C;

      matrix() 
        : aligned_cols(C % ratio ? (C / ratio + 1) * ratio : C)
        , allocated(sizeof(T) * R * aligned_cols)
        , data(static_cast<T*>(::_mm_malloc(allocated, alignment)))
      {}

      ~matrix() {
        ::_mm_free(data);
      }

      matrix(matrix<T, R, C>&& other) {
        operator=(std::forward<matrix<T, R, C>>(other));
      }

      matrix<T, R, C>& operator=(matrix<T, R, C>&& other) {
        aligned_cols = other.aligned_cols;
        allocated = other.allocated;
        data = other.data;

        other.data = nullptr;

        return *this;
      }

      matrix(matrix<T, R, C> const& other) {
        operator=(other);
      }

      matrix<T, R, C>& operator=(matrix<T, R, C> const& other) {
        aligned_cols = other.aligned_cols;
        allocated = other.allocated;

        data = static_cast<T*>(::_mm_malloc(allocated, alignment));
        ::std::memcpy(data, other.data, sizeof(T) * rows * aligned_cols);

        return *this;
      }

      T* operator[](::std::size_t const idx) {
        return &data[idx * aligned_cols];
      }

      T const* operator[](::std::size_t const idx) const {
        return &data[idx * aligned_cols];
      }
  };

  template<typename T, ::std::size_t R, ::std::size_t C>
  ::std::ostream& operator<<(::std::ostream& out, matrix<T, R, C> const& m) {
    for (::std::size_t i = 0; i < m.rows; ++i) {
      for (::std::size_t j = 0; j < m.cols; ++j)
        out << std::setw(6) << m[i][j] << " ";

      out << std::endl;
    }

    return out;
  }

}