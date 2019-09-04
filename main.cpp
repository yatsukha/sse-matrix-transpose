#define CATCH_CONFIG_MAIN
#include "catch.hpp"

//#include <iostream>
//#include <iomanip>
////#include <Windows.h>
//#include <emmintrin.h>
//#include <memory>
//#include <random>
//#include <chrono>
//
//#include <cassert>
//
//namespace pr {
//
//  class matrix {
//    std::shared_ptr<int> data;
//    std::size_t aligned_cols;
//
//    std::size_t static constexpr align = 16;
//    std::size_t static constexpr factor = align / sizeof(int);
//
//    static_assert(sizeof(int) == 4, "");
//
//    public:
//      std::size_t rows;
//      std::size_t cols;
//
//      matrix(std::size_t const rows, std::size_t const cols)
//        : rows(rows)
//        , cols(cols)
//        , aligned_cols(cols % factor == 0 ? cols : (cols / factor + 1) * factor)
//      {
//        data = std::shared_ptr<int>{
//          static_cast<int*>(
//              _mm_malloc(
//                sizeof(int) * rows * aligned_cols,
//                align
//              )
//            ),
//            [](int* ptr) {
//              _mm_free(ptr);
//            }
//        };
//
//        assert(data);
//      }
//
//      int* operator[](std::size_t const idx) {
//        return &data.get()[idx * aligned_cols];
//      }
//
//      const int* operator[](std::size_t const idx) const {
//        return &data.get()[idx * aligned_cols];
//      }
//  };
//
//  std::ostream& operator<<(std::ostream& out, matrix const& m) {
//    for (int i = 0; i < m.rows; ++i) {
//      for (int j = 0; j < m.cols; ++j)
//        out << std::setw(6)
//            << m[i][j];
//
//      out << std::endl;
//    }
//
//    return out;
//  }
//
//  matrix naive_transpose(matrix const& m) {
//    matrix ret(m.cols, m.rows);
//
//    for (int i = 0; i < m.rows; ++i)
//      for (int j = 0; j < m.cols; ++j)
//        ret[j][i] = m[i][j];
//
//    return ret;
//  }
//
//  matrix block_transpose(matrix const& m, std::size_t const block_size) {
//    matrix ret(m.cols, m.rows);
//
//    for (int i = 0; i < m.rows; i += block_size)
//      for (int j = 0; j < m.cols; j += block_size)
//        for (int k = i; k < i + block_size; ++k)
//          for (int l = j; l < j + block_size; ++l)
//            ret[l][k] = m[k][l];
//
//    return ret;
//  }
//
//  matrix sse_block_transpose(matrix const& m) {
//    matrix ret(m.cols, m.rows);
//
//    unsigned constexpr m0 = 0x44;
//    unsigned constexpr m1 = 0xEE;
//    unsigned constexpr m2 = 0x88;
//    unsigned constexpr m3 = 0xDD;
//
//    __m128 r0, r1, r2, r3;
//
//    int last_i_block = (m.rows / 4) * 4;
//    int last_j_block = (m.cols / 4) * 4;
//      
//    for (int i = 0; i < last_i_block; i += 4) {
//      for (int j = 0; j < last_j_block; j += 4) {
//        r0 = _mm_load_ps(reinterpret_cast<float const*>(&m[i    ][j]));
//        r1 = _mm_load_ps(reinterpret_cast<float const*>(&m[i + 1][j]));
//        r2 = _mm_load_ps(reinterpret_cast<float const*>(&m[i + 2][j]));
//        r3 = _mm_load_ps(reinterpret_cast<float const*>(&m[i + 3][j]));
//
//        _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
//
//        _mm_store_ps(reinterpret_cast<float*>(&ret[j    ][i]), r0);
//        _mm_store_ps(reinterpret_cast<float*>(&ret[j + 1][i]), r1);
//        _mm_store_ps(reinterpret_cast<float*>(&ret[j + 2][i]), r2);
//        _mm_store_ps(reinterpret_cast<float*>(&ret[j + 3][i]), r3);
//      }
//
//      for (int k = 0; k < 4; ++k)
//        for (int j = last_j_block; j < m.cols; ++j)
//          ret[j][i + k] = m[i + k][j];
//    }
//    
//    for (int i = last_i_block; i < m.rows; ++i)
//      for (int j = 0; j < m.cols; ++j)
//        ret[j][i] = m[i][j];
//
//    return ret;
//  }
//
//  matrix fill_random(
//    std::pair<std::size_t, std::size_t> const& dim, 
//    std::mt19937 generator, 
//    std::uniform_int_distribution<int> distribution
//  ) {
//    matrix m(dim.first, dim.second);
//
//    for (int i = 0; i < m.rows; ++i)
//      for (int j = 0; j < m.cols; ++j)
//        m[i][j] = distribution(generator);
//
//    return m;
//  }
//
//  class scoped_timer {
//    private:
//      std::chrono::time_point<std::chrono::steady_clock> start = 
//        std::chrono::steady_clock::now();
//
//    public:
//      scoped_timer() = default;
//      ~scoped_timer() {
//        auto end = std::chrono::steady_clock::now();
//
//        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
//                  << std::endl;
//      }
//  };
//
//}
//
//int main() noexcept {
//  auto m = 
//    pr::fill_random(
//      { 1 << 16, 256 }, 
//      std::mt19937{ std::random_device{}() }, 
//      std::uniform_int_distribution<int>(-10000, 10000)
//    );
//
//  std::cout << "starting..." << std::endl << std::endl;
//
//  for (int i = 0; i < 5; ++i) {
//    {
//      pr::scoped_timer _;
//      pr::naive_transpose(m);
//    }
//
//    {
//      pr::scoped_timer _;
//      pr::sse_block_transpose(m);
//    }
//
//    std::cout << std::endl;
//  }
//
//  std::cout << "----\n\n";
//
//  for (int i = 0; i < 5; ++i) {
//    {
//      pr::scoped_timer _;
//      pr::sse_block_transpose(m);
//    }
//
//    {
//      pr::scoped_timer _;
//      pr::naive_transpose(m);
//    }
//
//    std::cout << std::endl;
//  }
//}