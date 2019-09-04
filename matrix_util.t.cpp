#include "matrix_util.hpp"
#include "catch.hpp"

TEMPLATE_TEST_CASE("matrix transpose", "", int, float, ::std::size_t) {
  ::pr::matrix<TestType, 15, 15> m;

  for (::std::size_t i = 0; i < m.rows; ++i)
    for (::std::size_t j = 0; j < m.cols; ++j)
      m[i][j] = static_cast<TestType>(i * j);

  auto transposed = ::pr::transpose(m);

  std::cout << "aaa" << std::endl;

  for (::std::size_t i = 0; i < m.rows; ++i)
    for (::std::size_t j = 0; j < m.cols; ++j)
      REQUIRE(i * j == transposed[j][i]);
}

TEMPLATE_TEST_CASE("sse matrix transpose", "", int, float) {
  ::pr::matrix<TestType, 19, 13> m;

  for (::std::size_t i = 0; i < m.rows; ++i)
    for (::std::size_t j = 0; j < m.cols; ++j)
      m[i][j] = static_cast<TestType>(i * j);

  auto transposed = ::pr::sse_transpose(m);

  for (::std::size_t i = 0; i < m.rows; ++i)
    for (::std::size_t j = 0; j < m.cols; ++j)
      REQUIRE(i * j == transposed[j][i]);
}