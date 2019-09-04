#include "matrix_util.hpp"
#include "catch.hpp"

TEMPLATE_TEST_CASE("matrix transpose", "", int, float) {
  ::pr::matrix<TestType, 15, 17> m;

  for (::std::size_t i = 0; i < m.rows; ++i)
    for (::std::size_t j = 0; j < m.cols; ++j)
      m[i][j] = static_cast<TestType>(i * (j - i));

  auto transposed = ::pr::transpose(m);

  REQUIRE(17 == transposed.rows);
  REQUIRE(15 == transposed.cols);

  for (::std::size_t i = 0; i < m.rows; ++i)
    for (::std::size_t j = 0; j < m.cols; ++j)
      REQUIRE((i * (j - i)) == (transposed[j][i]));
}

TEMPLATE_TEST_CASE("sse matrix transpose", "", /*4 bytes ->*/ int, float, ::std::size_t, /*8 bytes ->*/ double, ::std::uint_fast64_t) {
  ::pr::matrix<TestType, 19, 13> m;

  for (::std::size_t i = 0; i < m.rows; ++i)
    for (::std::size_t j = 0; j < m.cols; ++j)
      m[i][j] = static_cast<TestType>(i * i * j);

  auto transposed = ::pr::sse_transpose(m);

  for (::std::size_t i = 0; i < m.rows; ++i)
    for (::std::size_t j = 0; j < m.cols; ++j)
      CHECK((i * i * j) == (transposed[j][i]));
}

TEST_CASE("sse 16 byte transpose") {
  struct data {
    ::std::uint_fast64_t a;
    ::std::uint_fast64_t b;
  };

  REQUIRE(16 == sizeof(data));

  ::pr::matrix<data, 31, 13> m;

  for (::std::size_t i = 0; i < m.rows; ++i)
    for (::std::size_t j = 0; j < m.cols; ++j)
      m[i][j] = {i, j};

  auto transposed = ::pr::sse_transpose(m);

  for (::std::size_t i = 0; i < m.rows; ++i)
    for (::std::size_t j = 0; j < m.cols; ++j) {
      CHECK(i == transposed[j][i].a);
      CHECK(j == transposed[j][i].b);
    }
}