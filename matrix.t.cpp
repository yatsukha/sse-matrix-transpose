#include "matrix.hpp"
#include "catch.hpp"

#include <type_traits>

TEST_CASE("matrix constructor and property test") {
  ::pr::matrix<int, 5, 5> m0;

  REQUIRE(5 == m0.rows);
  REQUIRE(5 == m0.cols);

  ::pr::matrix<int, 5, 6> m1;

  REQUIRE(5 == m1.rows);
  REQUIRE(6 == m1.cols);

  REQUIRE(false == ::std::is_same_v<decltype(m0), decltype(m1)>);
}

TEST_CASE("matrix indexing test") {
  ::pr::matrix<int, 5, 5> m0;

  for (::std::size_t i = 0; i < m0.rows; ++i)
    for (::std::size_t j = 0; j < m0.cols; ++j)
      m0[i][j] = static_cast<int>(i * j);

  ::pr::matrix<int, 5, 5> const& m1 = m0;

  for (::std::size_t i = 0; i < m0.rows; ++i)
    for (::std::size_t j = 0; j < m0.cols; ++j)
      REQUIRE(i * j == m1[i][j]);
}

TEST_CASE("row alignment test") {
  ::pr::matrix<int, 256, 256> m;

  for (::std::size_t i = 0; i < m.rows; ++i)
    REQUIRE(0 == reinterpret_cast<::std::size_t>(m[i]) % 16);
}