/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include "gtest/gtest.h"
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

// test for AVX backends

#ifdef CNN_USE_SSE
TEST(avx, sse_double) {
  typedef __m128d register_type;
  vectorize::detail::double_sse dsse1;
  vectorize::detail::double_sse dsse2;
  double a1 = 1e-50;
  double a2 = 2e-50;
  double a3 = 3e-100;
  double ans;
  register_type r1 = dsse1.set1(a1);
  register_type r2 = dsse1.set1(a2);
  register_type r3 = dsse1.set1(a3);
  dsse2.store(&ans, dsse1.madd(r1, r2, r3));
  std::cout << "SSE " << ans;
  std::cout << " CPU " << ((a1 * a2) + a3) << std::endl;
  EXPECT_EQ(ans, ((a1 * a2) + a3));
  a1 = 1e-2;
  a2 = 2e-2;
  a3 = 3e-4;
  r1 = dsse1.set1(a1);
  r2 = dsse1.set1(a2);
  r3 = dsse1.set1(a3);
  dsse2.store(&ans, dsse1.madd(r1, r2, r3));
  std::cout << "SSE " << ans;
  std::cout << " CPU " << ((a1 * a2) + a3) << std::endl;
  EXPECT_EQ(ans, ((a1 * a2) + a3));
}

TEST(avx, sse_float) {
  typedef __m128 register_type;
  vectorize::detail::float_sse fsse1;
  vectorize::detail::float_sse fsse2;
  float a1 = 1e-15;
  float a2 = 2e-15;
  float a3 = 3e-30;
  float ans;
  register_type r1 = fsse1.set1(a1);
  register_type r2 = fsse1.set1(a2);
  register_type r3 = fsse1.set1(a3);
  fsse2.store(&ans, fsse1.madd(r1, r2, r3));
  std::cout << "SSE " << ans;
  std::cout << " CPU " << ((a1 * a2) + a3) << std::endl;
  EXPECT_EQ(ans, ((a1 * a2) + a3));
  a1 = 1e-2;
  a2 = 2e-2;
  a3 = 3e-4;
  r1 = fsse1.set1(a1);
  r2 = fsse1.set1(a2);
  r3 = fsse1.set1(a3);
  fsse2.store(&ans, fsse1.madd(r1, r2, r3));
  std::cout << "SSE " << ans;
  std::cout << " CPU " << ((a1 * a2) + a3) << std::endl;
  EXPECT_EQ(ans, ((a1 * a2) + a3));
}
#endif

#ifdef CNN_USE_AVX
TEST(avx, avx_double) {
  typedef __m256d register_type;
  vectorize::detail::double_avx davx1;
  vectorize::detail::double_avx davx2;
  double a1 = 1e-50;
  double a2 = 2e-50;
  double a3 = 3e-100;
  double ans;
  register_type r1 = davx1.set1(a1);
  register_type r2 = davx1.set1(a2);
  register_type r3 = davx1.set1(a3);
  davx2.store(&ans, davx1.madd(r1, r2, r3));
  std::cout << "AVX " << ans;
  std::cout << " CPU " << ((a1 * a2) + a3) << std::endl;
  EXPECT_EQ(ans, ((a1 * a2) + a3));
  a1 = 1e-2;
  a2 = 2e-2;
  a3 = 3e-4;
  r1 = davx1.set1(a1);
  r2 = davx1.set1(a2);
  r3 = davx1.set1(a3);
  davx2.store(&ans, davx1.madd(r1, r2, r3));
  std::cout << "AVX " << ans;
  std::cout << " CPU " << ((a1 * a2) + a3) << std::endl;
  EXPECT_EQ(ans, ((a1 * a2) + a3));
}
TEST(avx, avx_float) {
  typedef __m256 register_type;
  vectorize::detail::float_avx favx1;
  vectorize::detail::float_avx favx2;
  float a1 = 1e-15;
  float a2 = 2e-15;
  float a3 = 3e-30;
  float ans;
  register_type r1 = favx1.set1(a1);
  register_type r2 = favx1.set1(a2);
  register_type r3 = favx1.set1(a3);
  favx2.store(&ans, favx1.madd(r1, r2, r3));
  std::cout << "AVX " << ans;
  std::cout << " CPU " << ((a1 * a2) + a3) << std::endl;
  EXPECT_EQ(ans, ((a1 * a2) + a3));
  a1 = 1e-2;
  a2 = 2e-2;
  a3 = 3e-4;
  r1 = favx1.set1(a1);
  r2 = favx1.set1(a2);
  r3 = favx1.set1(a3);
  favx2.store(&ans, favx1.madd(r1, r2, r3));
  std::cout << "AVX " << ans;
  std::cout << " CPU " << ((a1 * a2) + a3) << std::endl;
  EXPECT_EQ(ans, ((a1 * a2) + a3));
}
#endif  // CNN_USE_AVX

}  // namespace tiny-dnn
