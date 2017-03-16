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
#ifdef CNN_USE_DOUBLE
TEST(avx, sse_double) {
  typedef __m128d register_type;
  register_type r1;
  register_type r2;
  register_type r3;
  vectorize::detail::double_sse fsse1;
  vectorize::detail::double_sse fsse2;
  double a1 = 1e-50;
  double a2 = 2e-50;
  double a3 = 3e-100;
  r1        = fsse1.set1(a1);
  r2        = fsse1.set1(a2);
  r3        = fsse1.set1(a3);
  std::cout << "SSE " << fsse2.resemble(fsse1.madd(r1, r2, r3)) << std::endl;
  std::cout << "CPU " << ((a1 * a2) + a3) << std::endl;
  a1 = 1e-2;
  a2 = 2e-2;
  a3 = 3e-4;
  r1 = fsse1.set1(a1);
  r2 = fsse1.set1(a2);
  r3 = fsse1.set1(a3);
  std::cout << "SSE " << fsse2.resemble(fsse1.madd(r1, r2, r3)) << std::endl;
  std::cout << "CPU " << ((a1 * a2) + a3) << std::endl;
  /// EXPECT_NEAR(1., 1., AVXEPS);
}
#else
TEST(avx, sse_float) {
  typedef __m128 register_type;
  register_type r1;
  register_type r2;
  register_type r3;
  vectorize::detail::float_sse fsse1;
  vectorize::detail::float_sse fsse2;
  float a1 = 1e-16;
  float a2 = 2e-16;
  float a3 = 3e-32;
  r1       = fsse1.set1(a1);
  r2       = fsse1.set1(a2);
  r3       = fsse1.set1(a3);
  std::cout << "SSE " << fsse2.resemble(fsse1.madd(r1, r2, r3)) << std::endl;
  std::cout << "CPU " << ((a1 * a2) + a3) << std::endl;
  a1 = 1e-2;
  a2 = 2e-2;
  a3 = 3e-4;
  r1 = fsse1.set1(a1);
  r2 = fsse1.set1(a2);
  r3 = fsse1.set1(a3);
  std::cout << "SSE " << fsse2.resemble(fsse1.madd(r1, r2, r3)) << std::endl;
  std::cout << "CPU " << ((a1 * a2) + a3) << std::endl;
  /// EXPECT_NEAR(1., 1., AVXEPS);
  /// EXPECT_NEAR(1., 1., AVXEPS);
}
#endif
#endif

#ifdef CNN_USE_AVX
#ifdef CNN_USE_DOUBLE
TEST(avx, avx_double) {
  typedef __m128d register_type;
  register_type r1;
  register_type r2;
  register_type r3;
  vectorize::detail::double_avx favx1;
  vectorize::detail::double_avx favx2;
  double a1 = 1e-50;
  double a2 = 2e-50;
  double a3 = 3e-100;
  r1        = favx1.set1(a1);
  r2        = favx1.set1(a2);
  r3        = favx1.set1(a3);
  std::cout << "AVX " << favx2.resemble(favx1.madd(r1, r2, r3)) << std::endl;
  std::cout << "CPU " << (a1 * a2 + a3) << std::endl;
  a1 = 1e-2;
  a2 = 2e-2;
  a3 = 3e-4;
  r1 = favx1.set1(a1);
  r2 = favx1.set1(a2);
  r3 = favx1.set1(a3);
  std::cout << "AVX " << favx2.resemble(favx1.madd(r1, r2, r3)) << std::endl;
  std::cout << "CPU " << ((a1 * a2) + a3) << std::endl;
  /// EXPECT_NEAR(1., 1., AVXEPS);
}
#else
TEST(avx, avx_float) {
  typedef __m128 register_type;
  register_type r1;
  register_type r2;
  register_type r3;
  vectorize::detail::float_avx favx1;
  vectorize::detail::float_avx favx2;
  float a1 = 1e-16;
  float a2 = 2e-16;
  float a3 = 3e-32;
  r1       = favx1.set1(a1);
  r2       = favx1.set1(a2);
  r3       = favx1.set1(a3);
  std::cout << "AVX " << favx2.resemble(favx1.madd(r1, r2, r3)) << std::endl;
  std::cout << "CPU " << ((a1 * a2) + a3) << std::endl;
  a1 = 1e-2;
  a2 = 2e-2;
  a3 = 3e-4;
  r1 = favx1.set1(a1);
  r2 = favx1.set1(a2);
  r3 = favx1.set1(a3);
  std::cout << "AVX " << favx2.resemble(favx1.madd(r1, r2, r3)) << std::endl;
  std::cout << "CPU " << ((a1 * a2) + a3) << std::endl;
  /// EXPECT_NEAR(1., 1., AVXEPS);
}

#endif
#endif  // CNN_USE_AVX

}  // namespace tiny-dnn
