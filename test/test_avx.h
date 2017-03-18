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

// auxiliary function to compute relative differences in parts per billion (PPB)
void ppb_print(double reference,
               double measured,
               std::string ref,
               std::string mea) {
  auto maxv = std::max(std::abs(reference), std::abs(measured));
  auto diff = std::abs(reference - measured);
  auto ppb  = static_cast<size_t>(1e9f * static_cast<float>(diff / maxv));
  std::printf("%s %.20f   %s %.20f EVM %9zud ppb", ref.c_str(), reference,
              mea.c_str(), measured, ppb);
}
// test for AVX backends

inline void randomize_tensor_avx(tensor_t &tensor) {
  for (auto &vec : tensor) {
    uniform_rand(vec.begin(), vec.end(), -1.0f, 1.0f);
  }
}

// prepare tensor buffers for unit test
class tensor_buf_avx {
 public:
  tensor_buf_avx(tensor_buf_avx &other)
    : in_data_(other.in_data_), out_data_(other.out_data_) {
    for (auto &d : in_data_) in_ptr_.push_back(&d);
    for (auto &d : out_data_) out_ptr_.push_back(&d);
  }

  explicit tensor_buf_avx(const layer &l, bool randomize = true)
    : in_data_(l.in_channels()),
      out_data_(l.out_channels()),
      in_ptr_(l.in_channels()),
      out_ptr_(l.out_channels()) {
    for (size_t i = 0; i < l.in_channels(); i++) {
      in_data_[i].resize(1, vec_t(l.in_shape()[i].size()));
      in_ptr_[i] = &in_data_[i];
    }

    for (size_t i = 0; i < l.out_channels(); i++) {
      out_data_[i].resize(1, vec_t(l.out_shape()[i].size()));
      out_ptr_[i] = &out_data_[i];
    }

    if (randomize) {
      for (auto &tensor : in_data_) randomize_tensor_avx(tensor);
      for (auto &tensor : out_data_) randomize_tensor_avx(tensor);
    }
  }

  tensor_t &in_at(size_t i) { return in_data_[i]; }
  tensor_t &out_at(size_t i) { return out_data_[i]; }

  std::vector<tensor_t *> &in_buf() { return in_ptr_; }
  std::vector<tensor_t *> &out_buf() { return out_ptr_; }

 private:
  std::vector<tensor_t> in_data_, out_data_;
  std::vector<tensor_t *> in_ptr_, out_ptr_;
};

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
  //  std::cout << "SSE " << ans;
  //  std::cout << " CPU " << ((a1 * a2) + a3) << std::endl;
  EXPECT_EQ(ans, ((a1 * a2) + a3));
  a1 = 1e-2;
  a2 = 2e-2;
  a3 = 3e-4;
  r1 = dsse1.set1(a1);
  r2 = dsse1.set1(a2);
  r3 = dsse1.set1(a3);
  dsse2.store(&ans, dsse1.madd(r1, r2, r3));
  //  std::cout << "SSE " << ans;
  //  std::cout << " CPU " << ((a1 * a2) + a3) << std::endl;
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
  //  std::cout << "SSE " << ans;
  //  std::cout << " CPU " << ((a1 * a2) + a3) << std::endl;
  EXPECT_EQ(ans, ((a1 * a2) + a3));
  a1 = 1e-2;
  a2 = 2e-2;
  a3 = 3e-4;
  r1 = fsse1.set1(a1);
  r2 = fsse1.set1(a2);
  r3 = fsse1.set1(a3);
  fsse2.store(&ans, fsse1.madd(r1, r2, r3));
  //  std::cout << "SSE " << ans;
  //  std::cout << " CPU " << ((a1 * a2) + a3) << std::endl;
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
  //  std::cout << "AVX " << ans;
  //  std::cout << " CPU " << ((a1 * a2) + a3) << std::endl;
  EXPECT_EQ(ans, ((a1 * a2) + a3));
  a1 = 1e-2;
  a2 = 2e-2;
  a3 = 3e-4;
  r1 = davx1.set1(a1);
  r2 = davx1.set1(a2);
  r3 = davx1.set1(a3);
  davx2.store(&ans, davx1.madd(r1, r2, r3));
  //  std::cout << "AVX " << ans;
  //  std::cout << " CPU " << ((a1 * a2) + a3) << std::endl;
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
  //  std::cout << "AVX " << ans;
  //  std::cout << " CPU " << ((a1 * a2) + a3) << std::endl;
  EXPECT_EQ(ans, ((a1 * a2) + a3));
  a1 = 1e-2;
  a2 = 2e-2;
  a3 = 3e-4;
  r1 = favx1.set1(a1);
  r2 = favx1.set1(a2);
  r3 = favx1.set1(a3);
  favx2.store(&ans, favx1.madd(r1, r2, r3));
  // std::cout << "AVX " << ans;
  // std::cout << " CPU " << ((a1 * a2) + a3) << std::endl;
  EXPECT_EQ(ans, ((a1 * a2) + a3));
}

TEST(avx, fprop) {
  convolutional_layer<identity> l(7, 7, 5, 1, 1);

  tensor_buf_avx buf(l), buf2(l);

  l.set_backend_type(tiny_dnn::core::backend_t::internal);

  l.forward_propagation(buf.in_buf(), buf.out_buf());

  l.set_backend_type(tiny_dnn::core::backend_t::avx);

  l.forward_propagation(buf.in_buf(), buf2.out_buf());

  vec_t &out_avx   = buf2.out_at(0)[0];
  vec_t &out_noavx = buf.out_at(0)[0];

  for (size_t i = 0; i < out_avx.size(); i++) {
    ppb_print(out_noavx[i], out_avx[i], "CPU", "AVX");
    // check if all outputs between default backend and avx backend are the
    // same
    // EXPECT_EQ(out_avx[i], out_noavx[i]);
  }
}
#endif  // CNN_USE_AVX

}  // namespace tiny-dnn
