/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "./advection.hpp"

#include <gridtools/stencil_composition/cartesian.hpp>

#include "./computation.hpp"
#include "./tridiagonal.hpp"

namespace numerics {
namespace advection {
namespace {
using gt::extent;
using gt::make_param_list;
using gt::cartesian::call;
using gt::cartesian::call_proc;
using gt::cartesian::in_accessor;
using gt::cartesian::inout_accessor;
using namespace gt::cartesian::expressions;

struct stage_u {
  using flux = inout_accessor<0>;
  using u = in_accessor<1>;
  using in = in_accessor<2, extent<-3, 3, 0, 0>>;
  using dx = in_accessor<3>;
  using param_list = make_param_list<flux, u, in, dx>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t) {
    static constexpr real_t weights[] = {1_r / 30, -1_r / 4, 1_r,
                                         -1_r / 3, -1_r / 2, 1_r / 20};

    if (eval(u()) > 0_r) {
      eval(flux()) =
          eval(u() *
               -(weights[0] * in(-3, 0, 0) + weights[1] * in(-2, 0, 0) +
                 weights[2] * in(-1, 0, 0) + weights[3] * in() +
                 weights[4] * in(1, 0, 0) + weights[5] * in(2, 0, 0)) /
               dx());
    } else if (eval(u()) < 0_r) {
      eval(flux()) =
          eval(u() *
               (weights[5] * in(-2, 0, 0) + weights[4] * in(-1, 0, 0) +
                weights[3] * in() + weights[2] * in(1, 0, 0) +
                weights[1] * in(2, 0, 0) + weights[0] * in(3, 0, 0)) /
               dx());
    } else {
      eval(flux()) = 0_r;
    }
  }
};
struct stage_v {
  using flux = inout_accessor<0>;
  using v = in_accessor<1>;
  using in = in_accessor<2, extent<0, 0, -3, 3>>;
  using dy = in_accessor<3>;

  using param_list = make_param_list<flux, v, in, dy>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t) {
    static constexpr real_t weights[] = {1_r / 30, -1_r / 4, 1_r,
                                         -1_r / 3, -1_r / 2, 1_r / 20};

    if (eval(v()) > 0_r) {
      eval(flux()) =
          eval(v() *
               -(weights[0] * in(0, -3, 0) + weights[1] * in(0, -2, 0) +
                 weights[2] * in(0, -1, 0) + weights[3] * in() +
                 weights[4] * in(0, 1, 0) + weights[5] * in(0, 2, 0)) /
               dy());
    } else if (eval(v()) < 0_r) {
      eval(flux()) =
          eval(v() *
               (weights[5] * in(0, -2, 0) + weights[4] * in(0, -1, 0) +
                weights[3] * in() + weights[2] * in(0, 1, 0) +
                weights[1] * in(0, 2, 0) + weights[0] * in(0, 3, 0)) /
               dy());
    } else {
      eval(flux()) = 0_r;
    }
  }
};

struct stage_horizontal {
  using out = inout_accessor<0>;
  using in = in_accessor<1, extent<-3, 3, -3, 3>>;
  using u = in_accessor<2>;
  using v = in_accessor<3>;

  using dx = in_accessor<4>;
  using dy = in_accessor<5>;
  using dt = in_accessor<6>;

  using param_list = make_param_list<out, in, u, v, dx, dy, dt>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t) {
    auto flx = call<stage_u, full_t>::with(eval, u(), in(), dx());
    auto fly = call<stage_v, full_t>::with(eval, v(), in(), dy());

    eval(out()) = eval(in() - dt() * (flx + fly));
  }
};

struct stage_advection_w0 {
  using data = in_accessor<0>;
  using data_top = inout_accessor<1>;

  using param_list = make_param_list<data, data_top>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
    eval(data_top()) = eval(data());
  }
};

struct stage_advection_w_forward1 {
  using alpha = inout_accessor<0>;
  using beta = inout_accessor<1>;
  using gamma = inout_accessor<2>;
  using a = inout_accessor<3>;
  using b = inout_accessor<4>;
  using c = inout_accessor<5, extent<0, 0, 0, 0, -1, 0>>;
  using d = inout_accessor<6, extent<0, 0, 0, 0, -1, 0>>;

  using data = in_accessor<7, extent<0, 0, 0, 0, -1, 1>>;
  using data_tmp = inout_accessor<8>;

  using dz = in_accessor<9>;
  using dt = in_accessor<10>;
  using w = in_accessor<11, extent<0, 0, 0, 0, 0, 1>>;

  using param_list = make_param_list<alpha, beta, gamma, a, b, c, d, data,
                                     data_tmp, dz, dt, w>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
    eval(a()) = eval(-0.25_r * w() / dz());
    eval(c()) = eval(0.25_r * w(0, 0, 1) / dz());
    eval(b()) = eval(1_r / dt() - a() - c());
    eval(d()) = eval(1_r / dt() * data() -
                     0.25_r * w(0, 0, 1) * (data(0, 0, 1) - data()) / dz() -
                     0.25_r * w() * (data() - data_tmp()) / dz());

    eval(alpha()) = eval(-a());
    eval(beta()) = eval(a());
    eval(gamma()) = eval(-b());

    call_proc<tridiagonal::periodic_forward1, full_t::first_level>::with(
        eval, a(), b(), c(), d(), alpha(), beta(), gamma());

    eval(data_tmp()) = eval(data());
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, -1>) {
    eval(a()) = eval(-0.25_r * w() / dz());
    eval(c()) = eval(0.25_r * w(0, 0, 1) / dz());
    eval(b()) = eval(1_r / dt() - a() - c());
    eval(d()) = eval(1_r / dt() * data() -
                     0.25_r * w(0, 0, 1) * (data(0, 0, 1) - data()) / dz() -
                     0.25_r * w() * (data() - data(0, 0, -1)) / dz());

    call_proc<tridiagonal::periodic_forward1, full_t::modify<1, -1>>::with(
        eval, a(), b(), c(), d(), alpha(), beta(), gamma());
  }
  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
    eval(a()) = eval(-0.25_r * w() / dz());
    eval(c()) = eval(0.25_r * w(0, 0, 1) / dz());
    eval(b()) = eval(1_r / dt() - a() - c());
    eval(d()) = eval(1_r / dt() * data() -
                     0.25_r * w(0, 0, 1) * (data_tmp() - data()) / dz() -
                     0.25_r * w() * (data() - data(0, 0, -1)) / dz());

    call_proc<tridiagonal::periodic_forward1, full_t::last_level>::with(
        eval, a(), b(), c(), d(), alpha(), beta(), gamma());
  }
};

using stage_advection_w_backward1 = tridiagonal::periodic_backward1;
using stage_advection_w_forward2 = tridiagonal::periodic_forward2;
using stage_advection_w_backward2 = tridiagonal::periodic_backward2;

struct stage_advection_w3 {
  using out = inout_accessor<0>;
  using x = in_accessor<1>;
  using z = in_accessor<2>;
  using fact = in_accessor<3>;
  using in = in_accessor<4>;

  using dt = in_accessor<5>;

  using param_list = make_param_list<out, x, z, fact, in, dt>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t) {
    call_proc<tridiagonal::periodic3, full_t>::with(eval, out(), x(), z(),
                                                    fact());
  }
};

struct stage_advection_w3_rk {
  using out = inout_accessor<0>;
  using x = in_accessor<1>;
  using z = in_accessor<2>;
  using fact = in_accessor<3>;
  using in = in_accessor<4, extent<-3, 3, -3, 3>>;
  using in0 = in_accessor<5>;

  using u = in_accessor<6>;
  using v = in_accessor<7>;
  using dx = in_accessor<8>;
  using dy = in_accessor<9>;
  using dt = in_accessor<10>;

  using param_list =
      make_param_list<out, x, z, fact, in, in0, u, v, dx, dy, dt>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t) {
    auto vout =
        call<tridiagonal::periodic3, full_t>::with(eval, x(), z(), fact());
    auto flx = call<stage_u, full_t>::with(eval, u(), in(), dx());
    auto fly = call<stage_v, full_t>::with(eval, v(), in(), dy());
    eval(out()) = eval(in0() - dt() * (flx + fly) + (vout - in()));
  }
};

} // namespace

std::function<void(storage_t, storage_t, storage_t, storage_t, real_t)>
horizontal(vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta) {
  auto grid = computation_grid(resolution.x, resolution.y, resolution.z);
  return [grid = std::move(grid), delta](storage_t out, storage_t in,
                                         storage_t u, storage_t v, real_t dt) {
    gt::run_single_stage(stage_horizontal(), backend_t(), grid, out, in, u, v,
                         gt::make_global_parameter(delta.x),
                         gt::make_global_parameter(delta.y),
                         gt::make_global_parameter(dt));
  };
}

std::function<void(storage_t, storage_t, storage_t, real_t)>
vertical(vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta) {
  auto grid = computation_grid(resolution.x, resolution.y, resolution.z);
  auto const spec = [](auto out, auto in, auto w, auto alpha, auto beta,
                       auto gamma, auto fact, auto in_top, auto x_top,
                       auto z_top, auto dz, auto dt) {
    GT_DECLARE_TMP(real_t, a, b, c, d, x, z);
    return gt::multi_pass(
        gt::execute_forward().stage(stage_advection_w0(), in, in_top),
        gt::execute_forward()
            .k_cached(gt::cache_io_policy::flush(), a, b, c, d)
            .k_cached(gt::cache_io_policy::fill(), w)
            .stage(stage_advection_w_forward1(), alpha, beta, gamma, a, b, c, d,
                   in, in_top, dz, dt, w),
        gt::execute_backward()
            .k_cached(gt::cache_io_policy::flush(), x)
            .stage(stage_advection_w_backward1(), x, c, d),
        gt::execute_forward()
            .k_cached(gt::cache_io_policy::flush(), c, d)
            .stage(stage_advection_w_forward2(), a, b, c, d, alpha, gamma),
        gt::execute_backward().stage(stage_advection_w_backward2(), z, c, d, x,
                                     beta, gamma, fact, z_top, x_top),
        gt::execute_parallel().stage(stage_advection_w3(), out, x, z, fact, in,
                                     dt));
  };

  auto ij_slice = gt::storage::builder<storage_tr>
    .type<real_t>()
    .id<1>()
    .halos(halo, halo)
    .dimensions(resolution.x + 2 * halo, resolution.y + 2 * halo);

  auto alpha = ij_slice();
  auto beta = ij_slice();
  auto gamma = ij_slice();
  auto fact = ij_slice();
  auto in_top = ij_slice();
  auto x_top = ij_slice();
  auto z_top = ij_slice();

  return
      [grid = std::move(grid), spec = std::move(spec), alpha = std::move(alpha),
       beta = std::move(beta), gamma = std::move(gamma), fact = std::move(fact),
       in_top = std::move(in_top), x_top = std::move(x_top),
       z_top = std::move(z_top),
       delta](storage_t out, storage_t in, storage_t w, real_t dt) {
        gt::run(spec, backend_t(), grid, out, in, w, alpha, beta, gamma, fact,
                in_top, x_top, z_top, gt::make_global_parameter(delta.z),
                gt::make_global_parameter(dt));
      };
}

std::function<void(storage_t, storage_t, storage_t, storage_t, storage_t,
                   storage_t, real_t)>
runge_kutta_step(vec<std::size_t, 3> const &resolution,
                 vec<real_t, 3> const &delta) {
  auto grid = computation_grid(resolution.x, resolution.y, resolution.z);
  auto const spec = [](auto out, auto in, auto in0, auto u, auto v, auto w,
                       auto alpha, auto beta, auto gamma, auto fact,
                       auto in_top, auto x_top, auto z_top, auto dx, auto dy,
                       auto dz, auto dt) {
    GT_DECLARE_TMP(real_t, a, b, c, d, x, z);
    return gt::multi_pass(
        gt::execute_forward().stage(stage_advection_w0(), in, in_top),
        gt::execute_forward()
            .k_cached(gt::cache_io_policy::flush(), a, b, c, d)
            .k_cached(gt::cache_io_policy::fill(), w)
            .stage(stage_advection_w_forward1(), alpha, beta, gamma, a, b, c, d,
                   in, in_top, dz, dt, w),
        gt::execute_backward()
            .k_cached(gt::cache_io_policy::flush(), x)
            .stage(stage_advection_w_backward1(), x, c, d),
        gt::execute_forward()
            .k_cached(gt::cache_io_policy::flush(), c, d)
            .stage(stage_advection_w_forward2(), a, b, c, d, alpha, gamma),
        gt::execute_backward().stage(stage_advection_w_backward2(), z, c, d, x,
                                     beta, gamma, fact, z_top, x_top),
        gt::execute_parallel().stage(stage_advection_w3_rk(), out, x, z, fact,
                                     in, in0, u, v, dx, dy, dt));
  };

  auto ij_slice = gt::storage::builder<storage_tr>
    .type<real_t>()
    .id<1>()
    .halos(halo, halo)
    .dimensions(resolution.x + 2 * halo, resolution.y + 2 * halo);

  auto alpha = ij_slice();
  auto beta = ij_slice();
  auto gamma = ij_slice();
  auto fact = ij_slice();
  auto in_top = ij_slice();
  auto x_top = ij_slice();
  auto z_top = ij_slice();

  return [grid = std::move(grid), spec = std::move(spec),
          alpha = std::move(alpha), beta = std::move(beta),
          gamma = std::move(gamma), fact = std::move(fact),
          in_top = std::move(in_top), x_top = std::move(x_top),
          z_top = std::move(z_top),
          delta](storage_t out, storage_t in, storage_t in0, storage_t u,
                 storage_t v, storage_t w, real_t dt) {
    gt::run(spec, backend_t(), grid, out, in, in0, u, v, w, alpha, beta, gamma,
            fact, in_top, x_top, z_top, gt::make_global_parameter(delta.x),
            gt::make_global_parameter(delta.y),
            gt::make_global_parameter(delta.z), gt::make_global_parameter(dt));
  };
}

} // namespace advection
} // namespace numerics
