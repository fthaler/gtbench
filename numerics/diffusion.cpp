/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "./diffusion.hpp"

#include <gridtools/stencil/cartesian.hpp>

#include "./computation.hpp"
#include "./tridiagonal.hpp"

namespace numerics {
namespace diffusion {

namespace {
using gt::stencil::extent;
using gt::stencil::make_param_list;
using gt::stencil::cartesian::call_proc;
using gt::stencil::cartesian::in_accessor;
using gt::stencil::cartesian::inout_accessor;
using namespace gt::stencil::cartesian::expressions;

struct stage_horizontal {
  using out = inout_accessor<0>;
  using in = in_accessor<1, extent<-3, 3, -3, 3>>;

  using dx = in_accessor<2>;
  using dy = in_accessor<3>;
  using dt = in_accessor<4>;
  using coeff = in_accessor<5>;

  using param_list = make_param_list<out, in, dx, dy, dt, coeff>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t) {
    constexpr static real_t weights[] = {-1_r / 90, 5_r / 36,  -49_r / 36,
                                         49_r / 36, -5_r / 36, 1_r / 90};

    auto flx_x0 = eval((weights[0] * in(-3, 0) + weights[1] * in(-2, 0) +
                        weights[2] * in(-1, 0) + weights[3] * in(0, 0) +
                        weights[4] * in(1, 0) + weights[5] * in(2, 0)) /
                       dx());
    auto flx_x1 = eval((weights[0] * in(-2, 0) + weights[1] * in(-1, 0) +
                        weights[2] * in(0, 0) + weights[3] * in(1, 0) +
                        weights[4] * in(2, 0) + weights[5] * in(3, 0)) /
                       dx());
    auto flx_y0 = eval((weights[0] * in(0, -3) + weights[1] * in(0, -2) +
                        weights[2] * in(0, -1) + weights[3] * in(0, 0) +
                        weights[4] * in(0, 1) + weights[5] * in(0, 2)) /
                       dy());
    auto flx_y1 = eval((weights[0] * in(0, -2) + weights[1] * in(0, -1) +
                        weights[2] * in(0, 0) + weights[3] * in(0, 1) +
                        weights[4] * in(0, 2) + weights[5] * in(0, 3)) /
                       dy());

    flx_x0 = flx_x0 * eval(in() - in(-1, 0)) < 0_r ? 0_r : flx_x0;
    flx_x1 = flx_x1 * eval(in(1, 0) - in()) < 0_r ? 0_r : flx_x1;
    flx_y0 = flx_y0 * eval(in() - in(0, -1)) < 0_r ? 0_r : flx_y0;
    flx_y1 = flx_y1 * eval(in(0, 1) - in()) < 0_r ? 0_r : flx_y1;

    eval(out()) =
        eval(in() + coeff() * dt() *
                        ((flx_x1 - flx_x0) / dx() + (flx_y1 - flx_y0) / dy()));
  }
};

struct stage_diffusion_w0 {
  using data = in_accessor<0>;
  using data_top = inout_accessor<1>;

  using param_list = make_param_list<data, data_top>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
    eval(data_top()) = eval(data());
  }
};

struct stage_diffusion_w_forward1 {
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
  using coeff = in_accessor<11>;

  using param_list = make_param_list<alpha, beta, gamma, a, b, c, d, data,
                                     data_tmp, dz, dt, coeff>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
    eval(a()) = eval(c()) = eval(-coeff() / (2_r * dz() * dz()));
    eval(b()) = eval(1_r / dt() - a() - c());
    eval(d()) =
        eval(1_r / dt() * data() +
             0.5_r * coeff() * (data_tmp() - 2_r * data() + data(0, 0, 1)) /
                 (dz() * dz()));

    eval(alpha()) = eval(beta()) = eval(-coeff() / (2_r * dz() * dz()));
    eval(gamma()) = eval(-b());

    call_proc<tridiagonal::periodic_forward1, full_t::first_level>::with(
        eval, a(), b(), c(), d(), alpha(), beta(), gamma());

    eval(data_tmp()) = eval(data());
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, -1>) {
    eval(a()) = eval(c()) = eval(-coeff() / (2_r * dz() * dz()));
    eval(b()) = eval(1_r / dt() - a() - c());
    eval(d()) =
        eval(1_r / dt() * data() +
             0.5_r * coeff() * (data(0, 0, -1) - 2_r * data() + data(0, 0, 1)) /
                 (dz() * dz()));

    call_proc<tridiagonal::periodic_forward1, full_t::modify<1, -1>>::with(
        eval, a(), b(), c(), d(), alpha(), beta(), gamma());
  }
  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
    eval(a()) = eval(c()) = eval(-coeff() / (2_r * dz() * dz()));
    eval(b()) = eval(1_r / dt() - a() - c());
    eval(d()) =
        eval(1_r / dt() * data() +
             0.5_r * coeff() * (data(0, 0, -1) - 2_r * data() + data_tmp()) /
                 (dz() * dz()));
    call_proc<tridiagonal::periodic_forward1, full_t::last_level>::with(
        eval, a(), b(), c(), d(), alpha(), beta(), gamma());
  }
};

using stage_diffusion_w_backward1 = tridiagonal::periodic_backward1;
using stage_diffusion_w_forward2 = tridiagonal::periodic_forward2;
using stage_diffusion_w_backward2 = tridiagonal::periodic_backward2;

struct stage_diffusion_w3 {
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

} // namespace

std::function<void(storage_t, storage_t, real_t dt)>
horizontal(vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta,
           real_t coeff) {
  auto grid = computation_grid(resolution.x, resolution.y, resolution.z);
  return [grid = std::move(grid), delta, coeff](storage_t out, storage_t in,
                                                real_t dt) {
    gt::stencil::run_single_stage(
        stage_horizontal(), backend_t(), grid, out, in,
        gt::stencil::make_global_parameter(delta.x), gt::stencil::make_global_parameter(delta.y),
        gt::stencil::make_global_parameter(dt), gt::stencil::make_global_parameter(coeff));
  };
}

std::function<void(storage_t, storage_t, real_t dt)>
vertical(vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta,
         real_t coeff) {
  auto grid = computation_grid(resolution.x, resolution.y, resolution.z);
  auto const spec = [](auto out, auto in, auto alpha, auto beta, auto gamma,
                       auto fact, auto in_top, auto x_top, auto z_top, auto dz,
                       auto dt, auto coeff) {
    GT_DECLARE_TMP(real_t, a, b, c, d, z, x);
    return gt::stencil::multi_pass(
        gt::stencil::execute_forward().stage(stage_diffusion_w0(), in, in_top),
        gt::stencil::execute_forward()
            .k_cached(gt::stencil::cache_io_policy::flush(), a, b, c, d)
            .stage(stage_diffusion_w_forward1(), alpha, beta, gamma, a, b, c, d,
                   in, in_top, dz, dt, coeff),
        gt::stencil::execute_backward()
            .k_cached(gt::stencil::cache_io_policy::flush(), x)
            .stage(stage_diffusion_w_backward1(), x, c, d),
        gt::stencil::execute_forward()
            .k_cached(gt::stencil::cache_io_policy::flush(), c, d)
            .stage(stage_diffusion_w_forward2(), a, b, c, d, alpha, gamma),
        gt::stencil::execute_backward().stage(stage_diffusion_w_backward2(), z, c, d, x,
                                     beta, gamma, fact, z_top, x_top),
        gt::stencil::execute_parallel().stage(stage_diffusion_w3(), out, x, z, fact, in,
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

  return [grid = std::move(grid), spec = std::move(spec),
          alpha = std::move(alpha), beta = std::move(beta),
          gamma = std::move(gamma), fact = std::move(fact),
          in_top = std::move(in_top), x_top = std::move(x_top),
          z_top = std::move(z_top), delta,
          coeff](storage_t out, storage_t in, real_t dt) {
    gt::stencil::run(spec, backend_t(), grid, out, in, alpha, beta, gamma, fact, in_top,
            x_top, z_top, gt::stencil::make_global_parameter(delta.z),
            gt::stencil::make_global_parameter(dt), gt::stencil::make_global_parameter(coeff));
  };
}

} // namespace diffusion
} // namespace numerics
