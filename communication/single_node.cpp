/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "./single_node.hpp"

#include <gridtools/boundary_conditions/boundary.hpp>

namespace communication {

namespace single_node {

struct periodic_boundary {
  template <gt::sign I, gt::sign J, gt::sign K, typename DataField>
  GT_FUNCTION void operator()(gt::direction<I, J, K>, DataField &data,
                              gt::uint_t i, gt::uint_t j, gt::uint_t k) const {
    auto const &lengths = data.lengths();
    data(i, j, k) = data((i + lengths[0] - halo) % lengths[1] + halo,
                         (j + lengths[1] - halo) % lengths[1] + halo, k);
  }
};

std::function<void(storage_t &)>
comm_halo_exchanger(grid const &grid, gt::storage::info<3> const &sinfo) {
  gt::uint_t nx = grid.resolution.x;
  gt::uint_t ny = grid.resolution.y;
  gt::uint_t nz = grid.resolution.z;
  const gt::array<gt::halo_descriptor, 3> halos{
      {{halo, halo, halo, halo + nx - 1, halo + nx + halo},
       {halo, halo, halo, halo + ny - 1, halo + ny + halo},
       {0, 0, 0, nz - 1, nz}}};
  gt::boundary<periodic_boundary, backend_t> boundary(halos,
                                                      periodic_boundary());
  return [boundary = std::move(boundary)](storage_t &storage) {
    boundary.apply(storage);
  };
}

void comm_barrier(grid &) {}

} // namespace single_node

} // namespace communication
