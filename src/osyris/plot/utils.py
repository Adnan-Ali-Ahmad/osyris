# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Osyris contributors (https://github.com/osyris-project/osyris)

import numpy as np
from numba import njit, prange


def evaluate_on_grid(
    cell_positions_in_new_basis_x,
    cell_positions_in_new_basis_y,
    cell_positions_in_new_basis_z,
    cell_positions_in_original_basis_x,
    cell_positions_in_original_basis_y,
    cell_positions_in_original_basis_z,
    cell_values,
    cell_sizes,
    grid_lower_edge_in_new_basis_x,
    grid_lower_edge_in_new_basis_y,
    grid_lower_edge_in_new_basis_z,
    grid_spacing_in_new_basis_x,
    grid_spacing_in_new_basis_y,
    grid_spacing_in_new_basis_z,
    nx,
    ny,
    nz,
    ndim,
    ux,
    uy,
    uz,
    vx,
    vy,
    vz,
    nx_vec,
    ny_vec,
    nz_vec,
):
    inv_dx = 1.0 / grid_spacing_in_new_basis_x
    inv_dy = 1.0 / grid_spacing_in_new_basis_y
    inv_dz = 1.0 / grid_spacing_in_new_basis_z

    out = np.full(
        shape=(cell_values.shape[0], nz, ny, nx), fill_value=np.nan, dtype=np.float64
    )

    ncells = len(cell_positions_in_new_basis_x)
    diagonal = np.sqrt(ndim)

    has_y = cell_positions_in_original_basis_y is not None
    has_z = cell_positions_in_original_basis_z is not None

    for n in prange(ncells):
        half_size = cell_sizes[n] * diagonal
        current_val = cell_values[:, n]
        current_size = cell_sizes[n]

        pos_orig_x = cell_positions_in_original_basis_x[n]
        pos_orig_y = cell_positions_in_original_basis_y[n] if has_y else 0.0
        pos_orig_z = cell_positions_in_original_basis_z[n] if has_z else 0.0

        rel_x = cell_positions_in_new_basis_x[n] - grid_lower_edge_in_new_basis_x
        rel_y = cell_positions_in_new_basis_y[n] - grid_lower_edge_in_new_basis_y
        rel_z = cell_positions_in_new_basis_z[n] - grid_lower_edge_in_new_basis_z

        ix1 = int((rel_x - half_size) * inv_dx)
        ix2 = int((rel_x + half_size) * inv_dx) + 1
        iy1 = int((rel_y - half_size) * inv_dy)
        iy2 = int((rel_y + half_size) * inv_dy) + 1
        iz1 = int((rel_z - half_size) * inv_dz)
        iz2 = int((rel_z + half_size) * inv_dz) + 1

        ix1 = max(ix1, 0)
        ix2 = min(ix2, nx)
        iy1 = max(iy1, 0)
        iy2 = min(iy2, ny)
        iz1 = max(iz1, 0)
        iz2 = min(iz2, nz)

        for k in range(iz1, iz2):
            # get z coords
            z_map = (
                grid_lower_edge_in_new_basis_z + (k + 0.5) * grid_spacing_in_new_basis_z
            )

            pz_x = z_map * nx_vec
            pz_y = z_map * ny_vec
            pz_z = z_map * nz_vec

            for j in range(iy1, iy2):
                # get y coords
                y_map = (
                    grid_lower_edge_in_new_basis_y
                    + (j + 0.5) * grid_spacing_in_new_basis_y
                )

                py_x = y_map * vx
                py_y = y_map * vy
                py_z = y_map * vz

                pyz_x = py_x + pz_x
                pyz_y = py_y + pz_y
                pyz_z = py_z + pz_z

                for i in range(ix1, ix2):
                    # get x coords
                    x_map = (
                        grid_lower_edge_in_new_basis_x
                        + (i + 0.5) * grid_spacing_in_new_basis_x
                    )

                    # reconstruct pos
                    grid_x = x_map * ux + pyz_x

                    dist_x = grid_x - pos_orig_x
                    if np.abs(dist_x) > current_size:
                        continue

                    if has_y:
                        grid_y = x_map * uy + pyz_y
                        dist_y = grid_y - pos_orig_y
                        if np.abs(dist_y) > current_size:
                            continue

                    if has_z:
                        grid_z = x_map * uz + pyz_z
                        dist_z = grid_z - pos_orig_z
                        if np.abs(dist_z) > current_size:
                            continue

                    out[:, k, j, i] = current_val

    return out


# compile two versions of the above function
evaluate_on_grid_fast = njit(parallel=True, fastmath=True)(evaluate_on_grid)
evaluate_on_grid_precise = njit(parallel=True, fastmath=False)(evaluate_on_grid)


@njit(parallel=True)
def hist2d(x, y, values, xmin, xmax, nx, ymin, ymax, ny):
    out = np.zeros(shape=(values.shape[0], ny, nx), dtype=np.float64)
    counts = np.zeros(shape=(ny, nx), dtype=np.int64)
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    for i in prange(len(x)):
        indx = int((x[i] - xmin) / dx)
        indy = int((y[i] - ymin) / dy)
        if (indx >= 0) and (indx < nx) and (indy >= 0) and (indy < ny):
            out[:, indy, indx] += values[:, i]
            counts[indy, indx] += 1

    return out, counts
