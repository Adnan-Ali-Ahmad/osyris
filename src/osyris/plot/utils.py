# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Osyris contributors (https://github.com/osyris-project/osyris)

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def evaluate_on_grid(cell_positions_in_new_basis_x, cell_positions_in_new_basis_y,
                     cell_positions_in_new_basis_z, cell_positions_in_original_basis_x,
                     cell_positions_in_original_basis_y,
                     cell_positions_in_original_basis_z, cell_values, cell_sizes,
                     grid_lower_edge_in_new_basis_x, grid_lower_edge_in_new_basis_y,
                     grid_lower_edge_in_new_basis_z, grid_spacing_in_new_basis_x,
                     grid_spacing_in_new_basis_y, grid_spacing_in_new_basis_z,
                     grid_positions_in_original_basis, ndim):

    nz, ny, nx = grid_positions_in_original_basis.shape[:3]
    diagonal = np.sqrt(ndim)

    inv_dx = 1.0 / grid_spacing_in_new_basis_x
    inv_dy = 1.0 / grid_spacing_in_new_basis_y
    inv_dz = 1.0 / grid_spacing_in_new_basis_z

    out = np.full(shape=(cell_values.shape[0], nz, ny, nx),
                  fill_value=np.nan,
                  dtype=np.float64)

    ncells = len(cell_positions_in_new_basis_x)

    for n in prange(ncells):

        half_size = cell_sizes[n] * diagonal

        current_val = cell_values[:,
                                  n]  # cache cell value (avoids repeated memory lookups)
        current_size = cell_sizes[n]

        # cell position in original basis
        pos_orig_x = cell_positions_in_original_basis_x[n]
        pos_orig_y = cell_positions_in_original_basis_y[n] if has_y else 0.0
        pos_orig_z = cell_positions_in_original_basis_z[n] if has_z else 0.0

        ix1 = int((cell_positions_in_new_basis_x[n] - half_size -
                   grid_lower_edge_in_new_basis_x) * inv_dx)
        ix2 = int((cell_positions_in_new_basis_x[n] + half_size -
                   grid_lower_edge_in_new_basis_x) * inv_dx) + 1

        iy1 = int((cell_positions_in_new_basis_y[n] - half_size -
                   grid_lower_edge_in_new_basis_y) * inv_dy)
        iy2 = int((cell_positions_in_new_basis_y[n] + half_size -
                   grid_lower_edge_in_new_basis_y) * inv_dy) + 1

        iz1 = int((cell_positions_in_new_basis_z[n] - half_size -
                   grid_lower_edge_in_new_basis_z) * inv_dz)
        iz2 = int((cell_positions_in_new_basis_z[n] + half_size -
                   grid_lower_edge_in_new_basis_z) * inv_dz) + 1

        ix1 = max(ix1, 0)
        ix2 = min(ix2, nx)
        iy1 = max(iy1, 0)
        iy2 = min(iy2, ny)
        iz1 = max(iz1, 0)
        iz2 = min(iz2, nz)

        for k in range(iz1, iz2):
            for j in range(iy1, iy2):
                for i in range(ix1, ix2):
                    dist_x = grid_positions_in_original_basis[k, j, i, 0] - pos_orig_x
                    if np.abs(dist_x) > current_size:
                        continue
                    
                    dist_y = grid_positions_in_original_basis[k, j, i, 1] - pos_orig_y
                    if np.abs(dist_y) > current_size:
                        continue

                    dist_z = grid_positions_in_original_basis[k, j, i, 2] - pos_orig_z
                    if np.abs(dist_z) > current_size:
                        continue

                    out[:, k, j, i] = current_val

    return out


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
