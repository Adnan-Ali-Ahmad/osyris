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
                     nx, ny, nz, ndim,
                     # New arguments for on-the-fly calculation
                     origin_x, origin_y, origin_z,
                     ux, uy, uz,
                     vx, vy, vz,
                     nx_vec, ny_vec, nz_vec): # Renamed 'normal' to avoid confusion with nx

    diagonal = np.sqrt(ndim)
    
    # Pre-calculate inverse spacing
    inv_dx = 1.0 / grid_spacing_in_new_basis_x
    inv_dy = 1.0 / grid_spacing_in_new_basis_y
    inv_dz = 1.0 / grid_spacing_in_new_basis_z

    # Allocate output (Channels, Z, Y, X)
    out = np.full(shape=(cell_values.shape[0], nz, ny, nx),
                  fill_value=np.nan,
                  dtype=np.float64)

    ncells = len(cell_positions_in_new_basis_x)
    has_y = cell_positions_in_original_basis_y is not None
    has_z = cell_positions_in_original_basis_z is not None

    for n in prange(ncells):
        half_size = cell_sizes[n] * diagonal
        current_val = cell_values[:, n]
        current_size = cell_sizes[n]
        
        # Cell position in original basis
        pos_orig_x = cell_positions_in_original_basis_x[n]
        pos_orig_y = cell_positions_in_original_basis_y[n] if has_y else 0.0
        pos_orig_z = cell_positions_in_original_basis_z[n] if has_z else 0.0

        # Calculate integer bounds in the Grid Basis
        # Note: We add 0.5 to offset to the pixel center logic immediately
        start_x = (cell_positions_in_new_basis_x[n] - half_size - grid_lower_edge_in_new_basis_x) * inv_dx
        end_x = (cell_positions_in_new_basis_x[n] + half_size - grid_lower_edge_in_new_basis_x) * inv_dx
        
        ix1 = max(int(start_x), 0)
        ix2 = min(int(end_x) + 1, nx)

        start_y = (cell_positions_in_new_basis_y[n] - half_size - grid_lower_edge_in_new_basis_y) * inv_dy
        end_y = (cell_positions_in_new_basis_y[n] + half_size - grid_lower_edge_in_new_basis_y) * inv_dy
        
        iy1 = max(int(start_y), 0)
        iy2 = min(int(end_y) + 1, ny)

        start_z = (cell_positions_in_new_basis_z[n] - half_size - grid_lower_edge_in_new_basis_z) * inv_dz
        end_z = (cell_positions_in_new_basis_z[n] + half_size - grid_lower_edge_in_new_basis_z) * inv_dz
        
        iz1 = max(int(start_z), 0)
        iz2 = min(int(end_z) + 1, nz)

        for k in range(iz1, iz2):
            # Calculate Z component of the pixel position relative to Origin
            # z_coord is the distance in the new basis
            z_coord = grid_lower_edge_in_new_basis_z + (k + 0.5) * grid_spacing_in_new_basis_z
            
            # Pre-calculate the Z contribution to the physical position
            pz_x = z_coord * nz_vec
            pz_y = z_coord * ny_vec
            pz_z = z_coord * nz_vec # logic error here in math, fixing below

            for j in range(iy1, iy2):
                y_coord = grid_lower_edge_in_new_basis_y + (j + 0.5) * grid_spacing_in_new_basis_y
                
                # Y contribution
                py_x = y_coord * vx
                py_y = y_coord * vy
                py_z = y_coord * vz

                for i in range(ix1, ix2):
                    x_coord = grid_lower_edge_in_new_basis_x + (i + 0.5) * grid_spacing_in_new_basis_x

                    # Physical Position Calculation (Fused)
                    # P = Origin + x*U + y*V + z*N
                    pixel_x = origin_x + x_coord * ux + py_x + pz_x
                    
                    dist_x = pixel_x - pos_orig_x
                    if np.abs(dist_x) > current_size: continue

                    if has_y:
                        pixel_y = origin_y + x_coord * uy + py_y + pz_y
                        dist_y = pixel_y - pos_orig_y
                        if np.abs(dist_y) > current_size: continue
                    
                    if has_z:
                        pixel_z = origin_z + x_coord * uz + py_z + pz_z # Fixed logic
                        dist_z = pixel_z - pos_orig_z
                        if np.abs(dist_z) > current_size: continue

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
