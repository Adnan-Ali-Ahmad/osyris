# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Osyris contributors (https://github.com/osyris-project/osyris)

import numpy as np
from . import utils


def translate(dataset, new_origin):
    """
    Translate all positionnal coordinates to new origin
    """
    for g in dataset.groups.keys():
        for element in dataset[g]:
            unit = dataset[g][element].unit
            if unit.is_compatible_with("meter") and hasattr(dataset[g][element],
                                                            "nvec"):
                dataset[g][element] -= new_origin
    dataset.origin = new_origin


# def rotate(dataset, new_basis, dr_L=None):
#     """
#     Rotates all vectors in dataset to align with vector in 'new_basis'
#     """
#     new_basis = utils._parse_basis(dataset, new_basis, dr_L)

#     try:
#         old_basis = dataset.basis
#     except AttributeError:
#         old_basis = [0, 0, 1]  # assume it's the ramses grid
#     if hasattr(new_basis, 'nvec'):  # if it's a vector
#         new_basis = [new_basis.x.values, new_basis.y.values, new_basis.z.values]
#     rot_angle = np.arccos(
#         np.dot(old_basis, new_basis) /
#         (np.linalg.norm(old_basis) * np.linalg.norm(new_basis)))
#     rot_vector = np.cross(new_basis, old_basis)
#     r_m = utils._rotation_matrix(rot_vector, rot_angle)
#     for g in dataset.groups.keys():
#         for element in dataset[g]:
#             if hasattr(dataset[g][element], 'nvec'):
#                 # all of this will be simplified once matmul is integraded into Vector
#                 vector = np.array([
#                     dataset[g][element].x.values, dataset[g][element].y.values,
#                     dataset[g][element].z.values
#                 ])
#                 vector = r_m @ vector
#                 dataset[g][element] = dataset[g][element].__class__(
#                     x=vector[0],
#                     y=vector[1],
#                     z=vector[2],
#                     unit=dataset[g][element].unit)
#     dataset.basis = new_basis

def rotate(dataset, new_basis, dr_L=None):
    """
    Rotates all vectors in dataset to align with vector in 'new_basis'
    """
    new_basis = utils._parse_basis(dataset, new_basis, dr_L)
    
    new_z = np.array([new_basis.x.values, new_basis.y.values, new_basis.z.values])
    new_z = new_z / np.linalg.norm(new_z)

    # if new_basis is co-linear with old_basis
    if abs(new_z[2]) < 0.99:
        new_x = np.cross([0, 0, 1], new_z)
    else:
        new_x = np.cross([0, 1, 0], new_z)
    new_x = new_x / np.linalg.norm(new_x)

    new_y = np.cross(new_z, new_x)

    for g in dataset.groups.keys():
        for element in dataset[g]:
            vec_obj = dataset[g][element]
            if hasattr(vec_obj, 'nvec') and vec_obj.nvec == 3:
                x = vec_obj.x.values
                y = vec_obj.y.values
                z = vec_obj.z.values

                xx = x * new_x[0] + y * new_x[1] + z * new_x[2]
                yy = x * new_y[0] + y * new_y[1] + z * new_y[2]
                zz = x * new_z[0] + y * new_z[1] + z * new_z[2]

                dataset[g][element] = vec_obj.__class__(
                    x=xx,
                    y=yy,
                    z=zz,
                    unit=vec_obj.unit
                )

    dataset.basis = new_z
