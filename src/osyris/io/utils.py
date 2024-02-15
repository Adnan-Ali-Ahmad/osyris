# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Osyris contributors (https://github.com/osyris-project/osyris)

import glob
import os
import struct
import cupy as np
from ..core import Vector
from .. import units as ureg


def generate_fname(nout, path="", ftype="", cpuid=1, ext=""):

    if nout == -1:
        filelist = sorted(glob.glob(os.path.join(path, "output*")))
        number = filelist[-1].split("_")[-1]
    else:
        number = str(nout).zfill(5)

    infile = os.path.join(path, "output_" + number)
    if len(ftype) > 0:
        infile = os.path.join(infile, ftype + "_" + number)
        if cpuid > 0:
            infile += ".out" + str(cpuid).zfill(5)

    if len(ext) > 0:
        infile += ext

    return infile


def read_parameter_file(fname=None, delimiter="="):
    """
    Read info file and create dictionary
    """
    out = {}
    with open(fname, 'r') as f:
        content = f.readlines()
    for line in content:
        sp = line.split(delimiter)
        if len(sp) > 1:
            value = sp[1].strip()
            try:
                value = eval(value)
            except NameError:
                pass
            out[sp[0].strip()] = value
    return out


def parse_units(s):
    """
    Parse the units of string s into pint units
    """
    if s == "Msol/y":
        return 1. * ureg("M_sol") / ureg("year")
    elif s == "Msol":
        return 1. * ureg("M_sol")
    elif s == "Lsol":
        return 1. * ureg("L_sol")
    elif s == "y":
        return 1. * ureg("year")
    elif s == "K":
        return 1. * ureg("K")


def read_sink_info(fname=None):
    """
    Read info file and return variable names dictionary.
    """
    variables = {}  # var name and units in this dictionary
    with open(fname, 'r') as f:
        data = f.readlines()
    for var in data[2].split():
        if "[" not in var:
            if var in ["x", "y", "z"]:
                variables[var] = 1. * ureg("cm")
            elif var in ["vx", "vy", "vz"]:
                variables[var] = 1. * ureg("cm") / ureg("s")
            else:
                # dimensionless unit
                if "/" in var:
                    var_name = var[:var.find(
                        "/")]  # case where units are scaled (eg. lx/|l|)
                else:
                    var_name = var
                variables[var_name] = 1. * ureg("dimensionless")
        else:
            var_name = var[:var.find("[")]  # remove units from string
            unit = parse_units(var[var.find("[") + 1:var.find("]")])  # unit string
            variables[var_name] = unit
    return variables


def read_binary_data(content=None,
                     fmt=None,
                     offsets=None,
                     skip_head=True,
                     increment=True):
    """
    Unpack binary data from a content buffer using a dict of offsets.
    Also increment the offsets of the corresponding data read, as well as
    increase the line count by 1.
    """

    byte_size = {
        "b": 1,
        "h": 2,
        "i": 4,
        "q": 8,
        "f": 4,
        "d": 8,
        "e": 8,
        "n": 8,
        "l": 8,
        "s": 1
    }

    if len(fmt) == 1:
        mult = 1
    else:
        mult = int(fmt[:-1])
    pack_size = mult * byte_size[fmt[-1]]

    offset = 0
    if offsets is not None:
        for key in offsets:
            offset += offsets[key] * byte_size[key]
        if increment:
            offsets[fmt[-1]] += mult
        offsets["n"] += 1
    if skip_head:
        offset += 4

    return struct.unpack(fmt, content[int(offset):int(offset) + pack_size])


def skip_binary_line(content, offsets):
    """
    Return the number of bytes necessary to skip the current line.
    """
    [nbytes] = read_binary_data(fmt="i",
                                content=content,
                                offsets=offsets,
                                skip_head=False,
                                increment=False)
    return nbytes


def make_vector_arrays(data, ndim):
    """
    Merge vector components in 2d arrays.
    """
    components = list("xyz"[:ndim])
    if len(components) > 1:
        delete = []
        for key in list(data.keys()):
            comp_list = None
            rawkey = None
            inds = [i for i, letter in enumerate(key) if letter == 'x']
            for ind in inds:
                comp_list = [key[:ind] + c + key[ind + 1:] for c in components]
                if all([item in data for item in comp_list]):
                    cut = ind - 1 if key[ind - 1] == "_" else ind
                    rawkey = key[:cut] + key[ind + 1:]
                    if len(rawkey) == 0:
                        rawkey = "position"
                    data[rawkey] = Vector(
                        **{components[c]: data[comp_list[c]]
                           for c in range(ndim)})
                    delete += comp_list
        for key in delete:
            del data[key]


def find_max_amr_level(levelmax, select):
    """
    Test the selection function in `select` on the range of possible AMR levels
    to determine the max level to read.
    """
    possible_levels = np.arange(1, levelmax + 1, dtype=int)
    func_test = select["level"](possible_levels)
    inds = np.argwhere(func_test).ravel()
    return possible_levels[inds.max()]
