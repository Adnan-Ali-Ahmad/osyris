# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Osyris contributors (https://github.com/nvaytet/osyris)

import numpy as np
import os
from ..core import Array, Datagroup
from .reader import ReaderKind
from .. import units
from . import utils


class SinkReader:
    def __init__(self):
        self.kind = ReaderKind.SINK
        self.initialized = False

    def initialize(self, meta, select, ramses_ism):
        sink = Datagroup()
        if select is False:
            return sink
        sink_file = utils.generate_fname(meta["nout"],
                                         meta["path"],
                                         ftype="sink",
                                         cpuid=0,
                                         ext=".csv")
        if not os.path.exists(sink_file):
            return

        if os.path.getsize(sink_file) == 0:
            # This is an empty sink file
            return sink
        else:
            if ramses_ism:
                sink_data = np.atleast_2d(np.loadtxt(raw_data, delimiter=',', skiprows=0))  # do not skip rows
            else:
                sink_data = np.atleast_2d(np.loadtxt(raw_data, delimiter=',', skiprows=2))

        if ramses_ism:
            variables = utils.read_sink_info(sink_file.replace(".csv",".info"))
            key_list = list(variables.keys())
            unit_list = list(variables.values())
            #key_list = ["id", "M", "x", "y", "z", "vx", "vy", "vz"]
            #unit_list = [1.*units.dimensionless, 1.*units.msun, 1.*units.cm,1.*units.cm,1.*units.cm,1.*units.cmps,1.*units.cmps,1.*units.cmps]
        else:
            with open(sink_file, 'r') as f:
                key_list = f.readline()
                unit_combinations = f.readline()

            key_list = key_list.lstrip(' #').rstrip('\n').split(',')
            unit_combinations = unit_combinations.lstrip(' #').rstrip('\n').split(',')

            # Parse units
            unit_list = []
            for u in unit_combinations:
                m = meta['unit_d'] * meta['unit_l']**3 * units.g  # noqa: F841
                l = meta['unit_l'] * units.cm  # noqa: F841, E741
                t = meta['unit_t'] * units.s  # noqa: F841
                if u.strip() == '1':
                    unit_list.append(1.0 * units.dimensionless)
                else:
                    unit_list.append(eval(u.replace(' ', '*')))

        sink = Datagroup()
        for i, (key, unit) in enumerate(zip(key_list, unit_list)):
            sink[key] = Array(values=sink_data[:, i] * unit.magnitude, unit=unit.units)
            if ramses_ism and key in ["x","y","z"]:
                sink[key] = (sink[key]*meta["unit_l"]).to(meta["scale"])
            elif ramses_ism and key in ["vx","vy","vz"]:
                sink[key] = (sink[key]*meta["unit_l"]/meta['unit_t'])
            elif not ramses_ism and unit_combinations[i] == 'l':
                sink[key] = sink[key].to(meta["scale"])
        utils.make_vector_arrays(sink, ndim=meta["ndim"])
        return sink
