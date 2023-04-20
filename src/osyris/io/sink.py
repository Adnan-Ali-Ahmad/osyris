# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Osyris contributors (https://github.com/osyris-project/osyris)

import numpy as np
import os
from ..core import Array, Datagroup
from .reader import ReaderKind
from .. import units as ureg
from . import utils


class SinkReader:

    def __init__(self):
        self.kind = ReaderKind.SINK
        self.initialized = False

    def initialize(self, meta, units, select, ramses_ism):
        if select is False:
            return
        sink = Datagroup()
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
            sink_data = np.atleast_2d(
                np.loadtxt(sink_file, delimiter=',', skiprows=0 if ramses_ism else 2))

        if ramses_ism:
            variables = utils.read_sink_info(sink_file.replace(".csv", ".info"))
            key_list = list(variables.keys())
            unit_list = list(variables.values())
        else:
            with open(sink_file, 'r') as f:
                key_list = f.readline()
                unit_combinations = f.readline()
            key_list = key_list.lstrip(' #').rstrip('\n').split(',')
            unit_combinations = unit_combinations.lstrip(' #').rstrip('\n').split(',')

            # Parse units
            unit_list = []
            m = units['mass']  # noqa: F841
            l = units['length']  # noqa: F841, E741
            t = units['time']  # noqa: F841
            for u in unit_combinations:
                if u.strip().replace("[", "").replace("]", "") == '1':
                    unit_list.append(1.0 * ureg('dimensionless'))
                else:
                    if all(x in u for x in ["[", "]"]):
                        # Legacy sink format quantities are not in code units
                        unit_list.append(1.0 *
                                         ureg(u.replace("[", "").replace("]", "")))
                    else:
                        unit_list.append(eval(u.replace(' ', '*')))

        sink = Datagroup()
        for i, (key, unit) in enumerate(zip(key_list, unit_list)):
            if len(sink_data.shape) == 1:  # if only 1 sink is present
                sink[key] = Array(values=sink_data[i] * unit.magnitude, unit=unit.units)
            else:
                sink[key] = Array(values=sink_data[:, i] * unit.magnitude,
                                  unit=unit.units)
            if ramses_ism and key in ["x", "y", "z"]:
                sink[key] = (sink[key] * meta["unit_l"])
            elif ramses_ism and key in ["vx", "vy", "vz"]:
                sink[key] = (sink[key] * meta["unit_l"] / meta['unit_t'])
            elif not ramses_ism and unit_combinations[i] == 'l':
                sink[key] = sink[key]
        utils.make_vector_arrays(sink, ndim=meta["ndim"])
        return sink
