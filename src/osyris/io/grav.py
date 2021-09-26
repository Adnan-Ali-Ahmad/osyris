# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Osyris contributors (https://github.com/nvaytet/osyris)
import os
from .reader import Reader, ReaderKind
from .. import config
from . import utils


class GravReader(Reader):
    def __init__(self):
        super().__init__(kind=ReaderKind.AMR)
        # self.fname = utils.generate_fname(nout, path, ftype="grav", cpuid=1)
        # self.code_units = code_units
        # self.ndim = ndim

    def initialize(self, meta, select):

        # super().__init__()
        fname = utils.generate_fname(meta["nout"], meta["path"], ftype="grav", cpuid=1)

        # Check if self-gravity files exist
        # fname = utils.generate_fname(nout, path, ftype="grav", cpuid=1)
        if not os.path.exists(fname):
            return
        # Add gravity fields
        # if self.initialized:
        descriptor = {"potential": "d"}
        for n in range(meta["ndim"]):
            descriptor["acceleration_" + "xyz"[n]] = "d"
        # Now add to the list of variables to be read
        for key in descriptor:
            read = True
            if "gravity" in select:
                if select["gravity"] is False:
                    read = False
            if "grav" in select:
                if select["grav"] is False:
                    read = False
            if key in select:
                if isinstance(select[key], bool):
                    read = select[key]
            self.variables[key] = {
                "read": read,
                "type": descriptor[key],
                "buffer": None,
                "pieces": {},
                "unit": config.get_unit(key, meta["unit_d"], meta["unit_l"],
                                        meta["unit_t"])
            }
        self.initialized = True

    def read_header(self, info):
        self.offsets["i"] += 4
        self.offsets["n"] += 4

    def read_domain_header(self):
        self.offsets['n'] += 2
        self.offsets['i'] += 2
