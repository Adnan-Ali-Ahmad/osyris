# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Osyris contributors (https://github.com/nvaytet/osyris)

"""
This file aims to re-introduce the ism_physics routines of osiris into Osyris.

To do:
-Opacities reader DONE
-Resistivities reader
-EOS reader DONE
"""

import struct
import os
import numpy as np
from ..core import Array
from .. import config
from .. import units
from ..io import utils
from scipy.interpolate import RegularGridInterpolator

# flake8: noqa

def ism_interpolate(table_container=None, values=[0], points=[0], in_log=False):

	func = RegularGridInterpolator(table_container["grid"], values)

	if in_log:
		return func(points)
	else:
		return np.power(10.0, func(points))


def read_opacity_table(fname, ngrp):
	"""
	Read binary opacity table in fname.
	"""

	print("Loading opacity table: "+fname)

	with open(fname, "rb") as f:
		data = f.read()

	# Create table container
	theTable = dict()

	# Initialise offset counters and start reading data
	offsets = {"i":0, "n":0, "d":0}

	# Get table dimensions
	nx,ny,nz = np.array(utils.read_binary_data(fmt="3i",content=data,increment=False))
	theTable["nx"] = np.array((ngrp, nx, ny, nz))
	# Read table coordinates:

	# x: density
	offsets["i"] += 3
	offsets["n"] += 9
	offsets["d"] += 1
	theTable["dens"] = utils.read_binary_data(fmt="%id"%theTable["nx"][1],content=data,offsets=offsets,increment=False)
	offsets["n"] -= 1

	# y: gas temperature
	offsets["n"] += theTable["nx"][1]
	offsets["d"] += 1
	theTable["tgas"] = utils.read_binary_data(fmt="%id"%theTable["nx"][2],content=data,offsets=offsets,increment=False)
	offsets["n"] -= 1

	# z: radiation temperature
	offsets["n"] += theTable["nx"][2]
	offsets["d"] += 1
	theTable["trad"] = utils.read_binary_data(fmt="%id"%theTable["nx"][3],content=data,offsets=offsets,increment=False)
	offsets["n"] -= 1

	# Now read opacities
	array_size = np.prod(theTable["nx"])
	array_fmt  = "%id" % array_size

	offsets["n"] += theTable["nx"][3]
	offsets["d"] += 1
	theTable["all_kappa_p"] = np.reshape(utils.read_binary_data(fmt=array_fmt,content=data, \
						offsets=offsets,increment=False),theTable["nx"],order="F")
	offsets["n"] -= 1

	offsets["n"] += array_size
	offsets["d"] += 1
	theTable["all_kappa_r"] = np.reshape(utils.read_binary_data(fmt=array_fmt,content=data, \
						offsets=offsets,increment=False),theTable["nx"],order="F")

	del data

	theTable["grid"] = (theTable["dens"],theTable["tgas"],theTable["trad"])


	print("Opacity table read successfully")

	return theTable


def get_opacities(dataset, fname, variables=["kappa_p", "kappa_r"]):
	"""
	Create opacity variables from interpolation of opacity table values in fname.
	"""
	default_units = {"kappa_p":"cm^2/g","kappa_r":"cm^2/g"}

	ngrp = dataset.meta["ngrp"]
	#if "opacity_table" not in dataset.meta:
	dataset.meta["opacity_table"] = read_opacity_table(fname, ngrp)

	for i in range(1, ngrp+1):
		if "radiative_temperature_{}".format(i) not in dataset["hydro"]:
			print("radiative_temperature_{} is not defined. Computing it now...".format(i), end="")
			rc = 1*units("radiation_constant")
			dataset["hydro"]["radiative_temperature_{}".format(i)] = ((dataset["hydro"]["radiative_energy_{}".format(i)]/rc)**.25).to("K")
			print(" done!")
		# check if radiative temperature out of grid bounds:
		rad_temp = np.copy(dataset["hydro"]["radiative_temperature_{}".format(i)].values)
		is_out_of_bounds = np.log10(rad_temp) < np.min(dataset.meta["opacity_table"]["grid"][2])
		if np.any(is_out_of_bounds):
			print("WARNING: Radiative temperature for group {} is out of bounds!".format(i))
			# proceed to fill those values with min grid trad
			rad_temp[is_out_of_bounds] = 10**np.min(dataset.meta["opacity_table"]["grid"][2])
		pts = np.array([np.log10(dataset["hydro"]["density"].values),np.log10(dataset["hydro"]["temperature"].values),np.log10(rad_temp)]).T
		for j,var in enumerate(variables):
			new_var = var + "_{}".format(i)
			print("Interpolating "+var+"_{}...".format(i), end="")
			vals = ism_interpolate(dataset.meta["opacity_table"], dataset.meta["opacity_table"]["all_kappa_r"][i-1], pts)
			print(" done!")
			dataset["hydro"][new_var] = Array(values = vals, unit = default_units[var])

	return


def read_eos_table(fname):
	"""
	Read binary EOS table in fname
	"""

	print("Loading EOS table: "+'"{}"'.format(fname)+"...", end="")

	# Read binary EOS file
	with open(fname, mode='rb') as f:
	    data = f.read()

	# Define data fields. Note that the order is important!
	data_fields = ["rho_eos","ener_eos","temp_eos","pres_eos","s_eos","cs_eos","xH_eos","xH2_eos","xHe_eos","xHep_eos"]

	# Create table container
	theTable = dict()

	# Initialise offset counters and start reading data
	offsets = {"i":0, "n":0, "d":0}

	# Get table dimensions
	theTable["nx"] = np.array(utils.read_binary_data(fmt="2i",content=data, increment=False))

	# Get table limits
	offsets["i"] += 2
	offsets["d"] += 1
	[theTable["rhomin"],theTable["rhomax"],theTable["emin"],theTable["emax"],theTable["yHe"]] = \
		utils.read_binary_data(fmt="5d",content=data,offsets=offsets, increment=False)
	offsets["n"] -= 1

	array_size = np.prod(theTable["nx"])
	array_fmt  = "%id" % array_size
	offsets["n"] += 5
	offsets["d"] += 1

	# Now loop through all the data fields
	for i in range(len(data_fields)):
		theTable[data_fields[i]] = np.reshape(utils.read_binary_data(fmt=array_fmt,content=data, \
			offsets=offsets, increment=False),theTable["nx"],order="F")
		offsets["n"] += array_size
		offsets["n"] -= 1
		offsets["d"] += 1

	del data

	Eint = theTable["ener_eos"]/theTable["rho_eos"]
	theTable["grid"] = (np.log10(theTable["rho_eos"][:,0]), np.log10(Eint[0,:]))

	print(" done!")

	return theTable


def get_eos(dataset, fname, variables=["rho_eos", "ener_eos", "temp_eos", "pres_eos", "s_eos", "cs_eos", "xH_eos", "xH2_eos", "xHe_eos", "xHep_eos"]):
	"""
	Create EOS variables from interpolation of eos table values in fname.
	"""

	default_units = {"rho_eos":"g/cm^3", "ener_eos":"erg","temp_eos":"K","pres_eos":"dyn/cm^2","s_eos":"erg/K/g","cs_eos":"cm/s","xH_eos":None,"xH2_eos":None,"xHe_eos":None,"xHep_eos":None}

	if dataset.meta["eos"] == 0:
		print("Simulation data did not use a tabulated EOS. Exiting.")
		return
	if "eos_table" not in dataset.meta:
		dataset.meta["eos_table"] = read_eos_table(fname=fname)

	pts = np.array([np.log10(dataset["hydro"]["density"].values), np.log10(dataset["hydro"]["internal_energy"].values/dataset["hydro"]["density"].values)]).T
	for var in variables:
		print("Interpolating "+var+"...", end="")
		vals = ism_interpolate(dataset.meta["eos_table"],np.log10(dataset.meta["eos_table"][var]),pts)
		dataset["hydro"][var] = Array(values = vals, unit = default_units[var])
		print(" done!")


def read_resistivity_table(fname="resistivities_masson2016.bin"):

	print("Loading resistivity table: "+fname)

	# Read binary resistivity file
	with open(fname, mode='rb') as res_file:
		data = res_file.read()
	res_file.close()

	# Create table container
	theTable = dict()

	# Initialise offset counters and start reading data
	offsets = {"i":0, "n":0, "d":0}

	# Get length of record on first line to determine number of dimensions in table
	rec_size = utils.read_binary_data(fmt="i",content=data,correction=-4)
	ndims = int(rec_size[0]/4)
	theTable["ndims"] = ndims

	# Get table dimensions
	nx,ny,nz = np.array(utils.read_binary_data(fmt="%ii"%ndims,content=data,increment=False))
	theTable["nx"] = np.array((nx, ny, nz))

	# Read table coordinates:

	# 1: density
	offsets["i"] += ndims
	offsets["d"] += 1
	theTable["dens"] = utils.read_binary_data(fmt="%id"%theTable["nx"][0],content=data,offsets=offsets, increment=False)

	# 2: gas temperature
	offsets["i"] += theTable["nx"][0]
	offsets["d"] += 1
	theTable["tgas"] = utils.read_binary_data(fmt="%id"%theTable["nx"][1],content=data,offsets=offsets, increment=False)

	if ndims == 4:
		# 3: ionisation rate
		offsets["i"] += theTable["nx"][1]
		offsets["d"] += 1
		theTable["ionx"] = utils.read_binary_data(fmt="%id"%theTable["nx"][2],content=data,offsets=offsets, increment=False)

	# 4: magnetic field
	offsets["i"] += theTable["nx"][-2]
	offsets["d"] += 1
	theTable["bmag"] = utils.read_binary_data(fmt="%id"%theTable["nx"][-1],content=data,offsets=offsets, increment=False)

	# Now read resistivities
	array_size = np.prod(theTable["nx"])
	array_fmt  = "%id" % array_size

	# Ohmic resistivity
	offsets["i"] += theTable["nx"][-1]
	offsets["d"] += 1
	theTable["eta_ohm"] = np.reshape(utils.read_binary_data(fmt=array_fmt,content=data, \
	            offsets=offsets, increment=False),theTable["nx"],order="F")

	# Ambipolar resistivity
	offsets["i"] += array_size
	offsets["d"] += 1
	theTable["eta_ad"] = np.reshape(utils.read_binary_data(fmt=array_fmt,content=data, \
	            offsets=offsets, increment=False),theTable["nx"],order="F")

	# Hall resistivity
	offsets["i"] += array_size
	offsets["d"] += 1
	theTable["eta_hall"] = np.reshape(utils.read_binary_data(fmt=array_fmt,content=data, \
	            offsets=offsets, increment=False),theTable["nx"],order="F")

	# Hall sign
	offsets["i"] += array_size
	offsets["d"] += 1
	theTable["eta_hsig"] = np.reshape(utils.read_binary_data(fmt=array_fmt,content=data, \
	            offsets=offsets, increment=False),theTable["nx"],order="F")

	del data

	if ndims == 4:
		theTable["grid"] = (theTable["dens"],theTable["tgas"],theTable["ionx"],theTable["bmag"])
	elif ndims == 3:
		theTable["grid"] = (theTable["dens"],theTable["tgas"],theTable["bmag"])

	# Additional parameters
	theTable["scale_dens"] = 0.844*2.0/1.66e-24 # 2.0*H2_fraction/mH
	theTable["ionis_rate"] = 1.0e-17

	print("Resistivity table read successfully")

	return theTable


def get_resistivity_table(dataset, fname, variables=["rho_eos", "ener_eos", "temp_eos", "pres_eos", "s_eos", "cs_eos", "xH_eos", "xH2_eos", "xHe_eos", "xHep_eos"]):
	"""
	Create EOS variables from interpolation of eos table values in fname.
	"""

	default_units = {"rho_eos":"g/cm^3", "ener_eos":"erg","temp_eos":"K","pres_eos":"dyn/cm^2","s_eos":"erg/K/g","cs_eos":"cm/s","xH_eos":None,"xH2_eos":None,"xHe_eos":None,"xHep_eos":None}

	if "res_table" not in dataset.meta:
		dataset.meta["res_table"] = read_resistivity_table(fname=fname)
