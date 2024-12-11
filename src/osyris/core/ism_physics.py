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
from numba import jit

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


def get_masson_table(theTable, data, ndims):

	# Initialise offset counters and start reading data
	offsets = {"i":0, "n":0, "d":0}

	# Get table dimensions
	nx,ny,nz = np.array(utils.read_binary_data(fmt="%ii"%ndims,content=data,increment=False))
	theTable["nx"] = np.array((nx, ny, nz))

	# Read table coordinates:

	# 1: density
	offsets["i"] += ndims
	offsets["d"] += 1
	theTable["dens"] = utils.read_binary_data(fmt="%id"%theTable["nx"][0],content=data,offsets=offsets, increment=False)
	offsets["n"] -= 1

	# 2: gas temperature
	offsets["n"] += theTable["nx"][0]
	offsets["d"] += 1
	theTable["tgas"] = utils.read_binary_data(fmt="%id"%theTable["nx"][1],content=data,offsets=offsets, increment=False)
	offsets["n"] -= 1

	if ndims == 4:
		# 3: ionisation rate
		offsets["n"] += theTable["nx"][1]
		offsets["d"] += 1
		theTable["ionx"] = utils.read_binary_data(fmt="%id"%theTable["nx"][2],content=data,offsets=offsets, increment=False)
		offsets["n"] -= 1

	# 4: magnetic field
	offsets["n"] += theTable["nx"][-2]
	offsets["d"] += 1
	theTable["bmag"] = utils.read_binary_data(fmt="%id"%theTable["nx"][-1],content=data,offsets=offsets, increment=False)
	offsets["n"] -= 1

	# Now read resistivities
	array_size = np.prod(theTable["nx"])
	array_fmt  = "%id" % array_size

	# Ohmic resistivity
	offsets["n"] += theTable["nx"][-1]
	offsets["d"] += 1
	theTable["eta_ohm"] = np.reshape(utils.read_binary_data(fmt=array_fmt,content=data, \
	            offsets=offsets, increment=False),theTable["nx"],order="F")
	offsets["n"] -= 1

	# Ambipolar resistivity
	offsets["n"] += array_size
	offsets["d"] += 1
	theTable["eta_ad"] = np.reshape(utils.read_binary_data(fmt=array_fmt,content=data, \
	            offsets=offsets, increment=False),theTable["nx"],order="F")
	offsets["n"] -= 1

	# Hall resistivity
	offsets["n"] += array_size
	offsets["d"] += 1
	theTable["eta_hall"] = np.reshape(utils.read_binary_data(fmt=array_fmt,content=data, \
	            offsets=offsets, increment=False),theTable["nx"],order="F")
	offsets["n"] -= 1

	# Hall sign
	offsets["n"] += array_size
	offsets["d"] += 1
	theTable["eta_hsig"] = np.reshape(utils.read_binary_data(fmt=array_fmt,content=data, \
	            offsets=offsets, increment=False),theTable["nx"],order="F")
	offsets["n"] -= 1

	del data

	theTable["grid"] = (theTable["dens"],theTable["tgas"],theTable["bmag"])

	# Additional parameters
	theTable["scale_dens"] = 0.844*2.0/1.66e-24 # 2.0*H2_fraction/mH
	theTable["ionis_rate"] = 1.0e-17

	print("Resistivity table read successfully")

	return theTable


@jit(nopython=True)
def compute_resistivities(resistivite_chimie_x, nx, ndims):
	# Computes conductivities, and then infers resistivities

	# Define some constants ==================================
	rg      = 0.1e-4       # grain radius 
	mp      = 1.6726e-24   # proton mass
	me      = 9.1094e-28   # electron mass
	mg      = 1.2566e-14   # grain mass
	e       = 4.803204e-10 # electron charge
	mol_ion = 29.0*mp      # molecular ion mass
	Met_ion = 23.5*mp      # atomic ion mass
	kb      = 1.3807e-16   # Boltzmann
	clight  = 2.9979250e+10
	#real(kind=8), parameter ::mH      = 1.6600000d-24
	#real(kind=8), parameter ::mu_gas = 2.31d0
	#real(kind=8) :: scale_d = mu_gas*mH
	rho_s      = 2.3
	rho_n_tot  = 1.17e-21
	a_0        = 0.0375e-4
	a_min      = 0.0181e-4
	a_max      = 0.9049e-4
	zeta       = a_min/a_max
	lambda_pow = -3.5
	# Compute grain distribution =============================
	nbins_grains = int((nx[0]-9)/3)
	nion = 9

	r_g = np.zeros((nbins_grains))

	Lp1 = lambda_pow + 1.0
	Lp3 = lambda_pow + 3.0
	Lp4 = lambda_pow + 4.0
	fnb = float(nbins_grains)

	if nbins_grains == 1:
		 r_g[0] = a_0
	else:
		for i in range(nbins_grains):
			r_g[i] = a_min*zeta**(-float(i+1)/fnb) * np.sqrt( Lp1/Lp3* (1.0-zeta**(Lp3/fnb))/(1.0-zeta**(Lp1/fnb)))
	q   = np.zeros((nx[0]))
	m   = np.zeros((nx[0]))
	m_g = np.zeros((nx[0]))
	q[:] = e
	q[0] = -e
	qchrg = [0.0,e,-e]
	for i in range(nion,nx[0]):
		q[i] = qchrg[(i-8) % 3]               
	m[0] = me        # e-
	m[1] = 23.5*mp   # metallic ions
	m[2] = 29.0*mp   # molecular ions
	m[3] = 3.0*mp    # H3+
	m[4] = mp        # H+
	m[5] = 12.0*mp   # C+
	m[6] = 4.0*mp    # He+
	if ndims == 4:
		m[7] = 39.098*mp # K+
		m[8] = 22.990*mp # Na+
	for i in range(nbins_grains):
		m_g[i] = 4.0/3.0*np.pi*r_g[i]**3*rho_s
		m[nion+1+3*i:nion+1+3*(i+1)] = m_g[i]

	# Compute conductivities =============================
	# Define magnetic field range and resolution
	bchimie = 200
	Barray = np.logspace(-10, 10, bchimie)
	
	nchimie = nx[1]
	tchimie = nx[2]

	xichimie = nx[3]
	eta_ohm  = np.zeros((nchimie,tchimie,xichimie,bchimie), dtype=np.float64)
	eta_ad   = np.zeros((nchimie,tchimie,xichimie,bchimie), dtype=np.float64)
	eta_hall = np.zeros((nchimie,tchimie,xichimie,bchimie), dtype=np.float64)
	eta_hsig = np.zeros((nchimie,tchimie,xichimie,bchimie), dtype=np.float64)

	tau_sn      = np.zeros((nx[0]))
	omega       = np.zeros((nx[0]))
	sigma       = np.zeros((nx[0]))
	phi         = np.zeros((nx[0]))
	zetas       = np.zeros((nx[0]))
	gamma_zeta  = np.zeros((nx[0]))
	gamma_omega = np.zeros((nx[0]))
	omega_bar   = np.zeros((nx[0]))

	for iX in range(xichimie):
		for iB in range(bchimie):
			for iT in range(tchimie):
				for iH in range(nchimie):
					B = Barray[iB]
					nh = resistivite_chimie_x[0,iH,iT,iX]  # density (.cc) of current point
					T  = resistivite_chimie_x[1,iH,iT,iX]
					xi = resistivite_chimie_x[2,iH,iT,iX]
					for i in range(nion):
						if  i==0 : # electron
							sigv = 3.16e-11 * (np.sqrt(8.0*kb*1.0e-7*T/(np.pi*me*1.0e-3))*1.0e-3)**1.3
							tau_sn[i] = 1.0/1.16*(m[i]+2.0*mp)/(2.0*mp)*1.0/(nh/2.0*sigv)
						else: # ions   
							muuu=m[i]*2.0*mp/(m[i]+2.0*mp)
							if (i==1) or (i==2):
								sigv=2.4e-9 *(np.sqrt(8.0*kb*1.0e-7*T/(np.pi*muuu*1.0e-3))*1.0e-3)**0.6
							elif i==3:
								sigv=2.0e-9 * (np.sqrt(8.0*kb*1.0e-7*T/(np.pi*muuu*1.0e-3))*1.0e-3)**0.15
							elif i==4:
								sigv=3.89e-9 * (np.sqrt(8.0*kb*1.0e-7*T/(np.pi*muuu*1.0e-3))*1.0e-3)**(-0.02)
							else:
								sigv=1.69e-9
							tau_sn[i] = 1.0/1.14*(m[i]+2.0*mp)/(2.0*mp)*1.0/(nh/2.0*sigv)
						omega[i] = q[i]*B/(m[i]*clight)
						sigma[i] = resistivite_chimie_x[i+3,iH,iT,iX]*nh*(q[i])**2*tau_sn[i]/m[i]
						gamma_zeta[i] = 1.0
						gamma_omega[i] = 1.0
					for i in range(nbins_grains):
						# g+
						tau_sn[nion+1+3*i] = 1.0/1.28*(m_g[i]+2.0*mp)/(2.0*mp)*1.0/(nh/2.0*(np.pi*r_g[i]**2*(8.0*kb*T/(np.pi*2.0*mp))**0.5))
						omega [nion+1+3*i] = q[nion+1+3*i]*B/(m_g[i]*clight)
						# g-
						tau_sn[nion+2+3*i] = tau_sn[nion+1+3*i]
						omega [nion+2+3*i] = q[nion+2+3*i]*B/(m_g[i]*clight)
						sigma[nion+1+3*i] = resistivite_chimie_x[nion+4+3*i,iH,iT,iX]*nh*(q[nion+1+3*i]**2)*tau_sn[nion+1+3*i]/m_g[i]
						sigma[nion+2+3*i] = resistivite_chimie_x[nion+5+3*i,iH,iT,iX]*nh*(q[nion+2+3*i]**2)*tau_sn[nion+2+3*i]/m_g[i]
					sigP =0.0
					sigO =0.0
					sigH =0.0
					for i in range(nx[0]):
						sigP =sigP + sigma[i]
						sigO =sigO + sigma[i]/(1.0+(omega[i]*tau_sn[i])**2)
						sigH =sigH - sigma[i]*omega[i]*tau_sn[i]/(1.0+(omega[i]*tau_sn[i])**2)
					eta_ohm [iH,iT,iX,iB] = np.log10(1.0/sigP)                        # Ohmic
					eta_ad  [iH,iT,iX,iB] = np.log10(sigO/(sigO**2+sigH**2)-1.0/sigP) # Ambipolar
					eta_hall[iH,iT,iX,iB] = np.log10(abs(sigH/(sigO**2+sigH**2)))     # Hall
					eta_hsig[iH,iT,iX,iB] = sigH / abs(sigH)                          # Hall sign

	return eta_ohm, eta_ad, eta_hall, eta_hsig

def read_marchand_table(data, ndims):

	# Initialise offset counters and start reading data
	offsets = {"i":0, "n":0, "d":0}

	# Get table dimensions
	nx = np.roll(np.array(utils.read_binary_data(fmt="%ii"%ndims,content=data)),1)
	nx_read = np.copy(nx)
	nx_read[0] += 7
	# Now read the bulk of the table containing abundances in one go
	offsets["i"] += ndims
	offsets["d"] += 1
	resistivite_chimie_x = np.reshape(utils.read_binary_data(fmt="%id"%(np.prod(nx_read)),content=data,offsets=offsets),nx_read,order="F")
	print("Abundances table read successfully")
	del data

	return resistivite_chimie_x, nx

def read_resistivity_table(fname="resistivities_masson2016.bin"):

	print("Loading resistivity table: "+fname)

	# Read binary resistivity file
	with open(fname, mode='rb') as res_file:
		data = res_file.read()
	res_file.close()

	# Create table container
	theTable = dict()

	# Get length of record on first line to determine number of dimensions in table
	rec_size = utils.read_binary_data(fmt="i",content=data,correction=-4)
	ndims = int(rec_size[0]/4)

	if ndims == 3:
		theTable = get_masson_table(theTable, data, ndims)
	elif ndims == 4:
		resistivite_chimie_x, nx = read_marchand_table(data, ndims)
		nminchimie, nmaxchimie = np.min(resistivite_chimie_x[0]), np.max(resistivite_chimie_x[0])
		print("Computing 3D resistivity table...", end="")
		eta_ohm, eta_ad, eta_hall, eta_hsig = compute_resistivities(resistivite_chimie_x, nx, ndims)
		print(" done!")
		theTable = {"eta_ad":eta_ad, "eta_ohm":eta_ohm, "eta_hall":eta_hall, "eta_hsig":eta_hsig}
		theTable["scale_dens"] = 0.844*2.0/1.66e-24 # 2.0*H2_fraction/mH
		theTable["ionis_rate"] = 1e-17
		tminchimie, tmaxchimie = np.min(resistivite_chimie_x[1]), np.max(resistivite_chimie_x[1])
		ximinchimie, ximaxchimie = np.min(resistivite_chimie_x[2]), np.max(resistivite_chimie_x[2])
		bminchimie, bmaxchimie = 1e-10, 1e+10
		dens_arr = np.log10(np.logspace(np.log10(nminchimie), np.log10(nmaxchimie), eta_ad.shape[0]))
		T_arr = np.log10(np.logspace(np.log10(tminchimie), np.log10(tmaxchimie), eta_ad.shape[1]))
		xi_arr = np.log10(np.logspace(np.log10(ximinchimie), np.log10(ximaxchimie), eta_ad.shape[2]))
		B_arr = np.log10(np.logspace(np.log10(bminchimie), np.log10(bmaxchimie), eta_ad.shape[3]))
		theTable["grid"] = (dens_arr,T_arr,xi_arr,B_arr)
		print("3D resistivity table successfully computed")
	theTable["ndims"] = ndims

	return theTable

def get_resistivities(dataset, fname, variables=["eta_ohm","eta_ad","eta_hall"]):
	"""
	Create EOS variables from interpolation of eos table values in fname.
	"""

	default_units = {"eta_ohm":"s", "eta_ad":"s","eta_hall":"s"}

	if "res_table" not in dataset.meta:
		dataset.meta["res_table"] = read_resistivity_table(fname=fname)

	# get density scale from metadata
	try:
		rho_to_nH = dataset.meta["res_table"]["scale_dens"]/dataset.meta["mu_gas"]
	except KeyError:
		rho_to_nH = dataset.meta["res_table"]["scale_dens"]/2.31 # Default mean atomic weight

	ionisation_rate = np.ones(dataset["hydro"].shape)*dataset.meta["res_table"]["ionis_rate"]

	# check if density out of grid bounds:
	dens = np.copy(dataset["hydro"]["density"].to("g/cm^3").values*rho_to_nH)
	is_out_of_bounds_rho = np.log10(dens) > np.max(dataset.meta["res_table"]["grid"][0])
	if np.any(is_out_of_bounds_rho):
		print("WARNING: Density is out of resistivity grid bounds!")
		# proceed to fill those values with min grid trad
		dens[is_out_of_bounds_rho] = 10**np.max(dataset.meta["res_table"]["grid"][0])

	# check if temperature out of grid bounds:
	temp = np.copy(dataset["hydro"]["temperature"].to("K").values)
	is_out_of_bounds_temp = np.log10(temp) > np.max(dataset.meta["res_table"]["grid"][1])
	if np.any(is_out_of_bounds_temp):
		print("WARNING: Temperature is out of resistivity grid bounds!")
		# proceed to fill those values with min grid trad
		temp[is_out_of_bounds_temp] = 10**np.max(dataset.meta["res_table"]["grid"][1])

	if dataset.meta["res_table"]["ndims"] == 4:
		pts = np.array([np.log10(dens),
						np.log10(temp),
						np.log10(ionisation_rate),
						np.log10(dataset["hydro"]["B_field"].norm.to("G").values)]).T
	elif dataset.meta["res_table"]["ndims"] == 3:
		pts = np.array([np.log10(dens),
						np.log10(temp),
						np.log10(dataset["hydro"]["B_field"].norm.to("G").values)]).T

	for var in variables:
		print("Interpolating "+var+"...", end="")
		vals = ism_interpolate(dataset.meta["res_table"],dataset.meta["res_table"][var],pts)
		if var == "eta_hall":
			hall_sign = np.sign(ism_interpolate(dataset.meta["res_table"],dataset.meta["res_table"]["eta_hsig"],pts,in_log=True))
			dataset["hydro"][var] = Array(values = vals*hall_sign, unit = "s")
		else:
			dataset["hydro"][var] = Array(values = vals, unit = default_units[var])
	
		print(" done!")

