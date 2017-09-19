# MD code to study stability, melting and growth of Zn-Sc
# approximants to Tsai QCs
# AIM "Square" meeting, San Jose, CA 2017

from __future__ import division
from __future__ import print_function

import numpy as np
import os, math
import hoomd
from hoomd import md
from hoomd import deprecated

hoomd.context.initialize()

Sc_N = 100
## number of Zn atoms is 6*Sc_N
Zn_N = 6*Sc_N
## total number of atoms
tot_N = Sc_N + Zn_N

## number density
number_density = 0.03

## temperature
T = 0.25

## Parameters acquired manually from doi:10.1038/nmat2044
## Zn Zn params
ZnZn_rmin = 2.0
ZnZn_d = dict(c1   = 5389.82,
              c2   = -0.506073,
              eta1 = 11.9575,
              eta2 = 3.10219,
              k    = 3.95513,
              phi  = -10.5802)

## Zn Sc params
ZnSc_rmin = 2.0
ZnSc_d = dict(c1   = 2.49701e7,
              c2   = 3.18315,
              eta1 = 20.176,
              eta2 = 3.3025,
              k    = 2.96421,
              phi  = -5.91716)

## Sc Sc params
ScSc_rmin = 2.0
ScSc_d = dict(c1   = 70.907,
              c2   = 38.7497,
              eta1 = 7.16201,
              eta2 = 5.71453,
              k    = 2.86125,
              phi  = -6.87912)

timeSteps = 50e6

filename = "ZnSc_QC"
init_file = None
restart_period = 1e3
dump_period = 10

# OPP defined as in doi: 10.1103/PhysRevB.85.092102
def OPP(r, rmin, rmax, c1, c2, eta1, eta2, k, phi):
    cos = math.cos(k*r + phi)
    sin = math.sin(k*r + phi)
    V = c1*pow(r, -eta1) + c2*cos*pow(r, -eta2)
    F = eta1*c1*pow(r, -eta1-1) + eta2*c2*cos*pow(r, -eta2-1) + k*c2*sin*pow(r, -eta2)
    return (V, F)

# Determine the potential range by searching for extrema
def determineRange(max_num_extrema, d):
    r = 2.0
    extrema_num = 0
    force1 = OPP(r, d['c1'], d['c2'], d['eta1'], d['eta2'], d['k'], d['phi'])[1]
    while (extrema_num < max_num_extrema and r < 8.0):
        r += 1e-5
        force2 = OPP(r, d['c1'], d['c2'], d['eta1'], d['eta2'], d['k'], d['phi'])[1]
        if (force1 * force2 < 0.0):
            extrema_num += 1
        force1 = force2
    return r

if init_file is None:
    if os.path.isfile(filename + '_restart.gsd'):
        system = hoomd.init.read_gsd(filename = filename + '_restart.gsd')
    else:
        # Initialize a system with particles placed at random with a given density
        nn = np.ceil(tot_N**(1.0/3.0))

        # Appropriate interparticle spacing and box dimensions
        L = pow(tot_N/number_density, 1./3.)
        sigma = L/nn

        snapshot = hoomd.data.make_snapshot(N=tot_N, box=hoomd.data.boxdim(Lx=L, Ly=L, Lz=L, xy=0, xz=0, yz=0), particle_types=['Zn','Sc'])

        for m in range(tot_N):
            z_ind = int(m/nn/nn)
            xy_ind = m%(nn*nn)
            y_ind = int(xy_ind/nn)
            x_ind = xy_ind%nn
            pos = np.array([x_ind*sigma, y_ind*sigma, z_ind*sigma]) - L/2.
            snapshot.particles.position[m] = pos
            if m < Zn_N:
                snapshot.particles.typeid[m] = 0
            else:
                snapshot.particles.typeid[m] = 1

        system = hoomd.init.read_snapshot(snapshot)
else:
    # Initialize the system from the input file
    system = hoomd.init.read_gsd(filename = init_file)

# Generate the pair interaction table
nl = md.nlist.cell()
table = md.pair.table(width = 1000, nlist = nl)

# find cutoff for each potential
ZnZn_rcut = 8. #determineRange(6, **ZnZn_d)
ZnSc_rcut = 8. #determineRange(5, **ZnSc_d)
ScSc_rcut = 8. #determineRange(4, **ScSc_d)

table.pair_coeff.set('Zn', 'Zn', func = OPP, rmin = ZnZn_rmin, rmax = ZnZn_rcut, coeff = ZnZn_d)
table.pair_coeff.set('Zn', 'Sc', func = OPP, rmin = ZnSc_rmin, rmax = ZnSc_rcut, coeff = ZnSc_d)
table.pair_coeff.set('Sc', 'Sc', func = OPP, rmin = ScSc_rmin, rmax = ScSc_rcut, coeff = ScSc_d)

# Start logging
# 1. set up the gsd restart file
gsd_restart = hoomd.dump.gsd(filename = filename + '_restart.gsd', group = hoomd.group.all(), truncate = True, period = restart_period)
# 2. set up the pos dump file
pos = deprecated.dump.pos(filename = filename+'.pos', period = dump_period)
# 3. set up the gsd dump file
gsd = hoomd.dump.gsd(filename = filename+'.gsd', group = hoomd.group.all(), period = dump_period)
# 4. set up the log file
logger = hoomd.analyze.log(filename = filename + ".log", period = int(dump_period/10),
quantities = ['time','potential_energy','pressure'])

# Integrate at constant temperature
nvt = md.integrate.nvt(group = hoomd.group.all(), tau = 1.0, kT = T)
md.integrate.mode_standard(dt = 0.01)
hoomd.run(timeSteps + 1)
