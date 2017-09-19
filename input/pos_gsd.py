from __future__ import division
from __future__ import print_function
import numpy as np
import tempfile

from glotzformats.reader import PosFileReader
from glotzformats.reader import GSDHoomdFileReader

import hoomd
hoomd.context.initialize()

f_in = "Zn6Sc_2x2x2"
f_out = "Zn6Sc_2x2x2"

reader = PosFileReader()
gsd_reader = GSDHoomdFileReader()

traj = reader.read(open(f_in+".pos","r"))

snapshot = traj[-1].make_snapshot()
system = hoomd.init.read_snapshot(snapshot)

hoomd.dump.gsd(f_out+'.gsd',period=None,overwrite=False,group=hoomd.group.all())

# make sure the file was written properly
traj_gsd = gsd_reader.read(open(f_out+'.gsd','rb'))
assert((traj_gsd[0].positions == traj[-1].positions).all())
assert((traj_gsd[0].orientations == traj[-1].orientations).all())
assert((np.asarray(traj_gsd[0].types) == np.asarray(traj[-1].types)).all())
