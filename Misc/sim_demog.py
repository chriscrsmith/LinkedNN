# e.g.     python sim_demog.py <simid> <t1 min>,<t1 max> <Na min>,<Na max> <N1 min>,<N1 max> <simid min>,<simid max> <outdir>  

import msprime
import sys
from scipy.stats import loguniform
import numpy as np
import os

simid = int(sys.argv[1])
t1_range = list(map(float,sys.argv[2].split(",")))
Na_range = list(map(float,sys.argv[3].split(",")))
N1_range = list(map(float,sys.argv[4].split(",")))
outdir = sys.argv[5]
os.makedirs(outdir + "/TreeSeqs/", exist_ok=True)
os.makedirs(outdir + "/Targets/", exist_ok=True)
fp = outdir + "/TreeSeqs/output_" + str(simid) + "_recap.trees"

if os.path.exists(fp) is True:
    sys.exit(0)

np.random.seed(seed=simid)
Na = loguniform.rvs(Na_range[0], Na_range[1], size=1)[0]
N1 = loguniform.rvs(N1_range[0], N1_range[1], size=1)[0]
t1 = loguniform.rvs(t1_range[0], t1_range[1], size=1)[0]
targets = np.array([Na, t1, N1])
np.save(outdir + "/Targets/target_" + str(simid) + ".npy", targets)
demography = msprime.Demography()
demography.add_population(initial_size=N1)
demography.add_population_parameters_change(time=t1, initial_size=Na)
ts = msprime.sim_ancestry(samples=10,
                          demography=demography,
                          random_seed=simid+1,  # (seeds must be greater than 0)
                          recombination_rate=1e-8,
                          sequence_length=1e8,)
ts.dump(fp)
