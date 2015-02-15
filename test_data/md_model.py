import numpy as np
import os.path
from spscicomp.common.logger import Logger
from spscicomp.tica.Tica_Amuse import TicaAmuse
from spscicomp.common.common_data_importer import CommonBinaryFileDataImporter
from spscicomp.kmeans.kmeans_main import kmeans
from spscicomp.hmm.use_hmm import use_hmm

'''
    Results for
        time_lag = 1
        n_components = 4
        kmeans_k = 100
        state_count = 3

        16:44:02 DEBUG spscicomp.tica.Tica_PrincipleComp: TICA: C extension be work in progress, using Python implementation
        16:44:02 DEBUG __main__: Converting data...
        16:44:32 DEBUG __main__: TICA output file ticad_bpti.npy not found!
        16:44:32 DEBUG __main__: Running TICA...
        16:44:56 DEBUG __main__: k-means output file kmeansd_bpti.npy not found!
        16:44:56 DEBUG __main__: Running k-means...
        16:44:56 DEBUG spscicomp.kmeans.kmeans_main: implementation chosen = <class 'spscicomp.kmeans.opencl.opencl_kmeans.OpenCLKmeans'>
        17:14:26 DEBUG __main__: HMM output file hmmd_bpti.npy not found!
        17:14:26 DEBUG __main__: Running HMM...
        17:14:26 DEBUG spscicomp.hmm.use_hmm: C-Kernel used
        17:17:59 DEBUG spscicomp.hmm.use_hmm: Finished HMM iteration with likelihood -9.50877e+06
        17:21:26 DEBUG spscicomp.hmm.use_hmm: Finished HMM iteration with likelihood -1.08209e+07
        17:21:26 DEBUG __main__: Transition matrix:
            [[  9.97891188e-01   4.80510877e-04   1.62991369e-03]
             [  3.44174478e-04   9.99570131e-01   8.61332228e-05]
             [  2.04592384e-03   1.41762692e-04   9.97815371e-01]]
'''

LOG = Logger(__name__).get()

time_lag = 1
n_components = 4
kmeans_k = 100
state_count = 3

binary_file = 'bpti.npy'
tica_file = 'ticad_bpti.npy'
kmeans_file = 'kmeansd_bpti.npy'
hmm_file = 'hmmd_bpti.npy'

if not os.path.exists(binary_file):
    LOG.debug('Converting data...')
    import mdtraj as md
    bpti_traj = md.load('all.xtc', top='bpti-c-alpha.pdb')
    data_flat = np.reshape(bpti_traj.xyz, (bpti_traj.xyz.shape[0], bpti_traj.xyz.shape[1] * bpti_traj.xyz.shape[2]))
    np.save(binary_file, data_flat)

if not os.path.exists(tica_file):
    LOG.debug('TICA output file '+tica_file+' not found!')
    LOG.debug('Running TICA...')
    amuse = TicaAmuse(binary_file, tica_file, i_addEps=1e-14, i_timeLag=time_lag)
    amuse.performAmuse(i_numDomComp=n_components)

if not os.path.exists(kmeans_file):
    LOG.debug('k-means output file '+kmeans_file+' not found!')
    LOG.debug('Running k-means...')
    data_assigns = np.array(kmeans(kmeans_k, importer=CommonBinaryFileDataImporter(filename=tica_file)), np.int16)
    np.save(kmeans_file, data_assigns)
else:
    data_assigns = np.load(kmeans_file)

if not os.path.exists(hmm_file):
    LOG.debug('HMM output file '+hmm_file+' not found!')
    LOG.debug('Running HMM...')
    d = len(data_assigns) / 10
    obs = np.array([data_assigns[x * d: x * d + d - 1] for x in range(10)])
    A, B, pi = use_hmm(observations=obs, state_count=state_count, symbol_count=kmeans_k, retries=2)
    np.save(hmm_file, A)
else:
    A = np.load(hmm_file)

LOG.debug("Transition matrix:\n" + str(A))
