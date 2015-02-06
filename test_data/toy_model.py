import numpy as np
from tica.Tica_Amuse import TicaAmuse
from hmm.algorithms import baum_welch_multiple
import hmm.kernel.c
from common.common_data_importer import CommonBinaryFileDataImporter
from kmeans.kmeans_main import kmeans

'''
    So far generated output:

    TICA: C extension not found, using Python implementation
    15001
    19:23:23 DEBUG kmeans_main: implementation chosen = <class 'opencl.opencl_kmeans.OpenCLKmeans'>
    Transition matrix:
    [[ 0.70012271  0.29997399]
     [ 0.29997399  0.70012271]]
'''
'''
Generate data for a toy model (for testing).
'''

# 1) Define a transition matrix P (metastable, i.e. should have elements close to 1 on the diagonal)
p11 = 0.97
p22 = 0.76
P = [[p11, 1 - p11], [1 - p22, p22]]

# 2) Generate a discrete trajectory (Markov chain) s(t) using P
s_t = [0]
for _ in range(0, 15000):
    r = np.random.random()
    state = s_t[-1]
    if r < P[state][state]:
        s_t = s_t + [state]
    else:
        s_t = s_t + [(state + 1) % 2]

# 3) Define an output probability distribution for each state i of P, e.g. a Gaussian
# with mean mu_i and covariance matrix C_i
mean_0, cov_0 = [0, 0], [[1.00, 0.50], [0.50, 1.00]]
mean_1, cov_1 = [5, 5], [[0.75, 0.25], [0.25, 0.75]]
normal_params = [[mean_0, cov_0], [mean_1, cov_1]]

# 4) For each time t, generate an output x(t) ~ G(mu_s(t), C_s(t))
x_t = []
for observation in s_t:
    params = normal_params[observation]
    x_t = x_t + [np.random.multivariate_normal(mean=params[0], cov=params[1])]

# 5) Define a rotation matrix and translation vector to rotate the low-dimensional x(t) into a high-
# dimensional space Omega. We get X(t)
angle = 0.333
transl = np.asarray([5, 5, 5, 5, 5], dtype=np.float32)
rot = np.matrix([[0, 0, np.cos(angle), -np.sin(angle), 0], [0, 0, np.sin(angle), np.cos(angle), 0]])
X_t = []
for distribution_value in x_t:
    x = np.asarray(rot.T.dot(distribution_value), dtype=np.float32)
    X_t.append(x[0])

# 6) Add Gaussian noise to all dimensions of Omega
mean, cov = [0, 0, 0, 0, 0], \
            [
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2]
            ]

for k in range(0, len(X_t)):
    X_t[k] = X_t[k] + np.random.multivariate_normal(mean=mean, cov=cov)

'''
    Reconstruction
'''

# I) Use TICA to find the relevant subspace of slow processes (should essentially invert the rotation
# matrix above)

binary_file = 'data.npy'
out_file = 'data_out.npy'
np.save(binary_file, X_t)

amuse = TicaAmuse(binary_file, out_file, i_addEps=1e-14)
amuse.performAmuse(2)

# II) Use k-means to discretize the data (must separate the two Gaussian distributions in order to
# work well)
k = 4
data_assigns = kmeans(k, importer=CommonBinaryFileDataImporter(filename=out_file))
print data_assigns
# III) Use HMM with 2 hidden states. You should be able to recover the transition matrix
# A, B, pi = hmm.utility.get_models()['equi32']
A = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=np.float32)
B = np.array(
    [
        [0.5/(0.5*k) for n in range(0, k)],
        [0.5/(0.5*k) for m in range(0, k)]
    ], dtype=np.float32)
pi = np.array([0.5, 0.5], dtype=np.float32)
data_assings = np.array(data_assigns)
d = len(data_assigns)/10
obs = np.array([ data_assigns[ x*d : x*d + d -1] for x in range(10)])
A, B, pi, eps, it = baum_welch_multiple(obs=obs, A=A, B=B, pi=pi, kernel=hmm.kernel.c, dtype=np.float32, maxit=100000)

print "Transition matrix:"
print A

# cleanup
# os.remove(binary_file)
# os.remove(out_file)
