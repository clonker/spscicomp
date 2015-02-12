import numpy as np
from spscicomp.tica.Tica_Amuse import TicaAmuse
from matplotlib import pyplot as plt
from spscicomp.tica.common_data_importer import CommonBinaryFileDataImporter

############################################
# tica patch of toy_model.py in dev branch
############################################

# 1) Define a transition matrix P (metastable, i.e. should have elements close to 1 on the diagonal)
p11 = 0.97
p22 = 0.76
P = [[p11, 1 - p11], [1 - p22, p22]]

# 2) Generate a discrete trajectory (Markov chain) s(t) using P
s_t = [0]
for _ in range(0, 1500):
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

amuse = TicaAmuse(binary_file, out_file, i_addEps=1e-9)
amuse.performAmuse(2)

sepImporter  = CommonBinaryFileDataImporter('data_out.npy')
sepData      = sepImporter.get_data(len(sepImporter._file))

plt.subplot(1, 2, 1)
plt.plot(sepData[:, 0], sepData[:, 1], 'o')
plt.title('Scatter Plot of 1st and 2nd ICs')
plt.ylabel('2nd IC')
plt.xlabel('1st IC')

plt.subplot(1, 2, 2)
plt.hist(sepData[:, 0], 100)
plt.title('Histogram of 1st IC')
plt.show()
