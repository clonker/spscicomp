import numpy as np
import argparse
import sys
from spscicomp.common.logger import Logger
from spscicomp.tica.Tica_Amuse import TicaAmuse
from spscicomp.common.common_data_importer import CommonBinaryFileDataImporter
from spscicomp.kmeans.kmeans_main import kmeans
from spscicomp.hmm.use_hmm import use_hmm


'''
Generate data for a toy model (for testing).
'''

LOG = Logger(__name__).get()

parser = argparse.ArgumentParser(description='Test the pipeline of algorithms using a random toy model.')
parser.add_argument('--plot', action='store_true', help='show plots')
parser.add_argument('-s', '--steps', default='1500', type=int, help='number of random walk steps (default: 1500)')
parser.add_argument('-k', default='4', type=int, help='number of clusters for the k-means algorithm (default: 4)')
args = parser.parse_args()
LOG.debug("Options given: " + str(args))

generate_plots = True  # args.plot
n_random_walk_steps = args.steps
kmeans_k = args.k

if generate_plots:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import pylab
    except:
        LOG.error("Failed to import matplotlib even though generate_plots was set to true.")

# 1) Define a transition matrix P (metastable, i.e. should have elements close to 1 on the diagonal)
p11 = 0.97
p22 = 0.76
P = [[p11, 1 - p11], [1 - p22, p22]]

LOG.debug("Initial transition matrix:\n" + str(P))

# 2) Generate a discrete trajectory (Markov chain) s(t) using P
s_t = []
for _ in range(0, n_random_walk_steps):
    r = np.random.random()
    if len(s_t) > 0:
        state = s_t[-1]
    else:
        state = np.random.random_integers(2) - 1

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

if generate_plots:
    LOG.debug("Generating plot for random walk scattered by normal distribution in 2D.")
    x = [ob[0] for ob in x_t]
    y = [ob[1] for ob in x_t]
    plt.ioff()
    plt.figure(1)
    plt.subplot(411)
    plt.title('Normal distribution of random walk')
    plt.scatter(x, y)


# 5) Define a rotation matrix and translation vector to rotate the low-dimensional x(t) into a high-
# dimensional space Omega. We get X(t)
angle = 3.141
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

if generate_plots:
    LOG.debug("Generating plot for rotated (by " + str(angle) + ") random walk with Gaussian background noise.")
    plt.figure(1)
    plt.subplot(412)
    plt.title('Projection of normal distribution of random walk, rotated and with noise')
    plt.scatter([ob[2] for ob in X_t], [ob[3] for ob in X_t])

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

if generate_plots:
    LOG.debug("Generating plot for obfuscated random walk after TICA processing.")
    importer = CommonBinaryFileDataImporter(filename=out_file)
    data = None
    while importer.has_more_data():
        data = importer.get_data(n_random_walk_steps)
    importer.close_file()
    plt.figure(1)
    plt.subplot(413)
    plt.title('TICA result')
    plt.scatter([ob[0] for ob in data], [ob[1] for ob in data])



# II) Use k-means to discretize the data (must separate the two Gaussian distributions in order to
# work well)
data_assigns = kmeans(kmeans_k, importer=CommonBinaryFileDataImporter(filename=out_file))

if generate_plots:
    LOG.debug("Generating plot for kmeans clustering.")
    colors = cm.rainbow(np.linspace(0, 1, kmeans_k))
    importer = CommonBinaryFileDataImporter(filename=out_file)
    data = None
    while importer.has_more_data():
        data = importer.get_data(n_random_walk_steps)
    importer.close_file()
    plt.figure(1)
    plt.subplot(414)
    plt.title('K-Means result')
    for k, x_t in enumerate(data):
        plt.plot(x_t[0], x_t[1], linestyle='None', marker='.', color=colors[data_assigns[k]])

    # fix axes
    x_lim = [sys.maxint, -sys.maxint]
    y_lim = [sys.maxint, -sys.maxint]
    for i in [1, 2, 3, 4]:
        axes = plt.subplot(4, 1, i)
        x_lim = [min(axes.get_xlim()[0], x_lim[0]), max(axes.get_xlim()[1], x_lim[1])]
        y_lim = [min(axes.get_ylim()[0], y_lim[0]), max(axes.get_ylim()[1], y_lim[1])]
    for i in [1, 2, 3, 4]:
        axes = plt.subplot(4, 1, i)
        axes.set_xlim(x_lim)
        axes.set_ylim(y_lim)
    # show plot
    plt.show()


# III) Use HMM with 2 hidden states. You should be able to recover the transition matrix
data_assigns = np.array(data_assigns)
d = len(data_assigns) / 10
obs = np.array([data_assigns[x * d: x * d + d - 1] for x in range(10)])
A, B, pi = use_hmm(observations=obs, state_count=2, symbol_count=kmeans_k)

if generate_plots:
    fig = plt.figure(2)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_aspect('equal')
    plt.title('Initial transition matrix')
    plt.imshow(P, interpolation='nearest', cmap=cm.ocean)
    plt.colorbar()

    ax = fig.add_subplot(1, 2, 2)
    ax.set_aspect('equal')
    plt.title('Estimated transition matrix')
    plt.imshow(A, interpolation='nearest', cmap=cm.ocean)
    plt.colorbar()

    plt.show()

LOG.debug("Estimated transition matrix:\n" + str(A))
