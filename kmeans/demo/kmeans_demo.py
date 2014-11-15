from kmeans import DefaultKmeans
from kmeans_data_importer import *
from kmeans_plot import *


def plot_data(data):
    global centers, colors, kmeans
    for i, p in enumerate(data):
        j = DefaultKmeans.closest_center(DefaultKmeans(), p, centers)
        plt.plot(p[0], p[1], linestyle='None', marker='.', color=colors[j])


def plot_centers():
    global centers, colors
    for i, center in enumerate(centers):
        if type(center) is list:
            center = center[0]
        plt.plot(center[0], center[1], linestyle='None', marker='o', color=colors[i, :])


k = 3
points = 1000
colors = cm.rainbow(np.linspace(0, 1, k))

data_file = "demo/data.txt"

importer = KmeansFileDataImporter(filename=data_file)

kmeans = DefaultKmeans(importer=importer)

_, history = kmeans.calculate_centers(k, save_history=True)

plt.ion()
plt.figure()
plt.show()

for i in xrange(len(history)):
    plt.clf()
    centers = history[i]
    plot_data(importer.get_data(points))
    plot_centers()
    plt.pause(0.5)
    importer.rewind()
plt.show()

importer.close_file()