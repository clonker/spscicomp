import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from spscicomp.common.common_data_importer import CommonFileDataImporter
from spscicomp.kmeans import DefaultKmeans
from spscicomp.kmeans.kmeans_data_generator import KmeansRandomDataGenerator
from spscicomp.kmeans.opencl.opencl_kmeans import OpenCLKmeans


#data_generator = KmeansRandomDataGenerator(100000, 2, 3)
f_name = 'unittest_data.txt'
#data_generator.to_file(f_name)

centers = 5
#f_name = '../unittest_data.txt'
importer = CommonFileDataImporter(filename=f_name)
opencl_kmeans = OpenCLKmeans(importer=importer, chunk_size=50000, max_steps=250)

c, assigns, history = opencl_kmeans.calculate_centers(centers, return_centers=True, save_history=True)
print assigns

plt.figure()

_colors = cm.rainbow(np.linspace(0, 1, len(c)))

for i, center in enumerate(c):
    if type(center) is list:
        center = center[0]
    plt.plot(center[0], center[1], linestyle='None', marker='o', color=_colors[i, :])

importer.rewind()
data = importer.get_data(10000000)
for idx, p in enumerate(data):
    j = assigns[idx]
    plt.plot(p[0], p[1], linestyle='None', marker='.', color=_colors[j])

plt.show()