#!/usr/bin/python2

import numpy as np
import matplotlib.pyplot as plt


def Dist(x, y):
    return np.sqrt(np.sum((x-y)**2))


def RandomDataSet(n, d, k):
    centers = RandomCenters(k, d)
    
    data = []
    for i in xrange(0, n):
        data.append(np.random.rand(d) + centers[np.random.randint(0, k)])
    
    return data


def RandomCenters(k, d):
    centers = []
    for i in xrange(0, k):
        centers.append(np.random.rand(d)*2)

    return centers


def ZeroCenters(k, d):
    centers = []
    for i in xrange(0, k):
        centers.append(np.zeros(d))

    return centers


def ClosestCenter(p, centers):
    min_dist = float('inf')
    closest_center = 0
    for i, center in enumerate(centers):
        dist = Dist(p, center)
        if dist < min_dist:
            min_dist = dist
            closest_center = i
    
    return int(closest_center)


def KmeansIterate(data, centers):
    centers_counter = np.zeros(k)
    new_centers = ZeroCenters(k, d)
       
    for p in data:
        closest_center = ClosestCenter(p, centers)
        new_centers[closest_center] += p
        centers_counter[closest_center] += 1

    for i, center in enumerate(new_centers):
        if centers_counter[i] > 0:
            new_centers[i] /= centers_counter[i]

    return new_centers


def MiniBatchKmeansIterate(data, centers, batch_size, centers_counter):
    mini_batch_centers = [list([]) for _ in xrange(batch_size)]

    for i in xrange(0, batch_size):
        j = np.random.randint(0, n)
        closest_center = ClosestCenter(data[j], centers)
        mini_batch_centers[closest_center].append(j)

    for i, mini_batch_center in enumerate(mini_batch_centers):
        for j in mini_batch_center:
            centers_counter[i] += 1
            eta = 1.0 / centers_counter[i]
            centers[i] = centers[i] * (1.0 - eta) + data[j] * eta

    return centers, centers_counter


def Kmeans(data, centers, max_steps):
    old_centers = ZeroCenters(k, d)
    i = 1

    while not np.array_equal(centers, old_centers):
        old_centers = centers
        centers = KmeansIterate(data, centers)
        #print centers
    
        i += 1
        if i > max_steps:
            break

    print i

    return centers


# See http://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf for a description of the algorithm.
def MiniBatchKmeans(data, centers, batch_size, max_steps):
    centers_counter = np.zeros(k)
    old_centers = ZeroCenters(k, d)
    i = 1

    while True:#not np.array_equal(centers, old_centers):
        old_centers = centers
        centers, centers_counter = MiniBatchKmeansIterate(data, centers, batch_size, centers_counter)

        i += 1
        if i > max_steps:
            break

    print i

    return centers



n = 1000
d = 2
k = 2
max_steps = 100
batch_size = 20

data = RandomDataSet(n, d, k)
centers = RandomCenters(k, d)
#print centers

#centers = Kmeans(data, centers, max_steps)
centers = MiniBatchKmeans(data, centers, batch_size, max_steps)
print centers

if k < 5:
    colors = ['blue', 'green', 'red', 'yellow']
    
    data_arr = np.array(data)
    centers_arr = np.array(centers)

    for i, p in enumerate(data):
        j = ClosestCenter(p, centers)
        plt.plot(data_arr[i, 0], data_arr[i, 1], linestyle='None', marker='.', color=colors[j])

    for j, center in enumerate(centers):
        plt.plot(centers_arr[j, 0], centers_arr[j, 1], linestyle='None', marker='o', color=colors[j])

    plt.show()

