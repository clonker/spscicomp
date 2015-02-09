import os
import string

import numpy as np
import pyopencl as cl

from spscicomp.kmeans.kmeans import DefaultKmeans
from spscicomp.kmeans.kmeans_metric import EuclideanMetric
from spscicomp.common.logger import Logger


LOG = Logger(__name__).get()


class OpenCLKmeans(DefaultKmeans):
    def __init__(self, metric=EuclideanMetric(), importer=None, chunk_size=1000, max_steps=100):
        super(OpenCLKmeans, self).__init__(metric, importer, chunk_size, max_steps)
        self.kernel = None
        self.ctx = None
        self.queue = None
        self.prg = None
        self.n_work_groups = None

    def __initialize_program(self, dim, k):
        platform = cl.get_platforms()
        devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        self.ctx = cl.Context(devices)
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = OpenCLKmeans.load_cl_program(self.ctx, 'opencl_kmeans.cl', dim, k)
        self.n_work_groups = sum(device.max_compute_units for device in devices)

    def kmeans_chunk_center(self, data, centers):
        data = data.astype(np.float32)
        centers = centers.astype(np.float32)
        k = len(centers)
        dim = len(centers[0])

        if not self.prg:
            self.__initialize_program(dim, k)

        out = np.zeros((10, 1), dtype=np.float32)
        new_centers = np.asarray(self.n_work_groups * [np.zeros(self._dimension, dtype=np.float32) for _ in xrange(k)], dtype=np.float32)
        data_assigns = np.empty((len(data), 1), dtype=np.int32)

        # create buffers
        data_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
        centers_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=centers)
        assigns_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, data_assigns.nbytes)
        new_centers_buf = cl.Buffer(
            self.ctx, cl.mem_flags.WRITE_ONLY, self.n_work_groups * dim * k * np.dtype('float32').itemsize
        )
        centers_counter_buf = cl.Buffer(
            self.ctx, cl.mem_flags.WRITE_ONLY, k*self.n_work_groups * np.dtype('int32').itemsize
        )
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=out)

        e = cl.enqueue_barrier(self.queue)
        e.wait()

        # run opencl extension
        self.prg.kmeans_chunk_center_cl(
            self.queue,
            (len(data),),
            None,
            assigns_buf,
            data_buf,
            centers_buf,
            centers_counter_buf,
            new_centers_buf, out_buf
        )

        # barrier
        e = cl.enqueue_barrier(self.queue)
        e.wait()

        # wait for it to finish and read out buffers
        cl.enqueue_copy(self.queue, data_assigns, assigns_buf)
        cl.enqueue_copy(self.queue, new_centers, new_centers_buf)
        cl.enqueue_copy(self.queue, out, out_buf)

        new_centers = new_centers[:k]
        self._data_assigns.extend(data_assigns.flatten().tolist())

        return new_centers.astype(dtype=np.float32)


    @staticmethod
    def load_cl_program(context, program, dimension, n_centers):
        with open(str(os.path.dirname(os.path.realpath(__file__))) + '/' + program, "r") as f:
            cl_program = f.read()
        if cl_program is not None:
            cl_program = string.Template(cl_program).substitute(
                DIM=dimension,
                K=n_centers,
                ACCURACY='float'
            )
            return cl.Program(context, cl_program).build()
        return None
