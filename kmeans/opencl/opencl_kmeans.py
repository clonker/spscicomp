import os

import numpy as np
import pyopencl as cl

from kmeans import DefaultKmeans
from kmeans_metric import EuclideanMetric
import logger


LOG = logger.Logger(__name__).get()


class OpenCLKmeans(DefaultKmeans):
    def __init__(self, metric=EuclideanMetric(), importer=None, chunk_size=1000, max_steps=100):
        super(OpenCLKmeans, self).__init__(metric, importer, chunk_size, max_steps)
        platform = cl.get_platforms()
        gpus = platform[0].get_devices(device_type=cl.device_type.CPU)
        self.ctx = cl.Context(devices=gpus)
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = OpenCLKmeans.load_cl_program(self.ctx, 'opencl_kmeans.cl')
        self.n_work_groups = sum(gpu.max_compute_units for gpu in gpus)

    # @profile
    def kmeans_chunk_center(self, data, centers):
        data = data.astype(np.float32)
        k = len(centers)
        dim = len(centers[0])
        centers = np.asarray(centers, dtype=np.float32)
        out = np.zeros((len(data), 1), dtype=np.float32)
        centers_counter = np.asarray(k * [0] * self.n_work_groups, dtype=np.int32)
        new_centers = np.asarray(self.n_work_groups * [np.zeros(self._dimension) for _ in xrange(k)], dtype=np.float32)
        data_assigns = np.empty((len(data), 1), dtype=np.int32)
        LOG.debug("data="+str(data))
        LOG.debug("centers=" + str(centers))
        LOG.debug("k=" + str(k) + ", dim=" + str(dim))
        LOG.debug("n_work_groups=" + str(self.n_work_groups))
        # create buffers
        data_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
        centers_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=centers)
        assigns_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, data_assigns.nbytes)
        new_centers_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, new_centers.nbytes)
        centers_counter_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, centers_counter.nbytes)
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=out)

        # run opencl extension
        #self.prg.kmeans_chunk_center_cl(self.queue, (len(data), 1), None,
        #                                data_buf,
        #                                centers_buf,
        #                                centers_counter_buf,
        #                                assigns_buf,
        #                                new_centers_buf,
        #                                out_buf,
        #                                np.int32(dim),
        #                                np.int32(k)
        #).wait()

        self.prg.kmeans_chunk_center_cl(self.queue, (len(data),), None, data_buf, assigns_buf, centers_buf, centers_counter_buf, new_centers_buf, out_buf, np.int32(dim), np.int32(k)).wait()

        # wait for it to finish and read out buffers
        #cl.enqueue_read_buffer(self.queue, assigns_buf, data_assigns).wait()
        cl.enqueue_read_buffer(self.queue, new_centers_buf, new_centers).wait()
        cl.enqueue_read_buffer(self.queue, centers_counter_buf, centers_counter).wait()
        cl.enqueue_read_buffer(self.queue, out_buf, out).wait()

        centers_counter = centers_counter[:k]
        new_centers = new_centers[:k]

        self._data_assigns.append(data_assigns)
        LOG.debug("cc=" + str(centers_counter))
        LOG.debug("new_centers=" + str(new_centers))
        LOG.debug("assigns=" + str(data_assigns))
        LOG.debug("out=" + str(out))

        return new_centers

    @staticmethod
    def load_cl_program(context, program):
        with open(str(os.path.dirname(os.path.realpath(__file__))) + '/' + program, "r") as f:
            cl_program = f.read()
        if cl_program is not None:
            return cl.Program(context, cl_program).build()
        return None
