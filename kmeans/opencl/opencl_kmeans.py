import os
from common_data_importer import CommonFileDataImporter
from kmeans import DefaultKmeans
from kmeans_metric import EuclideanMetric
import numpy as np
import pyopencl as cl


class OpenCLKmeans(DefaultKmeans):
    def __init__(self, metric=EuclideanMetric(), importer=None, chunk_size=1000, max_steps=100):
        super(OpenCLKmeans, self).__init__(metric, importer, chunk_size, max_steps)
        self.ctx = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = OpenCLKmeans.load_cl_program(self.ctx, 'opencl_kmeans.cl')

    def kmeans_chunk_center(self, data, centers):
        k = len(centers)
        centers_counter = np.zeros((k, 1), dtype=np.int32)
        new_centers = np.asarray([np.zeros(self._dimension) for _ in xrange(k)], dtype=np.float32)
        data_assigns = np.empty((len(data), 1), dtype=np.int32)
        dim = np.asarray([centers.ndim], dtype=np.int32)
        n_centers = np.asarray([k], dtype=np.int32)

        # create buffers
        data_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
        centers_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=centers)
        assigns_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, data_assigns.nbytes)
        new_centers_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, new_centers.nbytes)
        centers_counter_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, centers_counter.nbytes)

        # run opencl extension
        self.prg.kmeans_chunk_center_cl(self.queue, (len(data), 1), None,
                                        data_buf,
                                        centers_buf,
                                        assigns_buf,
                                        new_centers_buf,
                                        centers_counter_buf,
                                        np.int32(dim),
                                        np.int32(n_centers)
        ).wait()

        # wait for it to finish and read out buffers
        cl.enqueue_read_buffer(self.queue, assigns_buf, data_assigns).wait()
        cl.enqueue_read_buffer(self.queue, new_centers_buf, new_centers).wait()
        cl.enqueue_read_buffer(self.queue, centers_counter_buf, centers_counter).wait()

        self._data_assigns.append(data_assigns)

        for i, center in enumerate(new_centers):
            new_centers[i] = sum(p for k, p in enumerate(data) if k == i)
            if centers_counter[i] > 0:
                new_centers[i] /= centers_counter[i]
            else:
                new_centers[i] = centers[i]
        return new_centers

    @staticmethod
    def load_cl_program(context, program):
        with open(str(os.path.dirname(os.path.realpath(__file__))) + '/' + program, "r") as f:
            cl_program = f.read()
        if cl_program is not None:
            return cl.Program(context, cl_program).build()
        return None

"""
    main
"""
#importer = CommonFileDataImporter(filename='data.txt')

#km = OpenCLKmeans(importer=importer, chunk_size=500, max_steps=2)
#km.calculate_centers(5, initial_centers=[[-0.3,0.3],[0,0],[0.5,0.5],[-0.5,-0.5],[-0.5,0.5]])
