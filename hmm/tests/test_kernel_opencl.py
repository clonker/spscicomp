import hmm.kernel.opencl
import hmm.kernel.python
import unittest
import numpy
import pyopencl

# Observation Sequence
ob = numpy.array([1, 0, 1, 1, 0, 1, 1, 0])

T = len(ob)

# initial conditions
A = numpy.array(
    [[ 0.333, 0.333, 0.333 ],
     [ 0.333, 0.333, 0.333 ],
     [ 0.333, 0.333, 0.333 ]])

B = numpy.array(
    [[ 1.0, 0.0 ],
     [ 0.5, 0.5 ],
     [ 0.0, 1.0 ]])

pi = numpy.array([ 0.333, 0.333, 0.333])

N = len(pi)

M = len(B[0])

ctx = hmm.kernel.opencl.Context(N, M)



class TestForwardNoScaling(unittest.TestCase):

    def test_naive_is_correct_for_small_case(self):
        cl_alpha = hmm.kernel.opencl.forward_no_scaling_naive(ctx, A, B, pi, ob)
        alpha    = numpy.zeros((T,N), numpy.float32)
#        C = numpy.zeros((T,N,N), numpy.float32)
#        pyopencl.enqueue_copy(ctx.queue, C, ctx.matrices)
#        print C
        pyopencl.enqueue_copy(ctx.queue, alpha, cl_alpha)
        _, alpha2 = hmm.kernel.python.forward_no_scaling(A, B, pi, ob)
        numpy.testing.assert_array_almost_equal(alpha, alpha2)

    def test_naive_many_work_group_configurations(self):
        for work_groups in range(1, T):
            for work_units in range(2, T):
                ctx.set_work_group_sizes(work_groups, work_units)
                cl_alpha = hmm.kernel.opencl.forward_no_scaling_naive(ctx, A, B, pi, ob)
                alpha    = numpy.zeros((T,N), numpy.float32)
                pyopencl.enqueue_copy(ctx.queue, alpha, cl_alpha)
                _, alpha2 = hmm.kernel.python.forward_no_scaling(A, B, pi, ob)
                numpy.testing.assert_array_almost_equal(alpha, alpha2)

    def test_naive_1000_observation_len(self):
        ob = numpy.loadtxt('data/t1.1000.0.dat', numpy.int16)
        cl_alpha = hmm.kernel.opencl.forward_no_scaling_naive(ctx, A, B, pi, ob)
        _, alpha2 = hmm.kernel.python.forward_no_scaling(A, B, pi, ob)
        alpha    = numpy.zeros((1000,N), numpy.float32)
        pyopencl.enqueue_copy(ctx.queue, alpha, cl_alpha)
        numpy.testing.assert_array_almost_equal(alpha, alpha2)

    # Check if we get infinity as result. Should be, because of no scaling.
    def test_naive_2000_observation_len(self):
        ob = numpy.loadtxt('data/t1.2000.dat', numpy.int16)
        _ = hmm.kernel.opencl.forward_no_scaling_naive(ctx, A, B, pi, ob)


if __name__ == '__main__':
    unittest.main()
