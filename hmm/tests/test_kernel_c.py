import spscicomp.hmm.tests.test_kernel_python
import unittest
import spscicomp.hmm.kernel.c
import spscicomp.hmm.utility
import numpy

class TestCounts64(spscicomp.hmm.tests.test_kernel_python.TestCounts):
    kernel = spscicomp.hmm.kernel.c
    dtype = numpy.float64
    A, B, pi = spscicomp.hmm.utility.get_models()['equi64']

class TestScaling64(spscicomp.hmm.tests.test_kernel_python.TestScaling):
    kernel = spscicomp.hmm.kernel.c
    dtype = numpy.float64
    A, B, pi = spscicomp.hmm.utility.get_models()['equi64']

class TestCounts32(spscicomp.hmm.tests.test_kernel_python.TestCounts):
    kernel = spscicomp.hmm.kernel.c

class TestScaling32(spscicomp.hmm.tests.test_kernel_python.TestScaling):
    kernel = spscicomp.hmm.kernel.c

class TestForward64(spscicomp.hmm.tests.test_kernel_python.TestForward):
    kernel = spscicomp.hmm.kernel.c
    dtype = numpy.float64
    A, B, pi = spscicomp.hmm.utility.get_models()['equi64']

class TestForward32(spscicomp.hmm.tests.test_kernel_python.TestForward):
    kernel = spscicomp.hmm.kernel.c

class TestCallErrors64(spscicomp.hmm.tests.test_kernel_python.TestCallErrors):
    kernel = spscicomp.hmm.kernel.c
    dtype = numpy.float64
    A, B, pi = spscicomp.hmm.utility.get_models()['equi64']

class TestCallErrors32(spscicomp.hmm.tests.test_kernel_python.TestCallErrors):
    kernel = spscicomp.hmm.kernel.c
    dtype = numpy.float32


if __name__ == '__main__':
    unittest.main()
