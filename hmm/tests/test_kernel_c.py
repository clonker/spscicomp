import hmm.tests.test_kernel_python
import unittest
import hmm.kernel.c
import hmm.utility
import numpy

class TestCounts64(hmm.tests.test_kernel_python.TestCounts):
    kernel = hmm.kernel.c
    dtype = numpy.float64
    A, B, pi = hmm.utility.get_models()['equi64']

class TestScaling64(hmm.tests.test_kernel_python.TestScaling):
    kernel = hmm.kernel.c
    dtype = numpy.float64
    A, B, pi = hmm.utility.get_models()['equi64']

class TestCounts32(hmm.tests.test_kernel_python.TestCounts):
    kernel = hmm.kernel.c

class TestScaling32(hmm.tests.test_kernel_python.TestScaling):
    kernel = hmm.kernel.c

class TestForward64(hmm.tests.test_kernel_python.TestForward):
    kernel = hmm.kernel.c
    dtype = numpy.float64
    A, B, pi = hmm.utility.get_models()['equi64']

class TestForward32(hmm.tests.test_kernel_python.TestForward):
    kernel = hmm.kernel.c

class TestCallErrors64(hmm.tests.test_kernel_python.TestCallErrors):
    kernel = hmm.kernel.c
    dtype = numpy.float64
    A, B, pi = hmm.utility.get_models()['equi64']

class TestCallErrors32(hmm.tests.test_kernel_python.TestCallErrors):
    kernel = hmm.kernel.c
    dtype = numpy.float32


if __name__ == '__main__':
    unittest.main()
