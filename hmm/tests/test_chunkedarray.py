import unittest
import numpy
from hmm.utility import ChunkedArray



class TestChunkedArray(unittest.TestCase):

    def test_index(self):
        chunk = ChunkedArray(10, 2)
        numpy.testing.assert_equal(chunk.get_num_chunks(), 5, 'Chunkcount is not correct')
        chunk = ChunkedArray(11, 2)
        numpy.testing.assert_equal(chunk.get_num_chunks(), 6, 'Chunkcount is not correct')

if __name__ == '__main__':
    unittest.main()
