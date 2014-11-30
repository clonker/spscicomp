import unittest

from common_data_importer import *


class TestCommonDataImporter(unittest.TestCase):
    def setUp(self):
        pass

    def test_common_file_data_importer(self):
        f_name = 'common/unittest_data.txt'
        importer = CommonFileDataImporter(filename=f_name)
        line_count = sum(1 for _ in open(f_name))
        for i in xrange(5):
            curr_line_count = 0
            while True:
                chunk = importer.get_data(1000)
                curr_line_count += len(chunk)
                if not importer.has_more_data():
                    break
            importer.rewind()
            msg = 'Reading file. Expected line count: ' + str(line_count) + ', actual line count: ' + \
                  str(curr_line_count)
            self.assertEqual(line_count, curr_line_count, msg)


"""
    main
"""

if __name__ == '__main__':
    unittest.main()