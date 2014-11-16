from abc import ABCMeta, abstractmethod
import numpy as np


class KmeansDataImporter:
    """ abstract data importer """

    def __init__(self):
        pass

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_data(self, size):
        raise NotImplementedError('subclasses must override get_data()!')

    def rewind(self):
        pass

    @abstractmethod
    def has_more_data(self):
        raise NotImplementedError('subclasses must override has_more_data()!')


class KmeansFileDataImporter(KmeansDataImporter):
    """
    Import data from files. The data structure should be:
    One point occupies on line.
    Each point consist several dimensions with space as separator.
    """
    def __init__(self, filename):
        super(KmeansFileDataImporter, self).__init__()
        self._fileName = filename
        self._file = None
        self._fileLineEnum = None
        self._hasMoreData = True
        self.init_file_input_stream()

    def init_file_input_stream(self):
        self._file = open(name=self._fileName)
        self._fileLineEnum = enumerate(self._file)

    def get_data(self, size):
        """
        When get_data is called, a list of #size data is returned.
        """
        data = []
        for i in xrange(0, size):
            line = next(self._fileLineEnum, None)
            if line is not None:
                _, line = line
                parts = line.split()
                data.append(np.array(parts, dtype=np.float))
            else:
                self._hasMoreData = False
                self._file.close()
                break

        return data

    def rewind(self):
        """
        Reset the file pointer to the beginning.
        """
        self.init_file_input_stream()
        self._hasMoreData = True

    def has_more_data(self):
        """
        Test the pointer is at the end of the file or not.
        """
        return self._hasMoreData

    def close_file(self):
        if self._file is not None and not self._file.closed:
            self._file.close()