from abc import ABCMeta, abstractmethod
import numpy as np


class KmeansDataImporter:
    """ abstract metric """

    def __init__(self):
        pass

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_data(self, size):
        raise NotImplementedError('subclasses must override calculate_centers()!')


class KmeansFileDataImporter(KmeansDataImporter):
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
        data = []
        for i in range(0, size):
            line = next(self._fileLineEnum, None)
            if line is not None:
                _, line = line
                parts = line.split()
                data.append(np.array(parts, dtype=np.float))
            else:
                self._hasMoreData = False
                break

        return data

    def rewind(self):
        self._file.close()
        self.init_file_input_stream()

    def has_more_data(self):
        return self._hasMoreData