from abc import ABCMeta, abstractmethod


class KmeansDataImporter:
    """ abstract metric """

    def __init__(self):
        pass

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_chunk(self, offset, size):
        raise NotImplementedError('subclasses must override calculate_centers()!')

class FileDataImoprter(KmeansDataImporter):

    def __init__(self, fileName):
        super(FileDataImoprter, self).__init__()
        self._filename = fileName

    def get_chunk(self, offset, size):
        pass