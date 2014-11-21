from abc import ABCMeta, abstractmethod
import numpy as np


class CommonDataImporter:
    """
    This is an abstract data importer class. Implementations are expected to override the get_data and has_more_data
    methods.
    """

    def __init__(self):
        pass

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_data(self, size):
        raise NotImplementedError('Subclasses must override get_data()!')

    def rewind(self):
        pass

    @abstractmethod
    def has_more_data(self):
        raise NotImplementedError('Subclasses must override has_more_data()!')


class CommonSimpleDataImporter(CommonDataImporter):
    """"Import" data from a given data array."""

    def __init__(self, data):
        super(CommonSimpleDataImporter, self).__init__()
        self._data = data

    def get_data(self, size):
        """
        Return all available data regardless of the requested size.

        :param size: Size of data which is to be returned. This parameter is disregarded as all data is returned.
        :returns: All data.
        """
        return self._data

    def has_more_data(self):
        """
        Return if there is any more data. As all data is returned when using get_data, this function always returns False.

        :return: False since there never is any more data.
        """
        return False


class CommonFileDataImporter(CommonDataImporter):
    """
    Import data from a text file. The data structure should be as follows:
    One point occupies one line.
    Each point consists of several floats with space as a separator.
    """

    def __init__(self, filename):
        """
        Initialize the class variables and initialize the file input stream.

        :param filename: File from which data is to be loaded.
        """
        super(CommonFileDataImporter, self).__init__()
        self._fileName = filename
        self._file = None
        self._fileLineEnum = None
        self._hasMoreData = True
        self.init_file_input_stream()

    def init_file_input_stream(self):
        """Initialize the file input stream, that is, open the file and create the iterator on the file's lines."""
        self._file = open(name=self._fileName)
        self._fileLineEnum = enumerate(self._file)

    # @profile
    def get_data(self, size):
        """
        Return a numpy array of floats where each data point occupies one row of the array. The data is read from
        the current position of the pointer onwards. If the pointer reaches the end of the file, an array of all data
        points up to the end of the file is returned, the file is closed and the hasMoreData flag is set to False.

        :param size: Number of data points to be returned.
        :return: A numpy array of data points.
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
                self.close_file()
                break

        return np.array(data)

    def rewind(self):
        """
        Reset the file pointer to the beginning, that is, initialize the file and set the hasMoreData flag to True.
        """
        self.init_file_input_stream()
        self._hasMoreData = True

    def has_more_data(self):
        """
        Test if the pointer is at the end of the file or not.

        :return: True if there is more data after the pointer, and False if not.
        """
        return self._hasMoreData

    def close_file(self):
        """Close the file handle if it is open."""
        if self._file is not None and not self._file.closed:
            self._file.close()


class CommonBinaryFileDataImporter(CommonDataImporter):
    """
    Import data from a binary file. The file format should be as generated by numpy.save.
    """
    def __init__(self, filename):
        """
        Initialize the class variables and initialize the file input stream.

        :param filename: Binary file from which data is to be loaded.
        """
        super(CommonBinaryFileDataImporter, self).__init__()
        self._fileName = filename
        self._file = None
        self._hasMoreData = True
        self._position = 0
        self.init_file_input_stream()

    def init_file_input_stream(self):
        """Create a numpy array object which reads from the binary file using a memmap."""
        self._file = np.load(self._fileName, mmap_mode='r')

    # @profile
    def get_data(self, size):
        """
        Return a numpy array of floats where each data point occupies one row of the array. The data is read from
        the current position of the pointer onwards. If the pointer reaches the end of the file, an array of all data
        points up to the end of the file is returned and the hasMoreData flag is set to False.

        :param size: Number of data points to be returned.
        :return: A numpy array of data points.
        """
        start_position = self._position
        end_position = start_position + size - 1
        if len(self._file) < end_position:
            end_position = len(self._file)
        self._position += size
        if self._position > len(self._file):
            self._hasMoreData = False

        return self._file[start_position:end_position]

    def rewind(self):
        """
        Reset the file pointer to the beginning and set the hasMoreData flag to True.
        """
        self._position = 0
        self._hasMoreData = True

    def has_more_data(self):
        """
        Test if the pointer is at the end of the file or not.

        :return: True if there is more data after the pointer, and False if not.
        """
        return self._hasMoreData
