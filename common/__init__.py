"""
This package contains classes which may be of use in all parts of the project.

The DataImporter classes provide methods to do sequential reads of a specific size from a specified data source.
The basic usage is as follows:
-------------------------------------
while True:
    data = importer.get_data(size)
    if len(data) is 0:
        break
    else:
        # Do something
    if not importer.has_more_data():
        break
importer.rewind()
-------------------------------------
"""