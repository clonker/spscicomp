import logging


class Logger(object):
    def __init__(self, name):
        logger = logging.getLogger(name)  # log_namespace can be replaced with your namespace
        log_level = logging.DEBUG
        logger.setLevel(log_level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            handler.setLevel(log_level)
            logger.addHandler(handler)
        self._logger = logger

    def get(self):
        return self._logger