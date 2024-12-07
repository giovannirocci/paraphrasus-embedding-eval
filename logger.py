import logging, sys

class MyLog:
    logger = None

    def create_logger(self, name: str = 'logs'):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(name + ".log", mode="w+")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stdout_handler)
        return logger

    def __init__(self):
        self.logger = self.create_logger()

    def get_logger(self) -> logging.Logger:
        return self.logger

mylog = MyLog()