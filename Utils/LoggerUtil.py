import logging
import os


class LoggerUtil():
    @staticmethod
    def get_logger(log_name="Vector"):
        logger = logging.getLogger(log_name)
        return logger

    @staticmethod
    def init_logger(log_name, log_file):
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))
        log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(filename)s -   %(message)s',
                                       datefmt='%m/%d/%Y %H:%M:%S')

        logger = logging.getLogger(log_name)
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        logger.handlers = [console_handler]
        file_handler = logging.FileHandler(log_file, encoding="utf8", mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
