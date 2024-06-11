import logging
from datetime import datetime
from pathlib import Path


class Logger:
    def __init__(self, root):
        self.LOGGER_FOLDER = Path(root, 'drive/MyDrive/Photo_Recognition/Logs')
        self.LOG_FILE_NAME = self.get_log_file()
        self.LOGGERS = {}

    def get_log_file(self):
        timestamp = datetime.now()
        name_str = timestamp.strftime('logs_%m/%d/%Y_%H:%M.log')
        if not self.LOGGER_FOLDER:
            raise Exception(f'Log folder path was not found')
        filename = Path(self.LOGGER_FOLDER, name_str.replace(':', '_').replace('/', '_'))
        return filename

    def get_logger(self, arg):
        if self.LOGGERS.get(arg):
            return self.LOGGERS.get(arg)
        else:
            logger_ = logging.getLogger(arg)
            logger_.setLevel(logging.DEBUG)
            handler = logging.FileHandler(self.LOG_FILE_NAME)
            formatter = logging.Formatter('%(asctime)s %(name)22s %(levelname)5s %(message)s')
            handler.setFormatter(formatter)
            logger_.addHandler(handler)
            self.LOGGERS[arg] = logger_
            return logger_
