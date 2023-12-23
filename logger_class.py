import logging
from datetime import datetime
from os import path
from pathlib import Path

root_dir = Path().resolve()


class Logger:
    def __init__(self, log_folder):
        self.LOGGER_FOLDER = self.get_log_folder_path(log_folder)
        self.LOG_FILE_NAME = self.get_log_file()
        self.LOGGERS = {}

    @staticmethod
    def get_log_folder_path(log_folder):
        if log_folder is None:
            log_folder = Path(root_dir, 'Logs/')
        else:
            log_folder = Path(log_folder)

        if not Path(log_folder).exists():
            log_folder = Path(root_dir, 'Logs/')
            Path.mkdir(log_folder, exist_ok=True)

        return log_folder

    def get_log_file(self):
        timestamp = datetime.now()
        # name_str = timestamp.strftime('logs_%m/%d/%Y_%H:%M:%S.log')
        name_str = timestamp.strftime('logs_%m/%d/%Y_%H:%M.log')
        # self.logger_folder = self.get_log_folder_path()
        filename = path.join(self.LOGGER_FOLDER, name_str.replace(':', '_').replace('/', '_'))
        return filename

    def get_logger(self, arg):
        if self.LOGGERS.get(arg):
            return self.LOGGERS.get(arg)
        else:
            logger = logging.getLogger(arg)
            logger.setLevel(logging.DEBUG)
            handler = logging.FileHandler(self.LOG_FILE_NAME)
            formatter = logging.Formatter('%(asctime)s %(name)22s %(levelname)5s %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            self.LOGGERS[arg] = logger
            return logger


