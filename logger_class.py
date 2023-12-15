import logging
from datetime import datetime
from os import path
from pathlib import Path

root_dir = Path().resolve()


class Logger:
    def __init__(self):
        # self.logger_folder = Path(root_dir, 'app_sys/logs/')
        self.logger_folder = None
        self.log_filename = self.get_log_file()

    @staticmethod
    def get_log_folder_path():
        log_folder = Path(root_dir, 'Logs/')

        if not Path(log_folder).exists():
            log_folder = Path(root_dir, 'Logs/')
            Path.mkdir(log_folder, exist_ok=True)

        return log_folder

    def get_log_file(self):
        timestamp = datetime.now()
        name_str = timestamp.strftime('logs_%m/%d/%Y_%H:%M:%S.log')
        self.logger_folder = self.get_log_folder_path()
        filename = path.join(self.logger_folder, name_str.replace(':', '_').replace('/', '_'))
        return filename

    def get_logger(self, arg):
        """https://stackoverflow.com/questions/45701478/log-from-multiple-python-files-into-single-log-file-in-python"""
        log_format = '%(asctime)s %(name)22s %(levelname)5s %(message)s'
        logging.basicConfig(level=logging.DEBUG,
                            format=log_format,
                            filename=self.log_filename,
                            filemode='w',
                            force=True)
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(logging.Formatter(log_format))
        # logging.getLogger(name).addHandler(console) >> duplicate log lines
        logging.getLogger(arg)
        print(f'INFO: Logger root folder: {root_dir}')
        print(f'INFO: Logger folder: {self.logger_folder}')
        print(f'INFO: Logger filename: {self.log_filename}')

        return logging.getLogger(arg)
