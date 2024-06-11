from configparser import ConfigParser
from pathlib import Path


class ConfigRead:
    def __init__(self, package_path):
        self.config_file_path = Path(package_path, 'config.ini')
        self.config_read = ConfigParser()
        self.config_read.read(self.config_file_path)

    def parse_value(self, section, field):
        read_value = self.config_read[section][field]
        return read_value

    def parse_list(self, section, field):
        section_list = self.parse_value(section, field).split(',')
        return section_list

    def parse_dict(self, section, field):
        section_list = self.parse_list(section, field)
        default_list_value = [0] * len(section_list)
        dict_data = dict(zip(section_list, default_list_value))
        return dict_data
