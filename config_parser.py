import configparser
import zipfile
from pathlib import Path, PurePath
from logger import get_logger

root_dir = Path().resolve()
config_file = PurePath(root_dir, 'app_sys/', 'config.ini')
version_file = PurePath(root_dir, 'app_sys/', 'version_.ini')

logger = get_logger('config_parser')


# detection_all_image, image_process, download_id, version


def config_parser(section=None, mode=None, field=None, value=None):
    # Parse to config file
    config = configparser.RawConfigParser()

    # Check section config file
    if section == 'config':
        config.read(config_file)
    else:
        config.read(version_file)

    # Check mode
    if mode == 'read':
        # Extract local setting
        value_get = config.get('main', field)
        logger.debug(f'{field} was parsed successful')
        return value_get

    elif mode == 'write':
        # Modify setting
        config.set('main', field, value)

        with open(config_file, 'w') as configfile:
            config.write(configfile)
            logger.debug(f'{field} was update to {value}')


def update_version_parse(file):

    zf = zipfile.ZipFile(file)
    zf_config = zf.open("version_.ini", "r")
    zf_config_data = zf_config.read().decode('ascii')

    config = configparser.ConfigParser()
    config.read_string(zf_config_data)

    version_get = config.get('main', 'version')
    return version_get



