from wmi import WMI
from re import search
from pathlib import Path
from logger import  get_logger

root_dir = Path().resolve()
logger = get_logger('validation_login')


def validateLogin():
    try:
        pars = WMI().Win32_ComputerSystem()[0]
        fnd = 'UserName'
        id_ = getattr(pars, fnd, '')
        cop, gpid = id_.split('\\')
        f = search('[a-zA-Z]', gpid)
        logger.debug(f'Login data was extracted successful')
        if cop == 'CWWPVT' or f is None:
            return True
        else:
            return False
    except Exception as err:
        logger.debug(f'validateLogin error: {err.__class__.__name__} {err}')
