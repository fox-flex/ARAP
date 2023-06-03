import sys
from tqdm import tqdm
from enum import IntEnum
from datetime import datetime


class Logger:
    class Mode(IntEnum):
        ERROR = 0
        WARNING = 1
        INFO = 2
        DEBUG = 3
    
    mode: Mode = Mode.INFO
    pref: str = 'main'
    mode2str: dict = {
        Mode.ERROR: 'ERROR',
        Mode.WARNING: 'WARNING',
        Mode.INFO: 'INFO',
        Mode.DEBUG: 'DEBUG',
    }
    
    @staticmethod
    def set_log_level(mode: Mode):
        Logger.mode = mode
    
    @staticmethod
    def set_prefisx(pref: str):
        Logger.pref = pref
    
    @staticmethod
    def reset_prefisx():
        Logger.pref = 'main'
    
    @staticmethod
    def prepr_str(s: str, mode: Mode):
        now = datetime.now()
        dt_str = now.strftime("%d/%m/%Y %H:%M:%S")
        s = f'[{dt_str}][{Logger.pref}][{Logger.mode2str[mode]}] {s}'
        return s
    
    @staticmethod
    def log_debug(s: str):
        if Logger.mode >= Logger.Mode.DEBUG:
            s = Logger.prepr_str(s, Logger.Mode.DEBUG)
            print(s)
    
    @staticmethod
    def log_info(s: str):
        if Logger.mode >= Logger.Mode.INFO:
            s = Logger.prepr_str(s, Logger.Mode.INFO)
            print(s)

    @staticmethod
    def log_warning(s: str):
        if Logger.mode >= Logger.Mode.WARNING:
            s = Logger.prepr_str(s, Logger.Mode.WARNING)
            print(s, file=sys.stderr)

    @staticmethod
    def log_error(s: str):
        if Logger.mode >= Logger.Mode.ERROR:
            s = Logger.prepr_str(s, Logger.Mode.ERROR)
            print(s, file=sys.stderr)
    
    @staticmethod
    def tqdm_debug(data, total = None, desc: str = ''):
        if Logger.mode >= Logger.Mode.DEBUG:
            if total is None:
                total = len(data)
            desc = Logger.prepr_str(desc, Logger.Mode.DEBUG)
            return tqdm(data, total=total, desc=desc)
        else:
            return data
