import logging

LOG_LEVEL = {'notset': logging.NOTSET,
             'debug': logging.DEBUG,
             'info': logging.INFO,
             'warning': logging.WARNING,
             'error': logging.ERROR,
             'critical': logging.CRITICAL
             }


def setup_custom_logger(name: str, log_level: int, log_path: str):
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    if log_path == '':
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(filename=log_path, encoding='utf-8')

    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.addHandler(handler)
    return logger
