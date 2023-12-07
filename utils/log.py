import logging

LOG_LEVEL = {'notset': logging.NOTSET,
             'debug': logging.DEBUG,
             'info': logging.INFO,
             'warning': logging.WARNING,
             'error': logging.ERROR,
             'critical': logging.CRITICAL
             }


def setup_custom_logger(name, log_level):
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    #logging.DEBUG
    logger.setLevel(log_level)
    logger.addHandler(handler)
    return logger