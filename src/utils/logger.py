import logging
import sys
import datetime
import os


def load_logger(
        **kwargs
):
    level = kwargs.get('level', logging.INFO)
    format = kwargs.get('format', '%(asctime)s\t|%(funcName)20s |%(lineno)d\t|%(levelname)8s |%(message)s')
    handlers = kwargs.get('handlers', [sys.stdout])
    datetime_format = kwargs.get('datetime_format', '%Y%m%d_%H%M%S')

    if level == 'DEBUG':
        level = logging.DEBUG
    elif level == 'INFO':
        level = logging.INFO
    elif level == 'WARNING':
        level = logging.WARNING
    elif level == 'ERROR':
        level = logging.ERROR
    elif level == 'CRITICAL':
        level = logging.CRITICAL
    else:
        raise ValueError('Invalid logging level: {}'.format(level))

    handlers_ = []
    for handler, value in handlers.items():
        if handler == 'file':
            os.makedirs(value, exist_ok=True)
            log_fname = datetime.datetime.now().strftime('{}.log'.format(datetime_format))
            log_fpath = os.path.join(value, log_fname)  # assuming value is a directory
            handlers_.append(logging.FileHandler(log_fpath))
        elif handler == 'stream':
            if value == 'stdout':
                value = sys.stdout
            else:
                raise ValueError('Invalid logging stream: {}'.format(value))
            handlers_.append(logging.StreamHandler(value))
        else:
            raise ValueError('Invalid logging handler: {}'.format(handler))
    handlers = handlers_

    logging.basicConfig(
        level=level,
        format=format,
        handlers=handlers
    )

    return logging.getLogger(__name__)
            
        

