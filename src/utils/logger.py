import logging
import sys
import datetime
import os


def load_logger(
        **kwargs
):
    level = kwargs.get('level', logging.INFO)
    format = kwargs.get('format', '%(asctime)s\t|%(funcName)20s |%(lineno)d\t|%(levelname)8s |%(message)s')
    directory = kwargs.get('directory', None)
    handlers = kwargs.get('handlers', 'stdout')

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
    if not isinstance(handlers, list):
        handlers = [handlers]
    for handler in handlers:
        if handler == 'stdout':
            handlers_.append(logging.StreamHandler(sys.stdout))
        elif isinstance(handler, str):
            assert directory is not None, 'Logging directory must be specified when using file handler.'
            assert os.path.isdir(directory), 'Logging directory must be a directory.'
            log_fpath = os.path.join(directory, handler)
            handlers_.append(logging.FileHandler(log_fpath))
        else:
            raise ValueError('Invalid logging handler: {}'.format(handler))
    handlers = handlers_

    logging.basicConfig(
        level=level,
        format=format,
        handlers=handlers
    )

    return logging.getLogger(__name__)


def parse_save_checkpoint_path(
        **kwargs
):
    """Preprocess the inputed path for saving checkpoint file.
    
    If the input path is `None`, return `None`.
    If the input path is a file, the input path will be returned.
    If the input path is a directory, a new file with the current datetime will be created.
    """
    input_path = kwargs.get('input_path')
    ext = kwargs.get('ext')

    if input_path is None:
        return None
    
    input_path = input_path.strip().rstrip('/')
    
    # check if the path is a directory or file
    parent, filename = os.path.split(input_path)
    is_dir = '.' not in filename

    os.makedirs(parent, exist_ok=True)
    if is_dir:
        os.makedirs(input_path, exist_ok=True)
        target_filename = 'checkpoint.{}'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'), ext)
        checkpoint_path = os.path.join(input_path, target_filename)
    else:
        checkpoint_path = input_path
    
    return checkpoint_path
        

def parse_load_checkpoint_path(
        **kwargs
):
    """Preprocess the inputed path for loading checkpoint file.

    If the input path is `None`, return `None`.
    If the input path is a file, return the input path if exists, otherwise return -1.
    If the input path is a directory, return path to the last checkpoint file if exists, otherwise return -1.
    """
    input_path = kwargs.get('input_path')
    ext = kwargs.get('ext')

    logger = logging.getLogger(__name__)

    if input_path is None:
        return None
    
    input_path = input_path.strip().rstrip('/')
    
    if not os.path.exists(input_path):
        logger.warning('Checkpoint path was set but {} does not exist.'.format(input_path))
        return -1

    if os.path.isdir(input_path):
        filenames = [f for f in os.listdir(input_path) if f.endswith(ext)]
        if len(filenames) == 0:
            logger.warning('No checkpoint .{} file exists in {}.'.format(ext, input_path))
            return -1
        last = sorted(filenames)[-1]
        logger.warning('{} is a directory. So new checkpoint will be created.'.format(input_path))
        
        file_path = os.path.join(input_path, last)
        logger.info('Loading the lastest checkpoint at {}'.format(file_path))
    else:
        assert input_path.endswith(ext), 'Checkpoint path must be a .{} file.'.format(ext)
        logger.warning('{} is a file. So it will be overwritten.'.format(input_path))
        file_path = input_path

    return file_path
            
        

