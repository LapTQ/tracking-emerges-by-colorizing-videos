from torch.utils.data import Dataset, DataLoader
import importlib


class BaseMyDataset(Dataset):
    
    def __init__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self, index):
        raise NotImplementedError()
    

class BaseMyDataLoader(DataLoader):
    pass


def load_module(
        **kwargs
):
    module_name = kwargs['module_name']
    class_name = kwargs.get('class_name', 'MyDataLoader')

    module = importlib.import_module(
        name='{}.{}'.format(__name__, module_name)
    )
    cls_ = getattr(module, class_name)

    return cls_
    

