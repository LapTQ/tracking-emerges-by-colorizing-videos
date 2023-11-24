import numpy as np
import cv2

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


class BaseDataset(Dataset):
    
    def __init__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self, index: int):
        raise NotImplementedError()
    

class KineticsDataset(BaseDataset):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        