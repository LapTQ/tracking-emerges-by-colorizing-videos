from torch.utils.data import Dataset
import numpy as np
import cv2
import torch
from torchvision.transforms import ToTensor

class CustomDataset(Dataset):

    def __init__(
            self,
            **kwargs
    ):
        # parse kwargs
        self.n_references = kwargs['n_references']
        self.image_size = kwargs['image_size']      # (W, H)
        self.n_samples = kwargs['n_samples']

        self.imW = self.image_size[0]
        self.imH = self.image_size[1]


    def __len__(self):
        return self.n_samples
    
    
    def __getitem__(self, index):
        
        reference_images = [
            np.random.randint(0, 255, size=(self.imH, self.imW, 3), dtype=np.uint8)
            for _ in range(self.n_references)
        ]
        target_image = np.random.randint(0, 255, size=(self.imH, self.imW, 3), dtype=np.uint8)

        input_ = []
        label = []
        for frame in reference_images:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            input_.append(ToTensor()(gray.copy()))
            label.append(ToTensor()(frame.copy()))
        
        gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        input_.append(ToTensor()(gray.copy()))
        label.append(ToTensor()(target_image.copy()))

        input_ = torch.stack(input_, dim=0)
        label = torch.stack(label, dim=0)
        
        return input_, label

