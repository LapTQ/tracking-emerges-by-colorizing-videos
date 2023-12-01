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
        
        brighten_step = 20
        color = np.random.randint(0, 255 - brighten_step * self.n_references, size=(3,), dtype=np.uint8)
        img = np.tile(color, self.imW * self.imH).reshape(self.imH, self.imW, 3)
        
        reference_images = [
            (img + _ * brighten_step).astype(np.uint8)
            for _ in range(self.n_references)
        ]
        target_image = img + self.n_references * brighten_step

        inputs = []
        labels = []
        for frame in reference_images:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            inputs.append(ToTensor()(gray.copy()))
            labels.append(ToTensor()(frame.copy()))
        
        gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        inputs.append(ToTensor()(gray.copy()))
        labels.append(ToTensor()(target_image.copy()))

        inputs = torch.stack(inputs, dim=0)
        labels = torch.stack(labels, dim=0)
        
        return inputs, labels

