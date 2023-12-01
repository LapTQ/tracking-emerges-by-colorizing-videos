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

    
    def _create_fake_frames(
            self,
            **kwargs
    ):
        # parse kwargs
        mode = kwargs.get('mode', 'shape')

        if mode == 'shape':
            return self._create_fake_square()
        elif mode == 'mono':
            return self._create_fake_mono()
        else:
            raise NotImplementedError()


    def _create_fake_mono(self):
        brighten_step = 20
        color = np.random.randint(0, 255 - brighten_step * self.n_references, size=(3,), dtype=np.uint8)
        img = np.tile(color, self.imW * self.imH).reshape(self.imH, self.imW, 3)

        return ((img + _ * brighten_step)
            for _ in range(self.n_references + 1))
        
    
    def _create_fake_square(self):
        n_shapes = 2
        square_size = min(self.imW, self.imH) // 2
        
        colors = np.random.randint(0, 255, size=(n_shapes, 3), dtype=np.uint8)
        indexes = np.random.randint(0, self.imW * self.imH, size=(n_shapes,), dtype=np.uint32)
        indexes = np.stack([indexes // self.imW, indexes % self.imW], axis=1)

        ret = []
        for _ in range(self.n_references + 1):
            img = 100 + np.zeros((self.imH, self.imW, 3), dtype=np.uint8)
            indexes = np.random.randint(0, self.imW * self.imH, size=(n_shapes,), dtype=np.uint32)
            indexes = np.stack([indexes // self.imW, indexes % self.imW], axis=1)
            for i in range(n_shapes):
                x, y = indexes[i]
                img[max(x - square_size//2, 0):x + square_size//2, max(y - square_size//2, 0):y + square_size//2] = colors[i]
            img = cv2.blur(img, (square_size//2, square_size//2))
            ret.append(img)

        return ret

    
    def __getitem__(self, index):      
        
        *reference_images, target_image = self._create_fake_frames()

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

