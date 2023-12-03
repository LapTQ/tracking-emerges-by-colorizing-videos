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
        self.n_samples = kwargs['n_samples']
        self.input_transform = kwargs.get('input_transform', None)
        self.label_transform = kwargs.get('label_transform', None)

        self.imW = 512
        self.imH = 512


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
            inputs.append(frame)
            labels.append(frame)
        
        inputs.append(target_image)
        labels.append(target_image)

        if self.input_transform:
            inputs = [self.input_transform(input_.copy()) for input_ in inputs]
        if self.label_transform:
            labels = [self.label_transform(label.copy()) for label in labels]

        inputs = [torch.from_numpy(input_) if not isinstance(input_, torch.Tensor) \
                  else input_ \
                    for input_ in inputs]
        labels = [torch.from_numpy(label_) if not isinstance(label_, torch.Tensor) \
                    else label_ \
                        for label_ in labels]

        inputs = torch.stack(inputs, dim=0)
        labels = torch.stack(labels, dim=0)
        
        return inputs, labels

