from torch.utils.data import Dataset
import numpy as np
import cv2
import torch
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


class CustomDataset(Dataset):

    def __init__(
            self,
            **kwargs
    ):
        # parse kwargs
        self.dataset_dir = kwargs['dataset_dir']
        self.n_references = kwargs['n_references']
        self.n_samples = kwargs['n_samples']
        self.frame_rate = kwargs['frame_rate']
        self.input_transform = kwargs.get('input_transform', None)
        self.label_transform = kwargs.get('label_transform', None)

        self.video_paths = self._load_video_paths()
        self.first_size = None
    

    def _load_video_paths(self):
        video_paths = []
        for video_path in Path(self.dataset_dir).glob('**/*.mp4'):
            video_paths.append(str(video_path))
        return video_paths
    

    def _sample_video(self):
        while True:
            video_path = np.random.choice(self.video_paths)
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                break
            logger.warning(f'Video {video_path} is not opened. Sampling again...')
        return video_path, cap
    

    def _sample_frames(self, cap):
        fps = cap.get(cv2.CAP_PROP_FPS)
        step_size = int(fps / self.frame_rate)
        
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) < self.n_references * step_size + 1:
            return False, *([None] * self.n_references), None
        
        ret = []
        running_idx = np.random.randint(cap.get(cv2.CAP_PROP_FRAME_COUNT) - self.n_references * step_size - 1)
        i = 0
        while True:
            if len(ret) == self.n_references + 1:
                break
            _, frame = cap.read()
            if self.first_size is None:
                self.first_size = frame.shape[:2][::-1]
                logger.warning('The first video sampled is of size {}'.format(self.first_size))
            else:
                if frame.shape[:2][::-1] != self.first_size:
                    frame = cv2.resize(frame, self.first_size)
            if i == running_idx:
                ret.append(frame)
                running_idx += step_size
            i += 1
        
        return True, *ret


    def __len__(self):
        return self.n_samples

    
    def __getitem__(self, index):

        while True:
            video_path, cap = self._sample_video()
            success, *reference_images, target_image = self._sample_frames(cap)
            if success:
                break
            logger.warning(f'Video {video_path} is too short. Sampling again...')

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