import torchvision.transforms.v2 as transforms


CustomTransform = transforms.Grayscale

"""
The v2.Grayscale does not work with np.ndarray.
It only works with TvTensors or Tensor, all assume that the input is (C, H, W).
The output remains the same class as the input.


With TvTensors:
>>> a = np.random.randint(0, 255, size=(10, 10, 3)).astype('uint8')
>>> a = v2.ToImage()(a);  b = v2.Grayscale()(a)
>>> type(a), a.shape, a.min(), a.max(), a.dtype
<class 'torchvision.tv_tensors._image.Image'>, torch.Size([3, 10, 10]), tensor(1, dtype=torch.uint8), tensor(253, dtype=torch.uint8), torch.uint8
>>> type(b), b.shape, b.min(), b.max(), b.dtype
<class 'torchvision.tv_tensors._image.Image'>, torch.Size([1, 10, 10]), tensor(43, dtype=torch.uint8), tensor(218, dtype=torch.uint8), torch.uint8


With Tensor: only works with (C, H, W)
>>> a = torch.randint(0, 255, size=(3, 10, 10), dtype=torch.uint8)
>>> b = v2.Grayscale()(a)
>>> type(a), a.shape, a.min(), a.max(), a.dtype
<class 'torch.Tensor'>, torch.Size([3, 10, 10]), tensor(0, dtype=torch.uint8), tensor(254, dtype=torch.uint8), torch.uint8
>>> type(b), b.shape, b.min(), b.max(), b.dtype
<class 'torch.Tensor'>, torch.Size([1, 10, 10]), tensor(29, dtype=torch.uint8), tensor(230, dtype=torch.uint8), torch.uint8

Trying to use (H, W, C) will raise an error
>>> a = torch.randint(0, 255, size=(10, 10, 3), dtype=torch.uint8);
>>> b = v2.Grayscale()(a)
ValueError: too many values to unpack (expected 3)


With np.ndarray: even if the input is (C, H, W)...
>>> a = np.random.randint(0, 255, size=(3, 10, 10)).astype('uint8')
>>> b = v2.Grayscale()(a)
>>> type(a), a.shape, a.min(), a.max(), a.dtype
<class 'numpy.ndarray'>, (3, 10, 10), 0, 254, dtype('uint8')
>>> type(b), b.shape, b.min(), b.max(), b.dtype
<class 'numpy.ndarray'>, (3, 10, 10), 0, 254, dtype('uint8')

... or (H, W, C) will not work.
>>> a = np.random.randint(0, 255, size=(10, 10, 3)).astype('uint8')
>>> b = v2.Grayscale()(a)
>>> type(a), a.shape, a.min(), a.max(), a.dtype
<class 'numpy.ndarray'>, (10, 10, 3), 0, 254, dtype('uint8')
>>> type(b), b.shape, b.min(), b.max(), b.dtype
<class 'numpy.ndarray'>, (10, 10, 3), 0, 254, dtype('uint8')
"""