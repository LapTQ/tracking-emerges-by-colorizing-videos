import torchvision.transforms.v2 as transforms


CustomTransform = transforms.ToImage

"""
v2.ToImage will convert (H, W, C) to (C, H, W)
if the input is a np.ndarray whatever the dtype of the input is
and does not perform any scaling

>>> a = np.random.randint(0, 255, size=(10, 10, 3)).astype('uint8');  b = v2.ToImage()(a)
>>> a.shape, a.min(), a.max(), a.dtype
(10, 10, 3), 0, 253, dtype('uint8')
>>> b.shape, b.min(), b.max(), b.dtype
torch.Size([3, 10, 10]), tensor(0, dtype=torch.uint8), tensor(253, dtype=torch.uint8), torch.uint8

>>> a = np.random.randint(0, 255, size=(10, 10, 3)).astype('float32');  b = v2.ToImage()(a)
>>> a.shape, a.min(), a.max(), a.dtype
(10, 10, 3), 0.0, 254.0, dtype('float32'), 
>>> b.shape, b.min(), b.max(), b.dtype
torch.Size([3, 10, 10]), tensor(0.), tensor(254.), torch.float32


But if the input is a tensor, it will not change the shape

>>> a = torch.randint(0, 255, size=(5, 5, 3), dtype=torch.uint8);  b = v2.ToImage()(a)
>>> a.shape, a.min(), a.max(), a.dtype
torch.Size([5, 5, 3]), tensor(1, dtype=torch.uint8), tensor(247, dtype=torch.uint8), torch.uint8
>>> b.shape, b.min(), b.max(), b.dtype
torch.Size([5, 5, 3]), tensor(1, dtype=torch.uint8), tensor(247, dtype=torch.uint8), torch.uint8

>>> a = torch.randint(0, 255, size=(5, 5, 3), dtype=torch.float32);  b = v2.ToImage()(a)
>>> a.shape, a.min(), a.max(), a.dtype
torch.Size([5, 5, 3]), tensor(6.), tensor(253.), torch.float32
>>> b.shape, b.min(), b.max(), b.dtype
torch.Size([5, 5, 3]), tensor(6.), tensor(253.), torch.float32


If the input is a np.ndarray grayscale image, then it must be (H, W, 1) and not (H, W).
"""