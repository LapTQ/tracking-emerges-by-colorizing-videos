import torch
import torchvision.transforms.v2 as transforms


class CustomTransform(transforms.ToDtype):

    def __init__(self, **kwargs):
        
        kwargs['dtype'] = eval(kwargs['dtype'])

        super().__init__(**kwargs)


"""
v2.ToDtype will convert the input to the specified dtype.
If scale is set to True, then the input will be scaled according 
to the specified dtype. Note that dtype must be pytorch native dtype
such as torch.float32 and not string such as 'float32'.

For example, if the input is uint8 [0, 255], if set float32 scale=False,
then the output will be [0., 255.].
>>> a = np.random.randint(0, 255, size=(10, 10, 3)).astype('uint8')
>>> a = v2.ToImage()(a); b = v2.ToDtype(dtype=torch.float32, scale=False)(a)
>>> type(a), a.shape, a.min(), a.max(), a.dtype
<class 'torchvision.tv_tensors._image.Image'>, torch.Size([3, 10, 10]), tensor(0, dtype=torch.uint8), tensor(252, dtype=torch.uint8), torch.uint8
>>> type(b), b.shape, b.min(), b.max(), b.dtype
<class 'torchvision.tv_tensors._image.Image'>, torch.Size([3, 10, 10]), tensor(0.), tensor(252.), torch.float32

But if set scale=True, then the output will be [0., 1.].
>>> a = np.random.randint(0, 255, size=(10, 10, 3)).astype('uint8')
>>> a = v2.ToImage()(a); b = v2.ToDtype(dtype=torch.float32, scale=True)(a)
>>> type(a), a.shape, a.min(), a.max(), a.dtype
<class 'torchvision.tv_tensors._image.Image'>, torch.Size([3, 10, 10]), tensor(2, dtype=torch.uint8), tensor(254, dtype=torch.uint8), torch.uint8
>>> type(b), b.shape, b.min(), b.max(), b.dtype
<class 'torchvision.tv_tensors._image.Image'>, torch.Size([3, 10, 10]), tensor(0.0078), tensor(0.9961), torch.float32


Similarly, if the input is float32 [0., 1.], if set uint8 scale=False,
then the output will be [0, 1].
If set scale=True, then the output will be [0, 255].
And if the input is float32 [0., 255.], if set uint8 scale=True,
then the output can be [255, 255] or [0, 255] depending on the min...,
while the max cannot exceed 255.
"""