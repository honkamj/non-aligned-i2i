"""NDimensional helpers for building NDimensional models"""


import torch.nn

def conv_nd(n_dims: int):
    """Return any dimensional convolution module"""
    return getattr(torch.nn, f'Conv{n_dims}d')


def conv_transpose_nd(n_dims: int):
    """Return any dimensional tranpose convolution module"""
    return getattr(torch.nn, f'ConvTranspose{n_dims}d')


def avg_pool_nd(n_dims: int):
    """Return any dimensional average pooling module"""
    return getattr(torch.nn, f'AvgPool{n_dims}d')


def max_pool_nd(n_dims: int):
    """Return any dimensional max pooling module"""
    return getattr(torch.nn, f'MaxPool{n_dims}d')
